package luma

import "core:fmt"
import "core:math/bits"
import "vendor:glfw"
import vk "vendor:vulkan"

DEFAULT_PREFERRED_SURFACE_FORMAT :: vk.SurfaceFormatKHR {
	format     = .R8G8B8A8_UNORM,
	colorSpace = .SRGB_NONLINEAR,
}
DEFAULT_PREFERRED_PRESENT_MODE :: vk.PresentModeKHR.IMMEDIATE

chk_swapchain :: proc(result: vk.Result, update_swapchain: ^bool) {
	if result < .SUCCESS {
		if result == .ERROR_OUT_OF_DATE_KHR {
			update_swapchain^ = true
			return
		}
		fmt.panicf("[Vulkan-Swapchain] Vulkan Failure: %s", result)
	}
}


Swapchain :: struct {
	device:               ^Device,
	surface:              vk.SurfaceKHR,
	format:               vk.Format,
	swapchain:            vk.SwapchainKHR,
	// store the create info for easier update
	create_info:          vk.SwapchainCreateInfoKHR,
	images:               []vk.Image,
	image_views:          []vk.ImageView,
	acquire_semaphores:   []vk.Semaphore,
	timeline_semaphore:   vk.Semaphore,
	frame_idx:            u64,
	timeline_wait_values: []u64,
	current_image_idx:    u32,
	need_acquire:         bool,
	need_update:          bool,
}

Swapchain_Image :: struct {
	image: vk.Image,
	view:  vk.ImageView,
}

Swapchain_Desc :: struct {
	preferred_surface_format: Maybe(vk.SurfaceFormatKHR),
	preferred_present_mode:   Maybe(vk.PresentModeKHR),
}

// swapchain is handled individually in order to
// allow using the abstraction without a window (for offline tasks for example)
create_swapchain :: proc(
	d: ^Device,
	window: glfw.WindowHandle,
	desc: Swapchain_Desc,
) -> (
	swapchain: Swapchain,
) {
	avbailable_surface_formats: []vk.SurfaceFormatKHR
	available_present_modes: []vk.PresentModeKHR
	surface: vk.SurfaceKHR
	surface_caps: vk.SurfaceCapabilitiesKHR
	// surface capabilities
	{
		depth_formats := [?]vk.Format {
			.D32_SFLOAT_S8_UINT,
			.D24_UNORM_S8_UINT,
			.D16_UNORM_S8_UINT,
			.D32_SFLOAT,
			.D16_UNORM,
		}
		d.available_depth_formats = make([dynamic]vk.Format)
		for format in depth_formats {
			format_props := vk.FormatProperties2 {
				sType = .FORMAT_PROPERTIES_2,
			}
			vk.GetPhysicalDeviceFormatProperties2(d.physical_device, format, &format_props)
			if vk.FormatFeatureFlag.DEPTH_STENCIL_ATTACHMENT in
			   format_props.formatProperties.optimalTilingFeatures {
				append(&d.available_depth_formats, format)
				break
			}
		}

		chk(glfw.CreateWindowSurface(d.instance, window, nil, &surface))

		vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(d.physical_device, surface, &surface_caps)
		format_count: u32
		vk.GetPhysicalDeviceSurfaceFormatsKHR(d.physical_device, surface, &format_count, nil)
		vk.GetPhysicalDeviceSurfaceFormatsKHR(d.physical_device, surface, &format_count, nil)
		if format_count == 0 do fmt.panicf("[Swapchain] No surface formats available")
		avbailable_surface_formats = make(
			[]vk.SurfaceFormatKHR,
			format_count,
			context.temp_allocator,
		)
		vk.GetPhysicalDeviceSurfaceFormatsKHR(
			d.physical_device,
			surface,
			&format_count,
			raw_data(avbailable_surface_formats),
		)

		mode_count: u32
		vk.GetPhysicalDeviceSurfacePresentModesKHR(d.physical_device, surface, &mode_count, nil)
		if mode_count == 0 do fmt.panicf("[Swapchain] No present modes available")
		available_present_modes = make([]vk.PresentModeKHR, mode_count, context.temp_allocator)
		vk.GetPhysicalDeviceSurfacePresentModesKHR(
			d.physical_device,
			surface,
			&mode_count,
			raw_data(available_present_modes),
		)
	}

	// swapchain
	{
		/*
		Try getting the preffered format + color space,
		If you can't, try getting the format at least,
		If this also fails, get the first available format
		*/
		surface_format, same := choose_surface_format(
			desc.preferred_surface_format.? or_else DEFAULT_PREFERRED_SURFACE_FORMAT,
			avbailable_surface_formats,
		)
		if !same {
			fmt.printfln(
				"[Device] Preferred surface format was not available. %v has been chosen",
				surface_format,
			)
		}

		// preferred -> immediate -> fifo
		present_mode, same_ := choose_present_mode(
			desc.preferred_present_mode.? or_else DEFAULT_PREFERRED_PRESENT_MODE,
			available_present_modes,
		)
		if !same_ {
			fmt.printfln(
				"[Device] Preferred presentation mode was not available. %v has been chosen",
				present_mode,
			)
		}

		desired_image_count := surface_caps.minImageCount + 1
		image_count := max(surface_caps.maxImageCount, desired_image_count)

		swapchain.device = d

		// inlined swapchain_init
		swapchain.surface = surface
		swapchain.format = surface_format.format
		family_idx := swapchain.device.queues.graphics_family_idx
		swapchain.create_info = vk.SwapchainCreateInfoKHR {
			sType                 = .SWAPCHAIN_CREATE_INFO_KHR,
			surface               = surface,
			minImageCount         = image_count,
			imageFormat           = surface_format.format,
			imageColorSpace       = surface_format.colorSpace,
			imageExtent           = surface_caps.currentExtent,
			imageArrayLayers      = 1,
			imageUsage            = {.COLOR_ATTACHMENT},
			imageSharingMode      = .EXCLUSIVE,
			preTransform          = {.IDENTITY},
			queueFamilyIndexCount = 1,
			pQueueFamilyIndices   = &family_idx,
			compositeAlpha        = {.OPAQUE},
			presentMode           = present_mode,
			clipped               = true,
		}
		chk(
			vk.CreateSwapchainKHR(
				swapchain.device.device,
				&swapchain.create_info,
				nil,
				&swapchain.swapchain,
			),
		)

		// for each swapchain image, store the acquire semaphore and
		// the command buffer submit handle which draws to the image
		swapchain_image_count: u32
		chk(
			vk.GetSwapchainImagesKHR(
				swapchain.device.device,
				swapchain.swapchain,
				&swapchain_image_count,
				nil,
			),
		)
		swapchain.images = make([]vk.Image, swapchain_image_count)
		chk(
			vk.GetSwapchainImagesKHR(
				swapchain.device.device,
				swapchain.swapchain,
				&swapchain_image_count,
				raw_data(swapchain.images),
			),
		)
		swapchain.image_views = make([]vk.ImageView, swapchain_image_count)
		semaphore_ci := vk.SemaphoreCreateInfo {
			sType = .SEMAPHORE_CREATE_INFO,
		}
		swapchain.acquire_semaphores = make([]vk.Semaphore, swapchain_image_count)
		swapchain.timeline_wait_values = make([]u64, swapchain_image_count)
		for i in 0 ..< len(swapchain.images) {
			view_ci := vk.ImageViewCreateInfo {
				sType = .IMAGE_VIEW_CREATE_INFO,
				image = swapchain.images[i],
				viewType = .D2,
				format = surface_format.format,
				subresourceRange = {
					aspectMask = {.COLOR},
					baseMipLevel = 0,
					levelCount = 1,
					baseArrayLayer = 0,
					layerCount = 1,
				},
			}
			chk(
				vk.CreateImageView(
					swapchain.device.device,
					&view_ci,
					nil,
					&swapchain.image_views[i],
				),
			)
			chk(
				vk.CreateSemaphore(
					swapchain.device.device,
					&semaphore_ci,
					nil,
					&swapchain.acquire_semaphores[i],
				),
			)
		}

		timeline_ci := vk.SemaphoreTypeCreateInfo {
			sType         = .SEMAPHORE_TYPE_CREATE_INFO,
			semaphoreType = .TIMELINE,
			initialValue  = 0,
		}
		semaphore_ci.pNext = &timeline_ci
		chk(
			vk.CreateSemaphore(
				swapchain.device.device,
				&semaphore_ci,
				nil,
				&swapchain.timeline_semaphore,
			),
		)
		swapchain.need_acquire = true
	}

	return swapchain
}

// swapchain image is treated as GENERAL the rest of the time, this just
// flips it to PRESENT_SRC_KHR right before presenting since Vulkan requires it
swapchain_barrier_to_present :: proc(cb: vk.CommandBuffer, img: Swapchain_Image) {
	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		srcAccessMask = {.COLOR_ATTACHMENT_WRITE},
		oldLayout = .UNDEFINED,
		newLayout = .PRESENT_SRC_KHR,
		image = img.image,
		subresourceRange = {aspectMask = {.COLOR}, levelCount = 1, layerCount = 1},
	}
	dep := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}
	vk.CmdPipelineBarrier2(cb, &dep)
}

swapchain_cleanup :: proc(s: ^Swapchain) {
	for view in s.image_views {
		vk.DestroyImageView(s.device.device, view, nil)
	}
	for sem in s.acquire_semaphores {
		vk.DestroySemaphore(s.device.device, sem, nil)
	}
	vk.DestroySemaphore(s.device.device, s.timeline_semaphore, nil)
	delete(s.image_views)
	delete(s.images)
	delete(s.acquire_semaphores)
	delete(s.timeline_wait_values)

	vk.DestroySwapchainKHR(s.device.device, s.swapchain, nil)
	vk.DestroySurfaceKHR(s.device.instance, s.surface, nil)
}

swapchain_acquire_image :: proc(s: ^Swapchain) -> (image: Swapchain_Image) {
	if s.need_acquire {
		wait_info := vk.SemaphoreWaitInfo {
			sType          = .SEMAPHORE_WAIT_INFO,
			semaphoreCount = 1,
			pSemaphores    = &s.timeline_semaphore,
			pValues        = &s.timeline_wait_values[s.current_image_idx],
		}
		vk.WaitSemaphores(s.device.device, &wait_info, bits.U64_MAX)

		acquire_sem := s.acquire_semaphores[s.current_image_idx]
		chk_swapchain(
			vk.AcquireNextImageKHR(
				s.device.device,
				s.swapchain,
				bits.U64_MAX,
				acquire_sem,
				0,
				&s.current_image_idx,
			),
			&s.need_update,
		)
		s.need_acquire = false
		command_handler_request_semaphore_wait(&s.device.command_handler, acquire_sem)

		signal_value := s.frame_idx + u64(len(s.images))
		s.timeline_wait_values[s.current_image_idx] = signal_value
		command_handler_write_final_signal(
			&s.device.command_handler,
			s.timeline_semaphore,
			signal_value,
		)

		s.frame_idx += 1
	}

	return {image = s.images[s.current_image_idx], view = s.image_views[s.current_image_idx]}
}

swapchain_present :: proc(s: ^Swapchain) {
	fmt.assertf(
		!s.need_acquire,
		"[Swapchain] ASSERT Present called without a prior acquire." +
		"You must call swapchain_get_draw_texture() before presenting a frame.",
	)

	latest_handle, latest_semaphore := command_handler_get_latest_submission(
		&s.device.command_handler,
	)
	fmt.assertf(
		latest_semaphore != 0,
		"[Swapchain] ASSERT Present called but no GPU work was submitted." +
		"You must record and submit at least one command buffer before presenting.",
	)

	present_info := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &latest_semaphore,
		swapchainCount     = 1,
		pSwapchains        = &s.swapchain,
		pImageIndices      = &s.current_image_idx,
	}
	chk_swapchain(vk.QueuePresentKHR(s.device.queues.graphics, &present_info), &s.need_update)
	s.need_acquire = true
	s.device.command_handler.latest_submission = {}

	// TODO: handle update
	if s.need_update do return
}

@(private = "file")
choose_surface_format :: proc(
	preferred: vk.SurfaceFormatKHR,
	available: []vk.SurfaceFormatKHR,
) -> (
	format: vk.SurfaceFormatKHR,
	same: bool,
) {
	for format in available {
		if preferred.format == format.format && preferred.colorSpace == format.colorSpace {
			return format, true
		}
	}
	for format in available {
		if preferred.format == format.format {
			return format, false
		}
	}
	return available[0], false
}

@(private = "file")
choose_present_mode :: proc(
	preferred: vk.PresentModeKHR,
	available: []vk.PresentModeKHR,
) -> (
	format: vk.PresentModeKHR,
	same: bool,
) {
	for mode in available {
		if mode == preferred {
			return mode, true
		}
	}
	for mode in available {
		if mode == .IMMEDIATE {
			return mode, false
		}
	}
	return .FIFO, false
}
