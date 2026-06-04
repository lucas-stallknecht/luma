package noble

import "core:fmt"
import "vendor:glfw"
import vk "vendor:vulkan"

// Globals set by device_init
g_enable_logging: bool


chk :: proc(result: vk.Result, location := #caller_location) {
	if result != .SUCCESS {
		fmt.panicf("[Vulkan] Failure %s: %s", location, result)
	}
}

// ────────────────────────────────────────────────────────────────
// Vulkan Context

Device :: struct {
	instance:                vk.Instance,
	physical_device:         vk.PhysicalDevice,
	queues:                  struct {
		graphics:            vk.Queue,
		graphics_family_idx: u32,
		compute:             vk.Queue,
		compute_family_idx:  u32,
	},
	device:                  vk.Device,
	available_depth_formats: [dynamic]vk.Format,
	descriptor_pool:         vk.DescriptorPool,
	descriptor_layout:       vk.DescriptorSetLayout,
	descriptor_set:          vk.DescriptorSet,
	command_handler:         Command_Handler,
}

MAX_BINDLESS_IMAGES :: 1000
DEFAULT_PHYSICAL_DEVICE_SELECTION_FN :: proc(
	idx: int,
	properties: vk.PhysicalDeviceProperties2,
) -> bool {
	return true
}


Device_Desc :: struct {
	enable_logging:               bool,
	enable_validation:            bool,
	shared_types_file_path:       string,
	physical_device_selection_fn: Maybe(
		proc(idx: int, properties: vk.PhysicalDeviceProperties2) -> bool,
	),
}

device_init :: proc(d: ^Device, desc: Device_Desc) {
	g_enable_logging = desc.enable_logging
	// instance
	{
		vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))
		glfw_extensions := glfw.GetRequiredInstanceExtensions()

		layers := [?]cstring{"VK_LAYER_KHRONOS_validation"}

		extensions := make([dynamic]cstring, context.temp_allocator)
		append(&extensions, ..glfw_extensions)

		instance_ci := vk.InstanceCreateInfo {
			sType            = .INSTANCE_CREATE_INFO,
			pApplicationInfo = &{sType = .APPLICATION_INFO, apiVersion = vk.API_VERSION_1_3},
		}
		if desc.enable_validation {
			instance_ci.enabledLayerCount = u32(len(layers))
			instance_ci.ppEnabledLayerNames = raw_data(&layers)
			append(&extensions, vk.EXT_VALIDATION_FEATURES_EXTENSION_NAME)
		} else {
			instance_ci.enabledLayerCount = 0
		}

		if desc.enable_logging {
			fmt.println("[Device] Instance extensions", extensions[:])
		}
		instance_ci.enabledExtensionCount = u32(len(extensions))
		instance_ci.ppEnabledExtensionNames = raw_data(extensions)

		chk(vk.CreateInstance(&instance_ci, nil, &d.instance))
		vk.load_proc_addresses_instance(d.instance)
		if vk.EnumeratePhysicalDevices == nil do fmt.panicf("[Device] Failed to load instance functions")
	}

	// physical Device
	{
		physical_device_count: u32
		chk(vk.EnumeratePhysicalDevices(d.instance, &physical_device_count, nil))
		if physical_device_count == 0 do fmt.panicf("[Device] No suitable physical devices found")
		physical_devices := make(
			[]vk.PhysicalDevice,
			physical_device_count,
			context.temp_allocator,
		)
		chk(
			vk.EnumeratePhysicalDevices(
				d.instance,
				&physical_device_count,
				raw_data(physical_devices),
			),
		)
		for pd, i in physical_devices {
			device_properties := vk.PhysicalDeviceProperties2 {
				sType = .PHYSICAL_DEVICE_PROPERTIES_2,
			}
			vk.GetPhysicalDeviceProperties2(pd, &device_properties)

			selection_fn :=
				desc.physical_device_selection_fn.? or_else DEFAULT_PHYSICAL_DEVICE_SELECTION_FN
			if !selection_fn(i, device_properties) do continue

			d.physical_device = pd
			if desc.enable_logging {
				fmt.printfln(
					"[Device] Selected device: %s",
					device_properties.properties.deviceName,
				)
			}
			break
		}
		if d.physical_device == nil do fmt.panicf("[Device] No suitable physical device selected")
	}

	// queues
	{
		queue_family_count: u32
		vk.GetPhysicalDeviceQueueFamilyProperties(d.physical_device, &queue_family_count, nil)
		queue_families := make(
			[]vk.QueueFamilyProperties,
			queue_family_count,
			context.temp_allocator,
		)
		vk.GetPhysicalDeviceQueueFamilyProperties(
			d.physical_device,
			&queue_family_count,
			raw_data(queue_families),
		)

		for i in 0 ..< queue_family_count {
			if vk.QueueFlag.GRAPHICS in queue_families[i].queueFlags {
				d.queues.graphics_family_idx = i
				break
			}
			if vk.QueueFlag.COMPUTE in queue_families[i].queueFlags {
				d.queues.compute_family_idx = i
				break
			}
		}
	}

	// logical Device
	{
		queue_priority: f32 = 1.0
		queue_ci := [?]vk.DeviceQueueCreateInfo {
			{
				sType = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = d.queues.graphics_family_idx,
				queueCount = 1,
				pQueuePriorities = &queue_priority,
			},
			{
				sType = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = d.queues.graphics_family_idx,
				queueCount = 1,
				pQueuePriorities = &queue_priority,
			},
		}
		num_queues := 1 if d.queues.graphics_family_idx == d.queues.compute_family_idx else 2

		device_extensions := [?]cstring{"VK_KHR_swapchain", "VK_EXT_extended_dynamic_state3"}
		vk11_features := vk.PhysicalDeviceVulkan11Features {
			sType                = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
			shaderDrawParameters = true,
		}
		vk12_features := vk.PhysicalDeviceVulkan12Features {
			sType                                    = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			pNext                                    = &vk11_features,
			descriptorIndexing                       = true,
			descriptorBindingVariableDescriptorCount = true,
			runtimeDescriptorArray                   = true,
			bufferDeviceAddress                      = true,
			timelineSemaphore                        = true,
		}
		vk13_features := vk.PhysicalDeviceVulkan13Features {
			sType            = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
			pNext            = &vk12_features,
			synchronization2 = true,
			dynamicRendering = true,
		}
		vk10_features := vk.PhysicalDeviceFeatures {
			samplerAnisotropy = true,
			shaderInt64       = true,
		}

		device_ci := vk.DeviceCreateInfo {
			sType                   = .DEVICE_CREATE_INFO,
			pNext                   = &vk13_features,
			queueCreateInfoCount    = u32(num_queues),
			pQueueCreateInfos       = raw_data(&queue_ci),
			enabledExtensionCount   = u32(len(device_extensions)),
			ppEnabledExtensionNames = raw_data(&device_extensions),
			pEnabledFeatures        = &vk10_features,
		}
		chk(vk.CreateDevice(d.physical_device, &device_ci, nil, &d.device))
		vk.load_proc_addresses_device(d.device)
		if vk.BeginCommandBuffer == nil do fmt.panicf("[Device] Failed to load device functions")
		vk.GetDeviceQueue(d.device, d.queues.compute_family_idx, 0, &d.queues.compute)
		vk.GetDeviceQueue(d.device, d.queues.compute_family_idx, 0, &d.queues.graphics)
	}

	command_handler_init(
		&d.command_handler,
		d.device,
		d.queues.graphics,
		d.queues.graphics_family_idx,
	)
	free_all(context.temp_allocator)


	// bindless Descriptors
	{
		desc_pool_sizes := [?]vk.DescriptorPoolSize {
			{type = .SAMPLED_IMAGE, descriptorCount = MAX_BINDLESS_IMAGES},
		}
		desc_pool_ci := vk.DescriptorPoolCreateInfo {
			sType         = .DESCRIPTOR_POOL_CREATE_INFO,
			maxSets       = 3,
			poolSizeCount = len(desc_pool_sizes),
			pPoolSizes    = raw_data(&desc_pool_sizes),
		}
		chk(vk.CreateDescriptorPool(d.device, &desc_pool_ci, nil, &d.descriptor_pool))

		desc_variable_flags := vk.DescriptorBindingFlags{.VARIABLE_DESCRIPTOR_COUNT}
		desc_binding_flags := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
			sType         = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
			bindingCount  = 1,
			pBindingFlags = &desc_variable_flags,
		}
		desc_layout_bindings := vk.DescriptorSetLayoutBinding {
			descriptorType  = .SAMPLED_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES,
			stageFlags      = {.VERTEX, .FRAGMENT, .COMPUTE},
		}
		desc_layout_ci := vk.DescriptorSetLayoutCreateInfo {
			sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			pNext        = &desc_binding_flags,
			bindingCount = 1,
			pBindings    = &desc_layout_bindings,
		}
		chk(vk.CreateDescriptorSetLayout(d.device, &desc_layout_ci, nil, &d.descriptor_layout))

		variable_desc_count: u32 = MAX_BINDLESS_IMAGES
		variable_desc_ai := vk.DescriptorSetVariableDescriptorCountAllocateInfo {
			sType              = .DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
			descriptorSetCount = 1,
			pDescriptorCounts  = &variable_desc_count,
		}
		descriptor_set_ai := vk.DescriptorSetAllocateInfo {
			sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
			pNext              = &variable_desc_ai,
			descriptorPool     = d.descriptor_pool,
			descriptorSetCount = 1,
			pSetLayouts        = &d.descriptor_layout,
		}
		chk(vk.AllocateDescriptorSets(d.device, &descriptor_set_ai, &d.descriptor_set))
	}

}

device_cleanup :: proc(d: ^Device) {
	command_handler_cleanup(&d.command_handler)

	vk.DestroyDescriptorSetLayout(d.device, d.descriptor_layout, nil)
	vk.DestroyDescriptorPool(d.device, d.descriptor_pool, nil)

	delete(d.available_depth_formats)
	vk.DestroyDevice(d.device, nil)
	vk.DestroyInstance(d.instance, nil)
}

// ────────────────────────────────────────────────────────────────
// Objects creation

DEFAULT_PREFERRED_SURFACE_FORMAT :: vk.SurfaceFormatKHR {
	format     = .R8G8B8A8_UNORM,
	colorSpace = .SRGB_NONLINEAR,
}
DEFAULT_PREFERRED_PRESENT_MODE :: vk.PresentModeKHR.IMMEDIATE


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
		if !same && g_enable_logging {
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
		if !same_ && g_enable_logging {
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

create_pipeline_manager :: proc(
	d: ^Device,
	shader_directory: string,
	compile_shader_directory: string,
) -> Pipeline_Manager {
	return {
		device = d,
		raster_pipelines = make(map[string]Raster_Pipeline),
		shader_directory = shader_directory,
		compile_shader_directory = compile_shader_directory,
	}
}

// ────────────────────────────────────────────────────────────────
// Command helpers

bind_pipeline :: proc(cb: vk.CommandBuffer, pipeline: ^Raster_Pipeline) {
	vk.CmdBindPipeline(cb, .GRAPHICS, pipeline.pipeline)
	vk.CmdBindDescriptorSets(cb, .GRAPHICS, pipeline.layout, 0, 1, &pipeline.bindless_set, 0, nil)
}

// ────────────────────────────────────────────────────────────────

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
