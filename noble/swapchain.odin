package noble

import "core:fmt"
import "core:math/bits"
import vk "vendor:vulkan"

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
