package luma

import "core:math/bits"
import vk "vendor:vulkan"


Command_Handle :: struct {
	buffer_idx: u8,
	id:         u32, // 0 means unused
}

Command_Buffer :: struct {
	// allocation
	buffer:         vk.CommandBuffer,
	fence:          vk.Fence,
	semaphore:      vk.Semaphore,
	// usage
	current_handle: Command_Handle,
}

Command_Handler :: struct {
	device:                 vk.Device,
	queue:                  vk.Queue,
	command_pool:           vk.CommandPool,
	buffers:                [MAX_COMMAND_BUFFERS]Command_Buffer,
	latest_submission:      Command_Handle,
	wait_semaphore:         vk.Semaphore,
	final_signal:           struct {
		semaphore: vk.Semaphore,
		value:     u64,
	},
	available_buffer_count: u32,
	submission_counter:     u32,
}

MAX_COMMAND_BUFFERS: u8 : 8

command_handler_init :: proc(
	ch: ^Command_Handler,
	device: vk.Device,
	queue: vk.Queue,
	queue_family_idx: u32,
) {
	ch.device = device
	ch.queue = queue
	// build command buffers and their associated synch objects
	command_pool_ci := vk.CommandPoolCreateInfo {
		sType            = .COMMAND_POOL_CREATE_INFO,
		flags            = {.RESET_COMMAND_BUFFER, .TRANSIENT},
		queueFamilyIndex = queue_family_idx,
	}
	chk(vk.CreateCommandPool(device, &command_pool_ci, nil, &ch.command_pool))

	semaphore_ci := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
	}
	fence_ci := vk.FenceCreateInfo {
		sType = .FENCE_CREATE_INFO,
		flags = {.SIGNALED},
	}
	buffer_ai := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = ch.command_pool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}
	for &buf in ch.buffers {
		chk(vk.AllocateCommandBuffers(device, &buffer_ai, &buf.buffer))
		chk(vk.CreateFence(device, &fence_ci, nil, &buf.fence))
		chk(vk.CreateSemaphore(device, &semaphore_ci, nil, &buf.semaphore))
	}
	ch.available_buffer_count = len(ch.buffers)
}

command_handler_cleanup :: proc(ch: ^Command_Handler) {
	for &buf in ch.buffers {
		if buf.fence != 0 {
			vk.DestroyFence(ch.device, buf.fence, nil)
		}
		if buf.semaphore != 0 {
			vk.DestroySemaphore(ch.device, buf.semaphore, nil)
		}
		if buf.buffer != nil {
			vk.FreeCommandBuffers(ch.device, ch.command_pool, 1, &buf.buffer)
		}
	}
	vk.DestroyCommandPool(ch.device, ch.command_pool, nil)
}

command_handler_acquire :: proc(
	ch: ^Command_Handler,
) -> (
	handle: Command_Handle,
	cb: vk.CommandBuffer,
) {
	for ch.available_buffer_count == 0 {
		purge(ch)
	}
	// search for the buffer to acquire
	for &buf, i in ch.buffers {
		if buf.current_handle.id != 0 do continue

		ch.available_buffer_count -= 1
		begin_info := vk.CommandBufferBeginInfo {
			sType = .COMMAND_BUFFER_BEGIN_INFO,
			flags = {.ONE_TIME_SUBMIT},
		}
		chk(vk.BeginCommandBuffer(buf.buffer, &begin_info))

		return {buffer_idx = u8(i)}, buf.buffer
	}

	return {}, {}
}

command_handler_submit :: proc(
	ch: ^Command_Handler,
	handle: Command_Handle,
	signal_final_semaphore: bool,
) {
	ch.submission_counter += 1
	buf := &ch.buffers[handle.buffer_idx]
	buf.current_handle = {
		buffer_idx = handle.buffer_idx,
		id         = ch.submission_counter,
	}
	chk(vk.EndCommandBuffer(buf.buffer))

	wait_infos := [?]vk.SemaphoreSubmitInfo {
		{sType = .SEMAPHORE_SUBMIT_INFO, stageMask = {.ALL_COMMANDS}},
		{sType = .SEMAPHORE_SUBMIT_INFO, stageMask = {.ALL_COMMANDS}},
	}
	num_wait_semaphores: u32 = 0
	if ch.wait_semaphore != 0 {
		wait_infos[num_wait_semaphores].semaphore = ch.wait_semaphore
		num_wait_semaphores += 1
		ch.wait_semaphore = 0
	}
	if ch.latest_submission.id != 0 {
		prev_sem := ch.buffers[ch.latest_submission.buffer_idx].semaphore
		wait_infos[num_wait_semaphores].semaphore = prev_sem
		num_wait_semaphores += 1
	}

	signal_infos := [?]vk.SemaphoreSubmitInfo {
		{sType = .SEMAPHORE_SUBMIT_INFO, stageMask = {.ALL_COMMANDS}, semaphore = buf.semaphore},
		{sType = .SEMAPHORE_SUBMIT_INFO, stageMask = {.ALL_COMMANDS}},
	}
	num_signal_semaphores: u32 = 1
	if signal_final_semaphore && ch.final_signal.semaphore != 0 {
		signal_infos[num_signal_semaphores].semaphore = ch.final_signal.semaphore
		signal_infos[num_signal_semaphores].value = ch.final_signal.value
		num_signal_semaphores += 1
		ch.final_signal = {}
	}

	submit_info := vk.SubmitInfo2 {
		sType                    = .SUBMIT_INFO_2,
		commandBufferInfoCount   = 1,
		pCommandBufferInfos      = &vk.CommandBufferSubmitInfo {
			sType = .COMMAND_BUFFER_SUBMIT_INFO,
			commandBuffer = buf.buffer,
		},
		waitSemaphoreInfoCount   = num_wait_semaphores,
		pWaitSemaphoreInfos      = raw_data(&wait_infos),
		signalSemaphoreInfoCount = num_signal_semaphores,
		pSignalSemaphoreInfos    = raw_data(&signal_infos),
	}
	vk.ResetFences(ch.device, 1, &buf.fence)
	chk(vk.QueueSubmit2(ch.queue, 1, &submit_info, buf.fence))

	ch.latest_submission = buf.current_handle
}

command_handler_wait :: proc(ch: ^Command_Handler, handle: Command_Handle) {
	fence := ch.buffers[handle.buffer_idx].fence
	chk(vk.WaitForFences(ch.device, 1, &fence, true, bits.U64_MAX))
	purge(ch)
}

command_handler_request_semaphore_wait :: proc(ch: ^Command_Handler, semaphore: vk.Semaphore) {
	ch.wait_semaphore = semaphore
}

command_handler_write_final_signal :: proc(
	ch: ^Command_Handler,
	semaphore: vk.Semaphore,
	timeline_value: u64,
) {
	ch.final_signal = {
		semaphore = semaphore,
		value     = timeline_value,
	}
}

command_handler_get_latest_submission :: proc(
	ch: ^Command_Handler,
) -> (
	Command_Handle,
	vk.Semaphore,
) {
	return ch.latest_submission, ch.buffers[ch.latest_submission.buffer_idx].semaphore
}

@(private = "file")
purge :: proc(ch: ^Command_Handler) {
	// wait a tick for all the buffer fences
	n_buffers: u8 = len(ch.buffers)
	for i: u8 = 0; i < n_buffers; i += 1 {
		idx := (i + ch.latest_submission.buffer_idx + 1) % n_buffers
		buf := &ch.buffers[idx]
		if buf.current_handle.id != 0 {
			fence_res := vk.WaitForFences(ch.device, 1, &buf.fence, true, 0)
			// buffer is ready, reset and mark available
			if fence_res == .SUCCESS {
				vk.ResetCommandBuffer(buf.buffer, nil)
				vk.ResetFences(ch.device, 1, &buf.fence)
				buf.current_handle = {}
				ch.available_buffer_count += 1
			} else if fence_res != .TIMEOUT {
				chk(fence_res)
			}
		}
	}
}
