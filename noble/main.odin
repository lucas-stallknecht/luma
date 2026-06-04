package noble

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "vendor:glfw"
import vk "vendor:vulkan"

device_selection_fn :: proc(idx: int, properties: vk.PhysicalDeviceProperties2) -> bool {
	return idx == 0
}

handle_camera_inputs :: proc(win: ^Window, cam: ^Camera, dt: f32) {
	camera_rotate(cam, window_consume_mouse_delta(win))

	if win.pressed_keys[glfw.KEY_W] {
		camera_move_forward(cam, dt)
	}
	if win.pressed_keys[glfw.KEY_S] {
		camera_move_forward(cam, -dt)
	}
	if win.pressed_keys[glfw.KEY_D] {
		camera_move_right(cam, dt)
	}
	if win.pressed_keys[glfw.KEY_A] {
		camera_move_right(cam, -dt)
	}
	if win.pressed_keys[glfw.KEY_SPACE] {
		camera_move_up(cam, dt)
	}
	if win.pressed_keys[glfw.KEY_LEFT_CONTROL] {
		camera_move_up(cam, -dt)
	}
}

main :: proc() {
	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				for _, entry in track.allocation_map {
					fmt.printfln("%v leaked %v bytes", entry.location, entry.size)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}

	window: Window
	if !window_init(&window) do return
	defer window_cleanup(&window)

	device: Device
	device_init(
		&device,
		{
			enable_logging = true,
			enable_validation = true,
			shared_types_file_path = #directory + "shared_types.odin",
			physical_device_selection_fn = device_selection_fn,
		},
	)
	defer device_cleanup(&device)

	swapchain := create_swapchain(&device, window.glfw_window_ptr, {})
	defer swapchain_cleanup(&swapchain)

	pipeline_manager := create_pipeline_manager(&device, "shaders/", "shaders/compiled/")
	defer pipeline_manager_cleanup(&pipeline_manager)

	triangle_pipeline := pipeline_manager_add_raster(
		&pipeline_manager,
		{
			name = "triangle",
			vertex_shader = "triangle.vert",
			fragment_shader = "triangle.frag",
			raster = {primitive_topology = .TRIANGLE_LIST},
			push_constant_size = 16 * 4,
			color_attachments = {{format = swapchain.format}},
		},
	)

	camera := create_camera()
	camera_update_proj(&camera, f32(window.width) / f32(window.height))

	last_frame_time: f64 = glfw.GetTime()
	reload_key_prev: bool = false

	for !window_should_close(&window) {
		window_update(&window)
		time := glfw.GetTime()
		dt := f32(time - last_frame_time)
		last_frame_time = time

		handle_camera_inputs(&window, &camera, dt)
		if window.pressed_keys[glfw.KEY_R] && !reload_key_prev {
			if pipeline_reload_all(&pipeline_manager) {
				fmt.println("Shaders successfully reloaded")
			}
		}
		reload_key_prev = window.pressed_keys[glfw.KEY_R]

		if window.resized {
			camera_update_proj(&camera, f32(window.width) / f32(window.height))
			// TODO: resize other things
			window.resized = false
		}

		width := window.width
		height := window.height

		// render loop
		swapchain_image := swapchain_acquire_image(&swapchain)
		handle, cb := command_handler_acquire(&device.command_handler)

		rend_barrier := vk.ImageMemoryBarrier2 {
			sType = .IMAGE_MEMORY_BARRIER_2,
			srcStageMask = {},
			srcAccessMask = {},
			dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
			dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
			oldLayout = .UNDEFINED,
			newLayout = .COLOR_ATTACHMENT_OPTIMAL,
			image = swapchain_image.image,
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				levelCount = 1,
				layerCount = 1,
			},
		}
		rend_dep := vk.DependencyInfo {
			sType                   = .DEPENDENCY_INFO,
			imageMemoryBarrierCount = 1,
			pImageMemoryBarriers    = &rend_barrier,
		}
		vk.CmdPipelineBarrier2(cb, &rend_dep)

		t := f32(swapchain.frame_idx) * 0.0005
		color_attachments := vk.RenderingAttachmentInfo {
			sType = .RENDERING_ATTACHMENT_INFO,
			imageView = swapchain_image.view,
			imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
			loadOp = .CLEAR,
			storeOp = .STORE,
			clearValue = {
				color = vk.ClearColorValue {
					float32 = [4]f32{0.1, 0.1, 0.1 + 0.1 * math.sin(t), 1.0},
				},
			},
		}
		render_area := vk.Rect2D {
			extent = {width = u32(width), height = u32(height)},
		}
		rendering_info := vk.RenderingInfo {
			sType                = .RENDERING_INFO,
			renderArea           = render_area,
			layerCount           = 1,
			colorAttachmentCount = 1,
			pColorAttachments    = &color_attachments,
		}

		vk.CmdBeginRendering(cb, &rendering_info)

		vp := vk.Viewport {
			width  = f32(render_area.extent.width),
			height = f32(render_area.extent.height),
		}
		scissor := render_area
		vk.CmdSetViewportWithCount(cb, 1, &vp)
		vk.CmdSetScissorWithCount(cb, 1, &scissor)

		proj_view_matrix := camera.proj * camera_get_view(&camera)
		vk.CmdPushConstants(cb, triangle_pipeline.layout, {.VERTEX, .FRAGMENT}, 0, 16 * 4, &proj_view_matrix)
		bind_pipeline(cb, triangle_pipeline)
		vk.CmdDraw(cb, 3, 1, 0, 0)

		vk.CmdEndRendering(cb)

		present_barrier := vk.ImageMemoryBarrier2 {
			sType = .IMAGE_MEMORY_BARRIER_2,
			srcStageMask = {.COLOR_ATTACHMENT_OUTPUT},
			srcAccessMask = {.COLOR_ATTACHMENT_WRITE},
			dstStageMask = {},
			dstAccessMask = {},
			oldLayout = .COLOR_ATTACHMENT_OPTIMAL,
			newLayout = .PRESENT_SRC_KHR,
			image = swapchain_image.image,
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				levelCount = 1,
				layerCount = 1,
			},
		}
		present_dep := vk.DependencyInfo {
			sType                   = .DEPENDENCY_INFO,
			imageMemoryBarrierCount = 1,
			pImageMemoryBarriers    = &present_barrier,
		}
		vk.CmdPipelineBarrier2(cb, &present_dep)

		command_handler_submit(&device.command_handler, handle, true)
		swapchain_present(&swapchain)
	}

	vk.DeviceWaitIdle(device.device)
	if g_enable_logging {
		fmt.println("Shutting down")
	}
	return
}
