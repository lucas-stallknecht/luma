package luma

import "core:fmt"
import "core:math/linalg/glsl"
import "core:mem"
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
		{enable_validation = true, physical_device_selection_fn = device_selection_fn},
	)
	defer device_cleanup(&device)

	swapchain := create_swapchain(&device, window.glfw_window_ptr, {})
	defer swapchain_cleanup(&swapchain)

	pipeline_manager := create_pipeline_manager(&device, "shaders/", "shaders/compiled/")
	defer pipeline_manager_cleanup(&pipeline_manager)

	Visbuffer_Push :: struct {
		proj_view_matrix: glsl.mat4,
		vertex_buffer:    vk.DeviceAddress,
		draw_data_buffer: vk.DeviceAddress,
	}
	visbuffer_pipeline := pipeline_manager_add_raster(
		&pipeline_manager,
		{
			name = "visbuffer",
			vertex_shader = "visbuffer.vert",
			fragment_shader = "visbuffer.frag",
			raster = {primitive_topology = .TRIANGLE_LIST},
			push_constant_size = size_of(Visbuffer_Push),
			color_attachments = {{format = .R32_UINT}},
			depth_test = Depth_Test {
				enable_depth_write = true,
				compare_op = .LESS_OR_EQUAL,
				format = .D32_SFLOAT,
			},
		},
	)

	Present_Push :: struct {
		visbuffer_idx: u32,
	}
	present_pipeline := pipeline_manager_add_raster(
		&pipeline_manager,
		{
			name = "visbuffer_present",
			vertex_shader = "present.vert",
			fragment_shader = "present.frag",
			raster = {primitive_topology = .TRIANGLE_LIST},
			push_constant_size = size_of(Present_Push),
			color_attachments = {{format = swapchain.format}},
		},
	)

	camera := create_camera()
	camera_update_proj(&camera, f32(window.width) / f32(window.height))

	scene: Scene
	scene_init(&scene, &device, "assets/scene.bin")
	defer scene_cleanup(&scene, &device)

	visbuffer := create_image(
		&device,
		{
			width = window.width,
			height = window.height,
			format = .R32_UINT,
			usage = {.COLOR_ATTACHMENT, .TRANSFER_SRC, .STORAGE},
			memory = .GPU_ONLY,
			register_bindless = true,
		},
	)
	depth_image := create_image(
		&device,
		{
			width = window.width,
			height = window.height,
			format = .D32_SFLOAT,
			usage = {.DEPTH_STENCIL_ATTACHMENT},
			memory = .GPU_ONLY,
		},
	)
	default_sampler: vk.Sampler
	sampler_ci := vk.SamplerCreateInfo {
		sType        = .SAMPLER_CREATE_INFO,
		magFilter    = .NEAREST,
		minFilter    = .NEAREST,
		addressModeU = .CLAMP_TO_EDGE,
		addressModeV = .CLAMP_TO_EDGE,
		addressModeW = .CLAMP_TO_EDGE,
		mipLodBias   = 0.0,
		minLod       = 0.0,
		maxLod       = 0.0,
		borderColor  = .INT_OPAQUE_BLACK,
	}
	chk(vk.CreateSampler(device.device, &sampler_ci, nil, &default_sampler))
	bindless_register_sampler(&device, default_sampler)

	defer {
		if default_sampler != 0 {vk.DestroySampler(device.device, default_sampler, nil)}
		destroy_image(&device, visbuffer)
		destroy_image(&device, depth_image)
	}

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

		image_barriers(
			cb,
			{
				image = &visbuffer,
				dst_stage = {.COLOR_ATTACHMENT_OUTPUT},
				dst_access = {.COLOR_ATTACHMENT_WRITE},
			},
			{
				image = &depth_image,
				dst_stage = {.LATE_FRAGMENT_TESTS},
				dst_access = {.DEPTH_STENCIL_ATTACHMENT_WRITE},
			},
		)

		t := f32(swapchain.frame_idx) * 0.0005
		color_attachments := vk.RenderingAttachmentInfo {
			sType = .RENDERING_ATTACHMENT_INFO,
			imageView = visbuffer.view,
			imageLayout = .GENERAL,
			loadOp = .CLEAR,
			storeOp = .STORE,
			clearValue = {color = vk.ClearColorValue{uint32 = [4]u32{}}},
		}
		depth_attachment := vk.RenderingAttachmentInfo {
			sType = .RENDERING_ATTACHMENT_INFO,
			imageView = depth_image.view,
			imageLayout = .GENERAL,
			loadOp = .CLEAR,
			clearValue = {depthStencil = {depth = 1.0}},
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
			pDepthAttachment     = &depth_attachment,
		}

		vk.CmdBeginRendering(cb, &rendering_info)

		vp := vk.Viewport {
			width    = f32(render_area.extent.width),
			height   = f32(render_area.extent.height),
			minDepth = 0.0,
			maxDepth = 1.0,
		}
		scissor := render_area
		vk.CmdSetViewportWithCount(cb, 1, &vp)
		vk.CmdSetScissorWithCount(cb, 1, &scissor)

		push := Visbuffer_Push {
			proj_view_matrix = camera.proj * camera_get_view(&camera),
			vertex_buffer    = scene.position_buffer.device_address,
			draw_data_buffer = scene.draw_data_buffer.device_address,
		}
		vk.CmdPushConstants(
			cb,
			visbuffer_pipeline.layout,
			{.VERTEX, .FRAGMENT},
			0,
			size_of(Visbuffer_Push),
			&push,
		)
		bind_raster_pipeline(cb, visbuffer_pipeline)
		vk.CmdBindIndexBuffer(cb, scene.index_buffer.buffer, 0, .UINT32)
		vk.CmdDrawIndexedIndirect(
			cb,
			scene.draw_command_buffer.buffer,
			0,
			scene.draw_count,
			size_of(vk.DrawIndexedIndirectCommand),
		)

		vk.CmdEndRendering(cb)

		image_barriers(
			cb,
			{
				image = &visbuffer,
				src_stage = {.COLOR_ATTACHMENT_OUTPUT},
				src_access = {.COLOR_ATTACHMENT_WRITE},
				dst_stage = {.FRAGMENT_SHADER},
				dst_access = {.SHADER_STORAGE_READ},
			},
		)

		// Render a fullscreen triangle that samples the visbuffer
		swap_color_attachments := vk.RenderingAttachmentInfo {
			sType = .RENDERING_ATTACHMENT_INFO,
			imageView = swapchain_image.view,
			imageLayout = .GENERAL,
			loadOp = .CLEAR,
			storeOp = .STORE,
			clearValue = {color = vk.ClearColorValue{float32 = [4]f32{0.0, 0.0, 0.0, 0.0}}},
		}
		rendering_info2 := vk.RenderingInfo {
			sType                = .RENDERING_INFO,
			renderArea           = render_area,
			layerCount           = 1,
			colorAttachmentCount = 1,
			pColorAttachments    = &swap_color_attachments,
		}
		vk.CmdBeginRendering(cb, &rendering_info2)
		vk.CmdSetViewportWithCount(cb, 1, &vp)
		vk.CmdSetScissorWithCount(cb, 1, &scissor)

		present_pc := Present_Push {
			visbuffer_idx = visbuffer.bindless_index,
		}
		vk.CmdPushConstants(
			cb,
			present_pipeline.layout,
			{.VERTEX, .FRAGMENT},
			0,
			size_of(Present_Push),
			&present_pc,
		)
		bind_raster_pipeline(cb, present_pipeline)
		vk.CmdDraw(cb, 3, 1, 0, 0)
		vk.CmdEndRendering(cb)

		swapchain_barrier_to_present(cb, swapchain_image)

		command_handler_submit(&device.command_handler, handle, true)
		swapchain_present(&swapchain)
	}

	vk.DeviceWaitIdle(device.device)
	fmt.println("Shutting down")
	return
}
