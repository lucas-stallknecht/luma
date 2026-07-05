package luma

import "core:fmt"
import "core:math/linalg/glsl"
import "core:mem"
import "vendor:glfw"
import mu "vendor:microui"
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

	swapchain := create_swapchain(
		&device,
		window.glfw_window_ptr,
		{preferred_present_mode = vk.PresentModeKHR.FIFO},
	)
	defer swapchain_cleanup(&swapchain)

	pipeline_manager := create_pipeline_manager(&device, "shaders/", "shaders/compiled/")
	defer pipeline_manager_cleanup(&pipeline_manager)

	Visbuffer_Push :: struct {
		frame_data:       vk.DeviceAddress,
		vertex_buffer:    vk.DeviceAddress,
		draw_data_buffer: vk.DeviceAddress,
		uv_buffer:        vk.DeviceAddress,
		material_buffer:  vk.DeviceAddress,
	}
	visbuffer_pipeline := pipeline_manager_add_raster(
		&pipeline_manager,
		{
			name = "visbuffer",
			shader = "visbuffer.glsl",
			raster = {primitive_topology = .TRIANGLE_LIST, front_face = .CLOCKWISE},
			push_constant_size = size_of(Visbuffer_Push),
			color_attachments = {{format = .R32G32_UINT}},
			depth_test = Depth_Test {
				enable_depth_write = true,
				compare_op = .LESS_OR_EQUAL,
				format = .D32_SFLOAT,
			},
		},
	)

	Probe_Debug_Push :: struct {
		frame_data:            vk.DeviceAddress,
		vertex_buffer:         vk.DeviceAddress,
		normal_buffer:         vk.DeviceAddress,
		probe_position_buffer: vk.DeviceAddress,
		probe_sh_buffer:       vk.DeviceAddress,
	}
	probe_debug_pipeline := pipeline_manager_add_raster(
		&pipeline_manager,
		{
			name = "probe_debug",
			shader = "probe_debug.glsl",
			raster = {primitive_topology = .TRIANGLE_LIST, front_face = .CLOCKWISE},
			push_constant_size = size_of(Probe_Debug_Push),
			color_attachments = {{format = swapchain.format}},
			depth_test = Depth_Test {
				enable_depth_write = true,
				compare_op = .LESS_OR_EQUAL,
				format = .D32_SFLOAT,
			},
		},
	)

	Shading_Push :: struct {
		frame_data:       vk.DeviceAddress,
		visbuffer:        u32,
		draw_image:       u32,
		index_buffer:     vk.DeviceAddress,
		vertex_buffer:    vk.DeviceAddress,
		draw_data_buffer: vk.DeviceAddress,
		normal_buffer:    vk.DeviceAddress,
		tangent_buffer:   vk.DeviceAddress,
		uv_buffer:        vk.DeviceAddress,
		material_buffer:  vk.DeviceAddress,
		probe_sh_buffer:  vk.DeviceAddress,
	}
	shading_pipeline := pipeline_manager_add_compute(
		&pipeline_manager,
		{
			name = "shading",
			shader = "shading.glsl",
			push_constant_size = size_of(Shading_Push),
			uses_rt = true,
		},
	)

	Probe_Bake_Push :: struct {
		frame_data:            vk.DeviceAddress,
		index_buffer:          vk.DeviceAddress,
		normal_buffer:         vk.DeviceAddress,
		uv_buffer:             vk.DeviceAddress,
		draw_data_buffer:      vk.DeviceAddress,
		material_buffer:       vk.DeviceAddress,
		probe_position_buffer: vk.DeviceAddress,
		probe_sh_buffer:       vk.DeviceAddress,
	}
	probe_bake_pipeline := pipeline_manager_add_compute(
		&pipeline_manager,
		{
			name = "probe_bake",
			shader = "probe_bake.glsl",
			push_constant_size = size_of(Probe_Bake_Push),
			uses_rt = true,
		},
	)

	Present_Push :: struct {
		draw_image: u32,
	}
	present_pipeline := pipeline_manager_add_raster(
		&pipeline_manager,
		{
			name = "visbuffer_present",
			shader = "present.glsl",
			raster = {primitive_topology = .TRIANGLE_LIST},
			push_constant_size = size_of(Present_Push),
			color_attachments = {{format = swapchain.format}},
		},
	)

	// mu.Context alone is ~256KB, keep it off the stack
	ui := new(Ui)
	defer free(ui)
	ui_init(ui, &device, &pipeline_manager, swapchain.format)
	defer ui_cleanup(ui, &device)
	window_bind_ui(&window, &ui.ctx)

	camera := create_camera()
	camera_update_proj(&camera, f32(window.width) / f32(window.height))

	scene: Scene
	scene_init(&scene, &device, "assets/crytek_sponza/scene.bin")
	defer scene_cleanup(&scene, &device)

	gi: Gi_System
	gi_system_init(
		&gi,
		&device,
		"assets/sphere.bin",
		{
			probe_counts = {24, 6, 10},
			grid_min= {-11.5, 0.2, -5.4},
			grid_max = {10.5, 10.0, 5.0},
		}
	)
	defer gi_system_cleanup(&gi, &device)

	visbuffer := create_image(
		&device,
		{
			width = window.width,
			height = window.height,
			format = .R32G32_UINT,
			usage = {.COLOR_ATTACHMENT, .TRANSFER_SRC, .STORAGE},
			memory = .GPU_ONLY,
			register_bindless = .Storage,
		},
	)
	draw_image := create_image(
		&device,
		{
			width = window.width,
			height = window.height,
			format = .R32G32B32A32_SFLOAT,
			usage = {.TRANSFER_SRC, .STORAGE},
			memory = .GPU_ONLY,
			register_bindless = .Storage,
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
	texture_sampler: vk.Sampler
	texture_sampler_ci := vk.SamplerCreateInfo {
		sType        = .SAMPLER_CREATE_INFO,
		magFilter    = .LINEAR,
		minFilter    = .LINEAR,
		addressModeU = .REPEAT,
		addressModeV = .REPEAT,
		addressModeW = .REPEAT,
		mipLodBias   = -0.5,
		minLod       = 0.0,
		maxLod       = vk.LOD_CLAMP_NONE,
		borderColor  = .INT_OPAQUE_BLACK,
	}
	chk(vk.CreateSampler(device.device, &texture_sampler_ci, nil, &texture_sampler))
	texture_sampler_idx := bindless_register_sampler(&device, texture_sampler)

	Frame_Data :: struct {
		proj_view:          glsl.mat4,
		inv_proj_view:      glsl.mat4,
		camera_position:    glsl.vec3,
		texture_sampler:    u32,
		light_dir:          glsl.vec3,
		albedo_boost:       f32,
		light_color:        glsl.vec3,
		light_intensity:    f32,
		sky_color:          glsl.vec3,
		ssao_radius:        f32,
		grid_min:           glsl.vec3,
		probe_count:        u32,
		grid_spacing:       glsl.vec3,
		frame_idx:          u32,
		probe_counts:       [3]u32,
		ssao_pow:           f32,
	}
	light_dir := glsl.vec3{0.1, 1.0, -0.1}
	light_color := glsl.vec3{1.0, 1.0, 1.0}
	light_intensity: f32 = 4.0
	albedo_boost: f32 = 1.4
	sky_color := glsl.vec3{0.5, 0.7, 1.0}
	ssao_radius: f32 = 0.3
	ssao_pow: f32 = 1.0
	show_probes := false

	// one buffer per in-flight command buffer slot, so the CPU never overwrites frame
	// data the GPU hasn't finished reading yet
	frame_data_buffers: [MAX_COMMAND_BUFFERS]Buffer
	frame_data_mapped: [MAX_COMMAND_BUFFERS]rawptr
	for i in 0 ..< int(MAX_COMMAND_BUFFERS) {
		frame_data_buffers[i] = create_buffer(
			&device,
			{
				size = size_of(Frame_Data),
				usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
				memory = .CPU_UPLOAD,
			},
		)
		vk.MapMemory(
			device.device,
			frame_data_buffers[i].memory,
			0,
			vk.DeviceSize(size_of(Frame_Data)),
			{},
			&frame_data_mapped[i],
		)
	}

	defer {
		if texture_sampler != 0 {
			vk.DestroySampler(device.device, texture_sampler, nil)
		}
		destroy_image(&device, visbuffer)
		destroy_image(&device, draw_image)
		destroy_image(&device, depth_image)
		for i in 0 ..< int(MAX_COMMAND_BUFFERS) {
			vk.UnmapMemory(device.device, frame_data_buffers[i].memory)
			destroy_buffer(&device, &frame_data_buffers[i])
		}
	}

	BAKE_INTERVAL :: 1.0 / 30.0

	last_frame_time: f64 = glfw.GetTime()
	reload_key_prev: bool = false
	bake_accum: f32 = BAKE_INTERVAL

	for !window_should_close(&window) {
		window_update(&window)
		time := glfw.GetTime()
		dt := f32(time - last_frame_time)
		last_frame_time = time

		bake_accum += dt
		do_bake := bake_accum >= BAKE_INTERVAL
		if do_bake {
			bake_accum -= BAKE_INTERVAL
		}

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

		// ui
		mu.begin(&ui.ctx)
		if mu.window(&ui.ctx, "Debug", {x = 20, y = 20, w = 320, h = 640}) {
			fps := 1.0 / dt if dt > 0 else 0
			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.label(&ui.ctx, fmt.tprintf("%.2f ms (%.0f fps)", dt * 1000, fps))

			SWATCH_WIDTH :: 50
			SWATCH_GAP :: 8

			mu.label(&ui.ctx, "Light direction")
			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.slider(&ui.ctx, &light_dir.x, -1, 1)
			mu.slider(&ui.ctx, &light_dir.y, -1, 1)
			mu.slider(&ui.ctx, &light_dir.z, -1, 1)

			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.label(&ui.ctx, "Light color")
			mu.layout_row(&ui.ctx, {-(SWATCH_WIDTH + SWATCH_GAP)}, 0)
			light_color_top := ui_layout_cursor_y(&ui.ctx)
			mu.slider(&ui.ctx, &light_color.x, 0, 1)
			mu.slider(&ui.ctx, &light_color.y, 0, 1)
			mu.slider(&ui.ctx, &light_color.z, 0, 1)
			ui_color_rect(
				&ui.ctx,
				ui_swatch_rect(&ui.ctx, light_color_top, SWATCH_WIDTH),
				light_color,
			)

			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.label(&ui.ctx, "Light intensity")
			mu.slider(&ui.ctx, &light_intensity, 0, 10)

			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.label(&ui.ctx, "Albedo boost")
			mu.slider(&ui.ctx, &albedo_boost, 1, 4)

			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.label(&ui.ctx, "Sky color")
			mu.layout_row(&ui.ctx, {-(SWATCH_WIDTH + SWATCH_GAP)}, 0)
			sky_color_top := ui_layout_cursor_y(&ui.ctx)
			mu.slider(&ui.ctx, &sky_color.x, 0, 1)
			mu.slider(&ui.ctx, &sky_color.y, 0, 1)
			mu.slider(&ui.ctx, &sky_color.z, 0, 1)
			ui_color_rect(&ui.ctx, ui_swatch_rect(&ui.ctx, sky_color_top, SWATCH_WIDTH), sky_color)

			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.label(&ui.ctx, "SSAO radius")
			mu.slider(&ui.ctx, &ssao_radius, 0, 2)

			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.label(&ui.ctx, "SSAO power")
			mu.slider(&ui.ctx, &ssao_pow, 0.1, 8)

			mu.layout_row(&ui.ctx, {-1}, 0)
			mu.checkbox(&ui.ctx, "Show probes", &show_probes)
		}
		mu.end(&ui.ctx)

		// render loop
		swapchain_image := swapchain_acquire_image(&swapchain)
		handle, cb := command_handler_acquire(&device.command_handler)

		proj_view := camera.proj * camera_get_view(&camera)
		frame_data := Frame_Data {
			proj_view          = proj_view,
			inv_proj_view      = glsl.inverse(proj_view),
			camera_position    = camera.position,
			texture_sampler    = texture_sampler_idx,
			light_dir          = glsl.normalize(light_dir),
			light_color        = light_color,
			light_intensity    = light_intensity,
			albedo_boost       = albedo_boost,
			ssao_radius        = ssao_radius,
			sky_color          = sky_color,
			ssao_pow           = ssao_pow,
			grid_min           = gi.info.grid_min,
			probe_count        = gi.probe_count,
			grid_spacing       = gi.grid_spacing,
			probe_counts       = gi.info.probe_counts,
		}
		frame_data_buffer := &frame_data_buffers[handle.buffer_idx]
		mem.copy(frame_data_mapped[handle.buffer_idx], &frame_data, size_of(Frame_Data))

		// re-bake probes every BAKE_INTERVAL rather than every frame
		if do_bake {
			buffer_barriers(
				cb,
				{
					buffer = &gi.probe_sh_buffer,
					src_stage = {.COMPUTE_SHADER, .FRAGMENT_SHADER},
					src_access = {.SHADER_STORAGE_READ, .SHADER_STORAGE_WRITE},
					dst_stage = {.COMPUTE_SHADER},
					dst_access = {.SHADER_STORAGE_READ, .SHADER_STORAGE_WRITE},
				},
			)

			bake_pc := Probe_Bake_Push {
				frame_data            = frame_data_buffer.device_address,
				index_buffer          = scene.index_buffer.device_address,
				normal_buffer         = scene.normal_buffer.device_address,
				uv_buffer             = scene.uv_buffer.device_address,
				draw_data_buffer      = scene.draw_data_buffer.device_address,
				material_buffer       = scene.material_buffer.device_address,
				probe_position_buffer = gi.probe_position_buffer.device_address,
				probe_sh_buffer       = gi.probe_sh_buffer.device_address,
			}
			vk.CmdPushConstants(
				cb,
				probe_bake_pipeline.layout,
				{.COMPUTE},
				0,
				size_of(Probe_Bake_Push),
				&bake_pc,
			)
			bind_compute_pipeline(cb, probe_bake_pipeline)
			vk.CmdDispatch(cb, 1, gi.probe_count, 1)

			// the fresh SH field is read by the shading compute pass and by the probe debug fragment shader below
			buffer_barriers(
				cb,
				{
					buffer = &gi.probe_sh_buffer,
					src_stage = {.COMPUTE_SHADER},
					src_access = {.SHADER_STORAGE_WRITE},
					dst_stage = {.COMPUTE_SHADER, .FRAGMENT_SHADER},
					dst_access = {.SHADER_STORAGE_READ},
				},
			)
		}

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
			{
				image = &draw_image,
				src_stage = {.FRAGMENT_SHADER},
				src_access = {.SHADER_STORAGE_READ},
				dst_stage = {.COMPUTE_SHADER},
				dst_access = {.SHADER_STORAGE_WRITE},
			},
		)

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
			frame_data       = frame_data_buffer.device_address,
			vertex_buffer    = scene.position_buffer.device_address,
			draw_data_buffer = scene.draw_data_buffer.device_address,
			uv_buffer        = scene.uv_buffer.device_address,
			material_buffer  = scene.material_buffer.device_address,
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
				dst_stage = {.COMPUTE_SHADER},
				dst_access = {.SHADER_STORAGE_READ},
			},
		)

		shading_pc := Shading_Push {
			frame_data       = frame_data_buffer.device_address,
			visbuffer        = visbuffer.bindless_idx,
			draw_image       = draw_image.bindless_idx,
			index_buffer     = scene.index_buffer.device_address,
			vertex_buffer    = scene.position_buffer.device_address,
			draw_data_buffer = scene.draw_data_buffer.device_address,
			normal_buffer    = scene.normal_buffer.device_address,
			tangent_buffer   = scene.tangent_buffer.device_address,
			uv_buffer        = scene.uv_buffer.device_address,
			material_buffer  = scene.material_buffer.device_address,
			probe_sh_buffer  = gi.probe_sh_buffer.device_address,
		}
		vk.CmdPushConstants(
			cb,
			shading_pipeline.layout,
			{.COMPUTE},
			0,
			size_of(Shading_Push),
			&shading_pc,
		)
		bind_compute_pipeline(cb, shading_pipeline)
		vk.CmdDispatch(cb, window.width / 8, window.height / 8, 1)

		image_barriers(
			cb,
			{
				image = &draw_image,
				src_stage = {.COMPUTE_SHADER},
				src_access = {.SHADER_STORAGE_WRITE},
				dst_stage = {.FRAGMENT_SHADER},
				dst_access = {.SHADER_STORAGE_READ},
			},
			{
				image = &depth_image,
				src_stage = {.LATE_FRAGMENT_TESTS},
				src_access = {.DEPTH_STENCIL_ATTACHMENT_WRITE},
				dst_stage = {.EARLY_FRAGMENT_TESTS, .LATE_FRAGMENT_TESTS},
				dst_access = {.DEPTH_STENCIL_ATTACHMENT_READ, .DEPTH_STENCIL_ATTACHMENT_WRITE},
			},
		)

		// render a fullscreen triangle that samples the draw image and apply tonemapping
		swap_color_attachments := vk.RenderingAttachmentInfo {
			sType = .RENDERING_ATTACHMENT_INFO,
			imageView = swapchain_image.view,
			imageLayout = .GENERAL,
			loadOp = .CLEAR,
			storeOp = .STORE,
			clearValue = {color = vk.ClearColorValue{float32 = [4]f32{0.0, 0.0, 0.0, 0.0}}},
		}
		// reuse the visbuffer pass' depth so probes are occluded by real geometry
		// present/ui don't use it, relying on VK_EXT_dynamic_rendering_unused_attachments
		swap_depth_attachment := vk.RenderingAttachmentInfo {
			sType       = .RENDERING_ATTACHMENT_INFO,
			imageView   = depth_image.view,
			imageLayout = .GENERAL,
			loadOp      = .LOAD,
			storeOp     = .DONT_CARE,
		}
		rendering_info2 := vk.RenderingInfo {
			sType                = .RENDERING_INFO,
			renderArea           = render_area,
			layerCount           = 1,
			colorAttachmentCount = 1,
			pColorAttachments    = &swap_color_attachments,
			pDepthAttachment     = &swap_depth_attachment,
		}
		vk.CmdBeginRendering(cb, &rendering_info2)
		vk.CmdSetViewportWithCount(cb, 1, &vp)
		vk.CmdSetScissorWithCount(cb, 1, &scissor)

		present_pc := Present_Push {
			draw_image = draw_image.bindless_idx,
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

		if show_probes {
			// probes debug: draw directly on the swapchain
			probe_debug_pc := Probe_Debug_Push {
				frame_data            = frame_data_buffer.device_address,
				vertex_buffer         = gi.debug_sphere_vertex_buffer.device_address,
				normal_buffer         = gi.debug_sphere_normal_buffer.device_address,
				probe_position_buffer = gi.probe_position_buffer.device_address,
				probe_sh_buffer       = gi.probe_sh_buffer.device_address,
			}
			vk.CmdPushConstants(
				cb,
				probe_debug_pipeline.layout,
				{.VERTEX, .FRAGMENT},
				0,
				size_of(Probe_Debug_Push),
				&probe_debug_pc,
			)
			bind_raster_pipeline(cb, probe_debug_pipeline)
			vk.CmdDraw(cb, gi.debug_sphere_vertex_count, gi.probe_count, 0, 0)
		}

		ui_render(ui, cb, handle.buffer_idx, width, height)

		vk.CmdEndRendering(cb)

		swapchain_barrier_to_present(cb, swapchain_image)

		command_handler_submit(&device.command_handler, handle, true)
		swapchain_present(&swapchain)
		free_all(context.temp_allocator)
	}

	vk.DeviceWaitIdle(device.device)
	fmt.println("Shutting down")
	return
}
