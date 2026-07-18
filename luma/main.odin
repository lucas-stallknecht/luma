package luma

import imgui "../imgui"
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
		{enable_validation = ODIN_DEBUG, physical_device_selection_fn = device_selection_fn},
	)
	defer device_cleanup(&device)

	swapchain := create_swapchain(
		&device,
		window.glfw_window_ptr,
		{preferred_present_mode = vk.PresentModeKHR.FIFO},
	)
	defer swapchain_cleanup(&swapchain)

	ui_init(&device, &window, &swapchain)
	defer ui_cleanup()

	scene: Scene
	if !scene_init(&scene, &device, "assets/crytek_sponza/scene.bin") do return
	defer scene_cleanup(&scene, &device)

	gi: Gi_System
	if !gi_system_init(
		&gi,
		&device,
		"assets/sphere.bin",
		{
			probe_counts = {24, 10, 11},
			grid_min = {-12.5, -0.2, -6.1},
			grid_max = {11.5, 11.0, 5.5},
		},
	) {
		return
	}
	defer gi_system_cleanup(&gi, &device)

	rd: Renderer
	if !renderer_init(&rd, &device, &window, &swapchain) do return
	defer renderer_cleanup(&rd)

	camera := create_default_camera()
	camera_update_proj(&camera, f32(window.width) / f32(window.height))
	// seeds zero velocity on the first frame instead of a zeroed matrix's garbage
	rd.prev_proj_view = camera.proj * camera_get_view(&camera)

	light_dir := glsl.vec3{0.1, 1.0, -0.1}
	light_color := glsl.vec3{1.0, 1.0, 1.0}
	light_intensity: f32 = 12.0
	albedo_boost: f32 = 1.4
	rtao_radius: f32 = 0.3
	rtao_pow: f32 = 1.0
	cirrus: f32 = 0.4
	cumulus: f32 = 0.8
	cloud_noise_scale: f32 = 0.7
	cloud_noise_speed: f32 = 0.1
	show_probes := false
	bloom_intensity: f32 = 0.04
	bloom_filter_radius: f32 = 0.001

	last_frame_time: f64 = glfw.GetTime()
	reload_key_prev: bool = false
	bake_accum: f32 = GI_BAKE_INTERVAL

	for !window_should_close(&window) {
		window_update(&window)
		time := glfw.GetTime()
		dt := f32(time - last_frame_time)
		last_frame_time = time

		bake_accum += dt
		do_bake := bake_accum >= GI_BAKE_INTERVAL
		if do_bake {
			bake_accum -= GI_BAKE_INTERVAL
		}

		handle_camera_inputs(&window, &camera, dt)
		if window.pressed_keys[glfw.KEY_R] && !reload_key_prev {
			if pipeline_reload_all(&rd.pipeline_manager) {
				fmt.println("Shaders successfully reloaded")
			}
		}
		reload_key_prev = window.pressed_keys[glfw.KEY_R]

		if window.resized {
			camera_update_proj(&camera, f32(window.width) / f32(window.height))
			// TODO: resize the swapchain and size-dependent images
			window.resized = false
		}

		// ui
		ui_new_frame()
		imgui.SetNextWindowPos({20, 20}, .Once)
		imgui.SetNextWindowSize({320, 800}, .Once)
		if imgui.Begin("Debug") {
			fps := 1.0 / dt if dt > 0 else 0
			imgui.Text("%.2f ms (%.0f fps)", dt * 1000, fps)

			imgui.SeparatorText("Lighting")
			imgui.SliderFloat3("Direction", cast(^[3]f32)&light_dir, -1, 1)
			imgui.ColorEdit3("Color", cast(^[3]f32)&light_color)
			imgui.SliderFloat("Light intensity", &light_intensity, 0, 40)

			imgui.SeparatorText("Sky")
			imgui.SliderFloat("Cirrus clouds", &cirrus, 0, 1)
			imgui.SliderFloat("Cumulus clouds", &cumulus, 0, 1)
			imgui.SliderFloat("Noise size", &cloud_noise_scale, 0.01, 1.0)
			imgui.SliderFloat("Noise pan speed", &cloud_noise_speed, 0.01, 0.5)

			imgui.SeparatorText("Ambient Occlusion")
			imgui.SliderFloat("Radius", &rtao_radius, 0, 2)
			imgui.SliderFloat("Power", &rtao_pow, 0.1, 8)

			imgui.SeparatorText("Bloom")
			imgui.SliderFloat("Bloom intensity", &bloom_intensity, 0.001, 0.2)
			imgui.SliderFloat("Filter radius", &bloom_filter_radius, 0.001, 0.01)

			imgui.SeparatorText("Global Illumination")
			imgui.Checkbox("Show probes", &show_probes)
			imgui.SliderFloat("Albedo boost", &albedo_boost, 1, 4)
		}
		imgui.End()

		// render loop
		swapchain_image := swapchain_acquire_image(&swapchain)
		handle, cb := command_handler_acquire(&device.command_handler)

		proj_view := camera.proj * camera_get_view(&camera)
		frame_data := Frame_Data {
			proj_view         = proj_view,
			inv_proj_view     = glsl.inverse(proj_view),
			prev_proj_view    = rd.prev_proj_view,
			camera_position   = camera.position,
			texture_sampler   = rd.texture_sampler_idx,
			light_dir         = glsl.normalize(light_dir),
			light_color       = light_color,
			light_intensity   = light_intensity,
			albedo_boost      = albedo_boost,
			rtao_radius       = rtao_radius,
			rtao_pow          = rtao_pow,
			grid_min          = gi.info.grid_min,
			probe_count       = gi.probe_count,
			grid_spacing      = gi.grid_spacing,
			probe_counts      = gi.info.probe_counts,
			time              = f32(time),
			cirrus            = cirrus,
			cumulus           = cumulus,
			cloud_noise_scale = cloud_noise_scale,
			cloud_noise_speed = cloud_noise_speed,
			sky_cubemap       = rd.sky_cubemap.bindless_idx,
		}
		frame_data_buffer := &rd.frame_data_buffers[handle.buffer_idx]
		mem.copy(frame_data_buffer.mapped, &frame_data, size_of(Frame_Data))
		rd.prev_proj_view = proj_view

		rd.frame = {
			frame_data_buffer = frame_data_buffer,
			width = window.width,
			height = window.height,
			render_area = {extent = {width = window.width, height = window.height}},
			viewport = {
				width = f32(window.width),
				height = f32(window.height),
				minDepth = 0.0,
				maxDepth = 1.0,
			},
			do_bake = do_bake,
			show_probes = show_probes,
			bloom_intensity = bloom_intensity,
			bloom_filter_radius = bloom_filter_radius,
		}

		renderer_draw(&rd, cb, swapchain_image, &scene, &gi)
		swapchain_barrier_to_present(cb, swapchain_image)

		command_handler_submit(&device.command_handler, handle, true)
		swapchain_present(&swapchain)
		free_all(context.temp_allocator)
	}

	vk.DeviceWaitIdle(device.device)
	fmt.println("Shutting down")
}
