package luma

import "core:math/linalg/glsl"
import vk "vendor:vulkan"

SKY_CUBEMAP_SIZE :: 256
SKY_CUBEMAP_FACES :: 6
GI_BAKE_INTERVAL :: 1.0 / 30.0
RTAO_RES_DIVISOR :: 1
BLOOM_MIP_COUNT :: 4

Frame_Data :: struct {
	proj_view:         glsl.mat4,
	inv_proj_view:     glsl.mat4,
	prev_proj_view:    glsl.mat4,
	camera_position:   glsl.vec3,
	texture_sampler:   u32,
	light_dir:         glsl.vec3,
	albedo_boost:      f32,
	light_color:       glsl.vec3,
	light_intensity:   f32,
	grid_min:          glsl.vec3,
	probe_count:       u32,
	grid_spacing:      glsl.vec3,
	frame_idx:         u32,
	probe_counts:      [3]u32,
	rtao_pow:          f32,
	rtao_radius:       f32,
	time:              f32,
	cirrus:            f32,
	cumulus:           f32,
	cloud_noise_scale: f32,
	cloud_noise_speed: f32,
	sky_cubemap:       u32,
}

Sky_Bake_Push :: struct {
	frame_data:    vk.DeviceAddress,
	cubemap_image: u32,
}

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

Visbuffer_Push :: struct {
	frame_data:       vk.DeviceAddress,
	vertex_buffer:    vk.DeviceAddress,
	draw_data_buffer: vk.DeviceAddress,
	uv_buffer:        vk.DeviceAddress,
	material_buffer:  vk.DeviceAddress,
}

Rtao_Push :: struct {
	frame_data:       vk.DeviceAddress,
	visbuffer:        u32,
	rtao_image:       u32,
	index_buffer:     vk.DeviceAddress,
	vertex_buffer:    vk.DeviceAddress,
	draw_data_buffer: vk.DeviceAddress,
	normal_buffer:    vk.DeviceAddress,
}

Motion_Vectors_Push :: struct {
	frame_data:       vk.DeviceAddress,
	visbuffer:        u32,
	velocity_image:   u32,
	index_buffer:     vk.DeviceAddress,
	vertex_buffer:    vk.DeviceAddress,
	draw_data_buffer: vk.DeviceAddress,
}

Shading_Push :: struct {
	frame_data:            vk.DeviceAddress,
	visbuffer:             u32,
	draw_image:            u32,
	rtao_image:            u32,
	index_buffer:          vk.DeviceAddress,
	vertex_buffer:         vk.DeviceAddress,
	draw_data_buffer:      vk.DeviceAddress,
	normal_buffer:         vk.DeviceAddress,
	tangent_buffer:        vk.DeviceAddress,
	uv_buffer:             vk.DeviceAddress,
	material_buffer:       vk.DeviceAddress,
	probe_sh_buffer:       vk.DeviceAddress,
	probe_position_buffer: vk.DeviceAddress,
}

Bloom_Downsample_Push :: struct {
	src_texture: u32,
	src_sampler: u32,
	src_lod:     f32,
	dst_image:   u32,
	dst_width:   u32,
	dst_height:  u32,
}

Bloom_Upsample_Push :: struct {
	src_texture:   u32,
	src_sampler:   u32,
	src_lod:       f32, // coarser mip to tent-sample
	dst_lod:       f32, // this mip, to fetch the already-downsampled base value
	dst_image:     u32,
	dst_width:     u32,
	dst_height:    u32,
	filter_radius: f32,
}

Present_Push :: struct {
	draw_image:      u32,
	bloom_texture:   u32,
	bloom_sampler:   u32,
	bloom_intensity: f32,
	velocity_image:  u32,
}

Probe_Debug_Push :: struct {
	frame_data:            vk.DeviceAddress,
	vertex_buffer:         vk.DeviceAddress,
	normal_buffer:         vk.DeviceAddress,
	probe_position_buffer: vk.DeviceAddress,
	probe_sh_buffer:       vk.DeviceAddress,
}

Renderer :: struct {
	device:                    ^Device,
	pipeline_manager:          Pipeline_Manager,
	rg:                        RenderGraph,
	frame_data_buffers:        [MAX_COMMAND_BUFFERS]Buffer,
	sky_bake_pipeline:         ^Compute_Pipeline,
	sky_cubemap:               Image,
	sky_cubemap_array_view:    vk.ImageView,
	sky_cubemap_array_idx:     u32,
	probe_bake_pipeline:       ^Compute_Pipeline,
	visbuffer_pipeline:        ^Raster_Pipeline,
	visbuffer:                 Image,
	depth_image:               Image,
	rtao_pipeline:             ^Compute_Pipeline,
	rtao_image:                Image,
	rtao_image_tex_idx:        u32,
	motion_vectors_pipeline:   ^Compute_Pipeline,
	velocity_image:            Image,
	prev_proj_view:            glsl.mat4,
	shading_pipeline:          ^Compute_Pipeline,
	draw_image:                Image,
	draw_image_tex_idx:        u32,
	texture_sampler:           vk.Sampler,
	texture_sampler_idx:       u32,
	bloom_downsample_pipeline: ^Compute_Pipeline,
	bloom_upsample_pipeline:   ^Compute_Pipeline,
	bloom_image:               Image,
	bloom_scratch:             Image,
	bloom_mip_count:           u32,
	bloom_base_width:          u32,
	bloom_base_height:         u32,
	bloom_sampler:             vk.Sampler,
	bloom_sampler_idx:         u32,
	present_pipeline:          ^Raster_Pipeline,
	probe_debug_pipeline:      ^Raster_Pipeline,
	frame:                     struct {
		scene:               ^Scene,
		gi:                  ^Gi_System,
		frame_data_buffer:   ^Buffer,
		swapchain_view:      vk.ImageView,
		width, height:       u32,
		render_area:         vk.Rect2D,
		viewport:            vk.Viewport,
		do_bake:             bool,
		show_probes:         bool,
		bloom_intensity:     f32,
		bloom_filter_radius: f32,
	},
}

renderer_init :: proc(
	rd: ^Renderer,
	device: ^Device,
	window: ^Window,
	swapchain: ^Swapchain,
) -> bool {
	rd.device = device
	rd.pipeline_manager = create_pipeline_manager(device, "shaders/", "shaders/compiled/")
	render_graph_init(&rd.rg)

	// one buffer per in-flight command buffer slot, so the CPU never overwrites frame
	// data the GPU hasn't finished reading yet
	for i in 0 ..< int(MAX_COMMAND_BUFFERS) {
		rd.frame_data_buffers[i] = create_buffer(
			device,
			{
				size = size_of(Frame_Data),
				usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
				memory = .CPU_UPLOAD,
			},
		)
	}

	init_handle, init_cb := command_handler_acquire(&device.command_handler)

	// sky, baked into a small cubemap once a frame and sampled by shading + GI probe bake
	rd.sky_bake_pipeline = pipeline_manager_add_compute(
		&rd.pipeline_manager,
		{name = "sky_bake", shader = "sky_bake.glsl", push_constant_size = size_of(Sky_Bake_Push)},
	)
	rd.sky_cubemap = create_image(
		device,
		init_cb,
		{
			width = SKY_CUBEMAP_SIZE,
			height = SKY_CUBEMAP_SIZE,
			format = .R32G32B32A32_SFLOAT,
			usage = {.STORAGE, .SAMPLED},
			memory = .GPU_ONLY,
			array_layers = 6,
			register_bindless = .TextureCube,
		},
	)
	sky_cubemap_array_view_ci := vk.ImageViewCreateInfo {
		sType = .IMAGE_VIEW_CREATE_INFO,
		image = rd.sky_cubemap.image,
		viewType = .D2_ARRAY,
		format = rd.sky_cubemap.format,
		subresourceRange = {aspectMask = {.COLOR}, layerCount = 6, levelCount = 1},
	}
	chk(
		vk.CreateImageView(
			device.device,
			&sky_cubemap_array_view_ci,
			nil,
			&rd.sky_cubemap_array_view,
		),
	)
	rd.sky_cubemap_array_idx = bindless_register_storage_image_array(
		device,
		rd.sky_cubemap_array_view,
	)

	rd.probe_bake_pipeline = pipeline_manager_add_compute(
		&rd.pipeline_manager,
		{
			name = "probe_bake",
			shader = "probe_bake.glsl",
			push_constant_size = size_of(Probe_Bake_Push),
			uses_rt = true,
		},
	)

	rd.visbuffer_pipeline = pipeline_manager_add_raster(
		&rd.pipeline_manager,
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
	rd.visbuffer = create_image(
		device,
		init_cb,
		{
			width = window.width,
			height = window.height,
			format = .R32G32_UINT,
			usage = {.COLOR_ATTACHMENT, .TRANSFER_SRC, .STORAGE},
			memory = .GPU_ONLY,
			register_bindless = .Storage,
		},
	)
	rd.depth_image = create_image(
		device,
		init_cb,
		{
			width = window.width,
			height = window.height,
			format = .D32_SFLOAT,
			usage = {.DEPTH_STENCIL_ATTACHMENT},
			memory = .GPU_ONLY,
		},
	)

	rd.rtao_pipeline = pipeline_manager_add_compute(
		&rd.pipeline_manager,
		{
			name = "rtao",
			shader = "rtao.glsl",
			push_constant_size = size_of(Rtao_Push),
			uses_rt = true,
		},
	)
	rd.rtao_image = create_image(
		device,
		init_cb,
		{
			width = window.width / RTAO_RES_DIVISOR,
			height = window.height / RTAO_RES_DIVISOR,
			format = .R8_UNORM,
			usage = {.TRANSFER_SRC, .STORAGE, .SAMPLED},
			memory = .GPU_ONLY,
			register_bindless = .Storage,
		},
	)
	rd.rtao_image_tex_idx = bindless_register_texture(device, rd.rtao_image.view)

	rd.motion_vectors_pipeline = pipeline_manager_add_compute(
		&rd.pipeline_manager,
		{
			name = "motion_vectors",
			shader = "motion_vectors.glsl",
			push_constant_size = size_of(Motion_Vectors_Push),
		},
	)
	rd.velocity_image = create_image(
		device,
		init_cb,
		{
			width = window.width,
			height = window.height,
			format = .R16G16_SFLOAT,
			usage = {.STORAGE},
			memory = .GPU_ONLY,
			register_bindless = .Storage,
		},
	)

	rd.shading_pipeline = pipeline_manager_add_compute(
		&rd.pipeline_manager,
		{
			name = "shading",
			shader = "shading.glsl",
			push_constant_size = size_of(Shading_Push),
			uses_rt = true,
		},
	)
	rd.draw_image = create_image(
		device,
		init_cb,
		{
			width = window.width,
			height = window.height,
			format = .R32G32B32A32_SFLOAT,
			usage = {.TRANSFER_SRC, .STORAGE, .SAMPLED},
			memory = .GPU_ONLY,
			register_bindless = .Storage,
		},
	)
	rd.draw_image_tex_idx = bindless_register_texture(device, rd.draw_image.view)

	// mip 0 = half screen res
	rd.bloom_downsample_pipeline = pipeline_manager_add_compute(
		&rd.pipeline_manager,
		{
			name = "bloom_downsample",
			shader = "bloom_downsample.glsl",
			push_constant_size = size_of(Bloom_Downsample_Push),
		},
	)
	rd.bloom_upsample_pipeline = pipeline_manager_add_compute(
		&rd.pipeline_manager,
		{
			name = "bloom_upsample",
			shader = "bloom_upsample.glsl",
			push_constant_size = size_of(Bloom_Upsample_Push),
		},
	)
	rd.bloom_base_width = max(window.width / 2, 1)
	rd.bloom_base_height = max(window.height / 2, 1)
	rd.bloom_image = create_image(
		device,
		init_cb,
		{
			width = rd.bloom_base_width,
			height = rd.bloom_base_height,
			format = .R32G32B32A32_SFLOAT,
			usage = {.SAMPLED, .TRANSFER_DST},
			memory = .GPU_ONLY,
			register_bindless = .Texture,
			mips = true,
		},
	)
	rd.bloom_mip_count = min(rd.bloom_image.mip_levels, BLOOM_MIP_COUNT)
	rd.bloom_scratch = create_image(
		device,
		init_cb,
		{
			width = rd.bloom_base_width,
			height = rd.bloom_base_height,
			format = .R32G32B32A32_SFLOAT,
			usage = {.STORAGE, .TRANSFER_SRC},
			memory = .GPU_ONLY,
			register_bindless = .Storage,
		},
	)

	rd.present_pipeline = pipeline_manager_add_raster(
		&rd.pipeline_manager,
		{
			name = "visbuffer_present",
			shader = "present.glsl",
			raster = {primitive_topology = .TRIANGLE_LIST},
			push_constant_size = size_of(Present_Push),
			color_attachments = {{format = swapchain.format}},
		},
	)
	rd.probe_debug_pipeline = pipeline_manager_add_raster(
		&rd.pipeline_manager,
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

	command_handler_submit(&device.command_handler, init_handle, false)
	command_handler_wait(&device.command_handler, init_handle)

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
	chk(vk.CreateSampler(device.device, &texture_sampler_ci, nil, &rd.texture_sampler))
	rd.texture_sampler_idx = bindless_register_sampler(device, rd.texture_sampler)

	bloom_sampler_ci := vk.SamplerCreateInfo {
		sType        = .SAMPLER_CREATE_INFO,
		magFilter    = .LINEAR,
		minFilter    = .LINEAR,
		addressModeU = .CLAMP_TO_EDGE,
		addressModeV = .CLAMP_TO_EDGE,
		addressModeW = .CLAMP_TO_EDGE,
		minLod       = 0.0,
		maxLod       = vk.LOD_CLAMP_NONE,
		borderColor  = .INT_OPAQUE_BLACK,
	}
	chk(vk.CreateSampler(device.device, &bloom_sampler_ci, nil, &rd.bloom_sampler))
	rd.bloom_sampler_idx = bindless_register_sampler(device, rd.bloom_sampler)

	return true
}

renderer_cleanup :: proc(rd: ^Renderer) {
	if rd.texture_sampler != 0 {
		vk.DestroySampler(rd.device.device, rd.texture_sampler, nil)
	}
	if rd.bloom_sampler != 0 {
		vk.DestroySampler(rd.device.device, rd.bloom_sampler, nil)
	}
	destroy_image(rd.device, rd.sky_cubemap)
	vk.DestroyImageView(rd.device.device, rd.sky_cubemap_array_view, nil)
	destroy_image(rd.device, rd.visbuffer)
	destroy_image(rd.device, rd.depth_image)
	destroy_image(rd.device, rd.rtao_image)
	destroy_image(rd.device, rd.velocity_image)
	destroy_image(rd.device, rd.draw_image)
	destroy_image(rd.device, rd.bloom_image)
	destroy_image(rd.device, rd.bloom_scratch)
	for i in 0 ..< int(MAX_COMMAND_BUFFERS) {
		destroy_buffer(rd.device, &rd.frame_data_buffers[i])
	}

	render_graph_cleanup(&rd.rg)
	pipeline_manager_cleanup(&rd.pipeline_manager)
}

// ────────────────────────────────────────────────────────────────
// Passes

sky_bake_pass :: proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer) {
	rd := cast(^Renderer)user_data
	pc := Sky_Bake_Push {
		frame_data    = rd.frame.frame_data_buffer.device_address,
		cubemap_image = rd.sky_cubemap_array_idx,
	}
	vk.CmdPushConstants(
		cb,
		rd.sky_bake_pipeline.layout,
		{.COMPUTE},
		0,
		size_of(Sky_Bake_Push),
		&pc,
	)
	bind_compute_pipeline(cb, rd.sky_bake_pipeline)
	vk.CmdDispatch(cb, (SKY_CUBEMAP_SIZE + 7) / 8, (SKY_CUBEMAP_SIZE + 7) / 8, SKY_CUBEMAP_FACES)
}

probe_bake_pass :: proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer) {
	rd := cast(^Renderer)user_data
	pc := Probe_Bake_Push {
		frame_data            = rd.frame.frame_data_buffer.device_address,
		index_buffer          = rd.frame.scene.index_buffer.device_address,
		normal_buffer         = rd.frame.scene.normal_buffer.device_address,
		uv_buffer             = rd.frame.scene.uv_buffer.device_address,
		draw_data_buffer      = rd.frame.scene.draw_data_buffer.device_address,
		material_buffer       = rd.frame.scene.material_buffer.device_address,
		probe_position_buffer = rd.frame.gi.probe_position_buffer.device_address,
		probe_sh_buffer       = rd.frame.gi.probe_sh_buffer.device_address,
	}
	vk.CmdPushConstants(
		cb,
		rd.probe_bake_pipeline.layout,
		{.COMPUTE},
		0,
		size_of(Probe_Bake_Push),
		&pc,
	)
	bind_compute_pipeline(cb, rd.probe_bake_pipeline)
	vk.CmdDispatch(cb, 1, rd.frame.gi.probe_count, 1)
}

visbuffer_pass :: proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer) {
	rd := cast(^Renderer)user_data

	color_attachment := vk.RenderingAttachmentInfo {
		sType = .RENDERING_ATTACHMENT_INFO,
		imageView = rd.visbuffer.view,
		imageLayout = .GENERAL,
		loadOp = .CLEAR,
		storeOp = .STORE,
		clearValue = {color = vk.ClearColorValue{uint32 = [4]u32{}}},
	}
	depth_attachment := vk.RenderingAttachmentInfo {
		sType = .RENDERING_ATTACHMENT_INFO,
		imageView = rd.depth_image.view,
		imageLayout = .GENERAL,
		loadOp = .CLEAR,
		clearValue = {depthStencil = {depth = 1.0}},
	}
	rendering_info := vk.RenderingInfo {
		sType                = .RENDERING_INFO,
		renderArea           = rd.frame.render_area,
		layerCount           = 1,
		colorAttachmentCount = 1,
		pColorAttachments    = &color_attachment,
		pDepthAttachment     = &depth_attachment,
	}
	vk.CmdBeginRendering(cb, &rendering_info)
	vk.CmdSetViewportWithCount(cb, 1, &rd.frame.viewport)
	vk.CmdSetScissorWithCount(cb, 1, &rd.frame.render_area)

	pc := Visbuffer_Push {
		frame_data       = rd.frame.frame_data_buffer.device_address,
		vertex_buffer    = rd.frame.scene.position_buffer.device_address,
		draw_data_buffer = rd.frame.scene.draw_data_buffer.device_address,
		uv_buffer        = rd.frame.scene.uv_buffer.device_address,
		material_buffer  = rd.frame.scene.material_buffer.device_address,
	}
	vk.CmdPushConstants(
		cb,
		rd.visbuffer_pipeline.layout,
		{.VERTEX, .FRAGMENT},
		0,
		size_of(Visbuffer_Push),
		&pc,
	)
	bind_raster_pipeline(cb, rd.visbuffer_pipeline)
	vk.CmdBindIndexBuffer(cb, rd.frame.scene.index_buffer.buffer, 0, .UINT32)
	vk.CmdDrawIndexedIndirect(
		cb,
		rd.frame.scene.draw_command_buffer.buffer,
		0,
		rd.frame.scene.draw_count,
		size_of(vk.DrawIndexedIndirectCommand),
	)

	vk.CmdEndRendering(cb)
}

rtao_pass :: proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer) {
	rd := cast(^Renderer)user_data
	pc := Rtao_Push {
		frame_data       = rd.frame.frame_data_buffer.device_address,
		visbuffer        = rd.visbuffer.bindless_idx,
		rtao_image       = rd.rtao_image.bindless_idx,
		index_buffer     = rd.frame.scene.index_buffer.device_address,
		vertex_buffer    = rd.frame.scene.position_buffer.device_address,
		draw_data_buffer = rd.frame.scene.draw_data_buffer.device_address,
		normal_buffer    = rd.frame.scene.normal_buffer.device_address,
	}
	vk.CmdPushConstants(cb, rd.rtao_pipeline.layout, {.COMPUTE}, 0, size_of(Rtao_Push), &pc)
	bind_compute_pipeline(cb, rd.rtao_pipeline)
	vk.CmdDispatch(
		cb,
		(rd.frame.width / RTAO_RES_DIVISOR + 7) / 8,
		(rd.frame.height / RTAO_RES_DIVISOR + 7) / 8,
		1,
	)
}

motion_vectors_pass :: proc(
	resources: []Render_Resource,
	user_data: rawptr,
	cb: vk.CommandBuffer,
) {
	rd := cast(^Renderer)user_data
	pc := Motion_Vectors_Push {
		frame_data       = rd.frame.frame_data_buffer.device_address,
		visbuffer        = rd.visbuffer.bindless_idx,
		velocity_image   = rd.velocity_image.bindless_idx,
		index_buffer     = rd.frame.scene.index_buffer.device_address,
		vertex_buffer    = rd.frame.scene.position_buffer.device_address,
		draw_data_buffer = rd.frame.scene.draw_data_buffer.device_address,
	}
	vk.CmdPushConstants(
		cb,
		rd.motion_vectors_pipeline.layout,
		{.COMPUTE},
		0,
		size_of(Motion_Vectors_Push),
		&pc,
	)
	bind_compute_pipeline(cb, rd.motion_vectors_pipeline)
	vk.CmdDispatch(cb, (rd.frame.width + 7) / 8, (rd.frame.height + 7) / 8, 1)
}

shading_pass :: proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer) {
	rd := cast(^Renderer)user_data
	pc := Shading_Push {
		frame_data            = rd.frame.frame_data_buffer.device_address,
		visbuffer             = rd.visbuffer.bindless_idx,
		draw_image            = rd.draw_image.bindless_idx,
		rtao_image            = rd.rtao_image_tex_idx,
		index_buffer          = rd.frame.scene.index_buffer.device_address,
		vertex_buffer         = rd.frame.scene.position_buffer.device_address,
		draw_data_buffer      = rd.frame.scene.draw_data_buffer.device_address,
		normal_buffer         = rd.frame.scene.normal_buffer.device_address,
		tangent_buffer        = rd.frame.scene.tangent_buffer.device_address,
		uv_buffer             = rd.frame.scene.uv_buffer.device_address,
		material_buffer       = rd.frame.scene.material_buffer.device_address,
		probe_sh_buffer       = rd.frame.gi.probe_sh_buffer.device_address,
		probe_position_buffer = rd.frame.gi.probe_position_buffer.device_address,
	}
	vk.CmdPushConstants(cb, rd.shading_pipeline.layout, {.COMPUTE}, 0, size_of(Shading_Push), &pc)
	bind_compute_pipeline(cb, rd.shading_pipeline)
	vk.CmdDispatch(cb, (rd.frame.width + 7) / 8, (rd.frame.height + 7) / 8, 1)
}

// the whole downsample/upsample mip chain runs as one graph pass: rd.bloom_mip_count
// is fixed at init, so there's no per-frame benefit to a pass per mip
//
// bloom_scratch never leaves this pass, so it's never declared as a render graph resource
// bloom_image's actual write (via commit_mip) isn't declared either
// by the time this pass returns, commit_mip's own trailing barrier has already left
// bloom_image sampled-read again, matching the read usage this pass and present both
// declare to the graph
bloom_pass :: proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer) {
	rd := cast(^Renderer)user_data

	// copies one mip from bloom_scratch into bloom_image, then hands sampled-read access
	// on bloom_image back out so the next iteration (or present) can read it
	commit_mip :: proc(cb: vk.CommandBuffer, scratch: ^Image, image: ^Image, mip, w, h: u32) {
		image_barriers(
			cb,
			{
				image = scratch,
				src_stage = {.COMPUTE_SHADER},
				src_access = {.SHADER_STORAGE_WRITE},
				dst_stage = {.TRANSFER},
				dst_access = {.TRANSFER_READ},
			},
			{
				image = image,
				src_stage = {.COMPUTE_SHADER, .FRAGMENT_SHADER},
				src_access = {.SHADER_SAMPLED_READ},
				dst_stage = {.TRANSFER},
				dst_access = {.TRANSFER_WRITE},
			},
		)

		copy_region := vk.ImageCopy {
			srcSubresource = {aspectMask = {.COLOR}, layerCount = 1},
			dstSubresource = {aspectMask = {.COLOR}, mipLevel = mip, layerCount = 1},
			extent = {w, h, 1},
		}
		vk.CmdCopyImage(cb, scratch.image, .GENERAL, image.image, .GENERAL, 1, &copy_region)

		image_barriers(
			cb,
			{
				image = image,
				src_stage = {.TRANSFER},
				src_access = {.TRANSFER_WRITE},
				dst_stage = {.COMPUTE_SHADER, .FRAGMENT_SHADER},
				dst_access = {.SHADER_SAMPLED_READ},
			},
		)
	}

	bind_compute_pipeline(cb, rd.bloom_downsample_pipeline)
	for i in 0 ..< rd.bloom_mip_count {
		src_texture := rd.draw_image_tex_idx if i == 0 else rd.bloom_image.bindless_idx
		src_lod := 0.0 if i == 0 else f32(i - 1)
		w := max(rd.bloom_base_width >> i, 1)
		h := max(rd.bloom_base_height >> i, 1)

		image_barriers(
			cb,
			{
				image = &rd.bloom_scratch,
				src_stage = {.TRANSFER},
				src_access = {.TRANSFER_READ},
				dst_stage = {.COMPUTE_SHADER},
				dst_access = {.SHADER_STORAGE_WRITE},
			},
		)

		downsample_pc := Bloom_Downsample_Push {
			src_texture = src_texture,
			src_sampler = rd.bloom_sampler_idx,
			src_lod     = src_lod,
			dst_image   = rd.bloom_scratch.bindless_idx,
			dst_width   = w,
			dst_height  = h,
		}
		vk.CmdPushConstants(
			cb,
			rd.bloom_downsample_pipeline.layout,
			{.COMPUTE},
			0,
			size_of(Bloom_Downsample_Push),
			&downsample_pc,
		)
		vk.CmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1)

		commit_mip(cb, &rd.bloom_scratch, &rd.bloom_image, i, w, h)
	}

	bind_compute_pipeline(cb, rd.bloom_upsample_pipeline)
	for i := int(rd.bloom_mip_count) - 2; i >= 0; i -= 1 {
		w := max(rd.bloom_base_width >> u32(i), 1)
		h := max(rd.bloom_base_height >> u32(i), 1)

		image_barriers(
			cb,
			{
				image = &rd.bloom_scratch,
				src_stage = {.TRANSFER},
				src_access = {.TRANSFER_READ},
				dst_stage = {.COMPUTE_SHADER},
				dst_access = {.SHADER_STORAGE_WRITE},
			},
		)

		upsample_pc := Bloom_Upsample_Push {
			src_texture   = rd.bloom_image.bindless_idx,
			src_sampler   = rd.bloom_sampler_idx,
			src_lod       = f32(i + 1),
			dst_lod       = f32(i),
			dst_image     = rd.bloom_scratch.bindless_idx,
			dst_width     = w,
			dst_height    = h,
			filter_radius = rd.frame.bloom_filter_radius,
		}
		vk.CmdPushConstants(
			cb,
			rd.bloom_upsample_pipeline.layout,
			{.COMPUTE},
			0,
			size_of(Bloom_Upsample_Push),
			&upsample_pc,
		)
		vk.CmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1)

		commit_mip(cb, &rd.bloom_scratch, &rd.bloom_image, u32(i), w, h)
	}
}

present_pass :: proc(resources: []Render_Resource, user_data: rawptr, cb: vk.CommandBuffer) {
	rd := cast(^Renderer)user_data

	color_attachment := vk.RenderingAttachmentInfo {
		sType = .RENDERING_ATTACHMENT_INFO,
		imageView = rd.frame.swapchain_view,
		imageLayout = .GENERAL,
		loadOp = .CLEAR,
		storeOp = .STORE,
		clearValue = {color = vk.ClearColorValue{float32 = [4]f32{0.0, 0.0, 0.0, 0.0}}},
	}
	// reuse the visbuffer pass' depth so probes are occluded by real geometry
	// present/ui don't use it, relying on VK_EXT_dynamic_rendering_unused_attachments
	depth_attachment := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = rd.depth_image.view,
		imageLayout = .GENERAL,
		loadOp      = .LOAD,
		storeOp     = .DONT_CARE,
	}
	rendering_info := vk.RenderingInfo {
		sType                = .RENDERING_INFO,
		renderArea           = rd.frame.render_area,
		layerCount           = 1,
		colorAttachmentCount = 1,
		pColorAttachments    = &color_attachment,
		pDepthAttachment     = &depth_attachment,
	}
	vk.CmdBeginRendering(cb, &rendering_info)
	vk.CmdSetViewportWithCount(cb, 1, &rd.frame.viewport)
	vk.CmdSetScissorWithCount(cb, 1, &rd.frame.render_area)

	present_pc := Present_Push {
		draw_image      = rd.draw_image.bindless_idx,
		bloom_texture   = rd.bloom_image.bindless_idx,
		bloom_sampler   = rd.bloom_sampler_idx,
		bloom_intensity = rd.frame.bloom_intensity,
		velocity_image  = rd.velocity_image.bindless_idx,
	}
	vk.CmdPushConstants(
		cb,
		rd.present_pipeline.layout,
		{.VERTEX, .FRAGMENT},
		0,
		size_of(Present_Push),
		&present_pc,
	)
	bind_raster_pipeline(cb, rd.present_pipeline)
	vk.CmdDraw(cb, 3, 1, 0, 0)

	if rd.frame.show_probes {
		probe_debug_pc := Probe_Debug_Push {
			frame_data            = rd.frame.frame_data_buffer.device_address,
			vertex_buffer         = rd.frame.gi.debug_sphere_vertex_buffer.device_address,
			normal_buffer         = rd.frame.gi.debug_sphere_normal_buffer.device_address,
			probe_position_buffer = rd.frame.gi.probe_position_buffer.device_address,
			probe_sh_buffer       = rd.frame.gi.probe_sh_buffer.device_address,
		}
		vk.CmdPushConstants(
			cb,
			rd.probe_debug_pipeline.layout,
			{.VERTEX, .FRAGMENT},
			0,
			size_of(Probe_Debug_Push),
			&probe_debug_pc,
		)
		bind_raster_pipeline(cb, rd.probe_debug_pipeline)
		vk.CmdDraw(cb, rd.frame.gi.debug_sphere_vertex_count, rd.frame.gi.probe_count, 0, 0)
	}

	ui_draw(cb)

	vk.CmdEndRendering(cb)
}

// ────────────────────────────────────────────────────────────────
// Frame
// the rest of rd.frame must already be filled in for this frame before calling

renderer_draw :: proc(
	rd: ^Renderer,
	cb: vk.CommandBuffer,
	swapchain_image: Swapchain_Image,
	scene: ^Scene,
	gi: ^Gi_System,
) {
	rd.frame.scene = scene
	rd.frame.gi = gi
	rd.frame.swapchain_view = swapchain_image.view

	render_graph_begin(&rd.rg)

	render_graph_add_pass(
		&rd.rg,
		"sky_bake",
		sky_bake_pass,
		rd,
		{
			target = &rd.sky_cubemap,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_WRITE}},
		},
	)

	if rd.frame.do_bake {
		render_graph_add_pass(
			&rd.rg,
			"probe_bake",
			probe_bake_pass,
			rd,
			{
				target = &rd.sky_cubemap,
				usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_SAMPLED_READ}},
			},
			{
				target = &rd.frame.gi.probe_sh_buffer,
				usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_WRITE}},
			},
		)
	}

	render_graph_add_pass(
		&rd.rg,
		"visbuffer",
		visbuffer_pass,
		rd,
		{
			target = &rd.visbuffer,
			usage = {stage = {.COLOR_ATTACHMENT_OUTPUT}, access = {.COLOR_ATTACHMENT_WRITE}},
		},
		{
			target = &rd.depth_image,
			usage = {
				stage = {.EARLY_FRAGMENT_TESTS, .LATE_FRAGMENT_TESTS},
				access = {.DEPTH_STENCIL_ATTACHMENT_WRITE},
			},
		},
	)

	render_graph_add_pass(
		&rd.rg,
		"motion_vectors",
		motion_vectors_pass,
		rd,
		{
			target = &rd.visbuffer,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_READ}},
		},
		{
			target = &rd.velocity_image,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_WRITE}},
		},
	)

	render_graph_add_pass(
		&rd.rg,
		"rtao",
		rtao_pass,
		rd,
		{
			target = &rd.visbuffer,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_READ}},
		},
		{
			target = &rd.rtao_image,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_WRITE}},
		},
	)

	render_graph_add_pass(
		&rd.rg,
		"shading",
		shading_pass,
		rd,
		{
			target = &rd.visbuffer,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_READ}},
		},
		{
			target = &rd.rtao_image,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_SAMPLED_READ}},
		},
		{
			target = &rd.sky_cubemap,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_SAMPLED_READ}},
		},
		{
			target = &rd.frame.gi.probe_sh_buffer,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_READ}},
		},
		{
			target = &rd.draw_image,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_STORAGE_WRITE}},
		},
	)

	render_graph_add_pass(
		&rd.rg,
		"bloom",
		bloom_pass,
		rd,
		// bloom_downsample samples draw_image through a plain texture view (mip 0 only)
		{
			target = &rd.draw_image,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_SAMPLED_READ}},
		},
		{
			target = &rd.bloom_image,
			usage = {stage = {.COMPUTE_SHADER}, access = {.SHADER_SAMPLED_READ}},
		},
	)

	render_graph_add_pass(
		&rd.rg,
		"present",
		present_pass,
		rd,
		// present.glsl reads draw_image back through the storage view, not the texture one
		{
			target = &rd.draw_image,
			usage = {stage = {.FRAGMENT_SHADER}, access = {.SHADER_STORAGE_READ}},
		},
		{
			target = &rd.bloom_image,
			usage = {stage = {.FRAGMENT_SHADER}, access = {.SHADER_SAMPLED_READ}},
		},
		{
			target = &rd.velocity_image,
			usage = {stage = {.FRAGMENT_SHADER}, access = {.SHADER_STORAGE_READ}},
		},
		{
			target = &rd.frame.gi.probe_sh_buffer,
			usage = {stage = {.FRAGMENT_SHADER}, access = {.SHADER_STORAGE_READ}},
		},
		{
			target = &rd.depth_image,
			usage = {
				stage = {.EARLY_FRAGMENT_TESTS, .LATE_FRAGMENT_TESTS},
				access = {.DEPTH_STENCIL_ATTACHMENT_READ, .DEPTH_STENCIL_ATTACHMENT_WRITE},
			},
		},
	)

	render_graph_execute(&rd.rg, cb)
}
