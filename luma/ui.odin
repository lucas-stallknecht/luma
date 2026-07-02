package luma

import "core:math/linalg/glsl"
import mu "vendor:microui"
import vk "vendor:vulkan"

UI_MAX_VERTICES :: 1 << 16

Ui_Vertex :: struct {
	pos:   glsl.vec2,
	uv:    glsl.vec2,
	color: glsl.vec4,
}

Ui_Push :: struct {
	vertex_buffer: vk.DeviceAddress,
	screen_size:   glsl.vec2,
	atlas_texture: u32,
	atlas_sampler: u32,
}

Ui :: struct {
	ctx:            mu.Context,
	atlas_image:    Image,
	sampler:        vk.Sampler,
	sampler_idx:    u32,
	vertex_buffers: [MAX_COMMAND_BUFFERS]Buffer,
	vertex_mapped:  [MAX_COMMAND_BUFFERS]rawptr,
	pipeline:       ^Raster_Pipeline,
}

ui_init :: proc(
	ui: ^Ui,
	device: ^Device,
	pipeline_manager: ^Pipeline_Manager,
	color_format: vk.Format,
) {
	mu.init(&ui.ctx)
	ui.ctx.text_width = mu.default_atlas_text_width
	ui.ctx.text_height = mu.default_atlas_text_height

	// nearest filtering avoids sampling across glyph/icon boundaries in the packed atlas
	sampler_ci := vk.SamplerCreateInfo {
		sType        = .SAMPLER_CREATE_INFO,
		magFilter    = .NEAREST,
		minFilter    = .NEAREST,
		addressModeU = .CLAMP_TO_EDGE,
		addressModeV = .CLAMP_TO_EDGE,
		addressModeW = .CLAMP_TO_EDGE,
	}
	chk(vk.CreateSampler(device.device, &sampler_ci, nil, &ui.sampler))
	ui.sampler_idx = bindless_register_sampler(device, ui.sampler)

	handle, cb := command_handler_acquire(&device.command_handler)
	temp_pool := make([dynamic]Buffer, context.temp_allocator)

	ui.atlas_image = create_and_upload_image(
		device,
		&temp_pool,
		cb,
		raw_data(mu.default_atlas_alpha[:]),
		vk.DeviceSize(len(mu.default_atlas_alpha)),
		{
			width = mu.DEFAULT_ATLAS_WIDTH,
			height = mu.DEFAULT_ATLAS_HEIGHT,
			format = .R8_UNORM,
			usage = {.SAMPLED},
			memory = .GPU_ONLY,
			register_bindless = .Texture,
		},
	)

	command_handler_submit(&device.command_handler, handle, false)
	command_handler_wait(&device.command_handler, handle)
	for &t in temp_pool {
		destroy_buffer(device, &t)
	}

	for i in 0 ..< int(MAX_COMMAND_BUFFERS) {
		ui.vertex_buffers[i] = create_buffer(
			device,
			{
				size = vk.DeviceSize(UI_MAX_VERTICES * size_of(Ui_Vertex)),
				usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
				memory = .CPU_UPLOAD,
			},
		)
		vk.MapMemory(
			device.device,
			ui.vertex_buffers[i].memory,
			0,
			vk.DeviceSize(UI_MAX_VERTICES * size_of(Ui_Vertex)),
			{},
			&ui.vertex_mapped[i],
		)
	}

	ui.pipeline = pipeline_manager_add_raster(
		pipeline_manager,
		{
			name = "ui",
			vertex_shader = "ui.vert",
			fragment_shader = "ui.frag",
			raster = {primitive_topology = .TRIANGLE_LIST},
			push_constant_size = size_of(Ui_Push),
			color_attachments = {{format = color_format, blend = .Alpha}},
		},
	)
}

ui_cleanup :: proc(ui: ^Ui, device: ^Device) {
	for i in 0 ..< int(MAX_COMMAND_BUFFERS) {
		vk.UnmapMemory(device.device, ui.vertex_buffers[i].memory)
		destroy_buffer(device, &ui.vertex_buffers[i])
	}
	destroy_image(device, ui.atlas_image)
	if ui.sampler != 0 {
		vk.DestroySampler(device.device, ui.sampler, nil)
	}
}

@(private = "file")
Ui_Batch :: struct {
	clip:         vk.Rect2D,
	first_vertex: u32,
	vertex_count: u32,
}

@(private = "file")
ui_clip_to_screen :: proc(rect: mu.Rect, width, height: u32) -> vk.Rect2D {
	x0 := clamp(rect.x, 0, i32(width))
	y0 := clamp(rect.y, 0, i32(height))
	x1 := clamp(rect.x + rect.w, 0, i32(width))
	y1 := clamp(rect.y + rect.h, 0, i32(height))
	return {offset = {x0, y0}, extent = {u32(max(x1 - x0, 0)), u32(max(y1 - y0, 0))}}
}

@(private = "file")
ui_color_to_vec4 :: proc(color: mu.Color) -> glsl.vec4 {
	return {f32(color.r) / 255, f32(color.g) / 255, f32(color.b) / 255, f32(color.a) / 255}
}

ui_color_rect :: proc(ctx: ^mu.Context, rect: mu.Rect, rgb: glsl.vec3) {
	mu.draw_rect(
		ctx,
		rect,
		{
			u8(clamp(rgb.r, 0, 1) * 255),
			u8(clamp(rgb.g, 0, 1) * 255),
			u8(clamp(rgb.b, 0, 1) * 255),
			255,
		},
	)
}

ui_layout_cursor_y :: proc(ctx: ^mu.Context) -> i32 {
	layout := mu.get_layout(ctx)
	return layout.body.y + layout.position.y
}

// spans from `top` down to the last widget placed since, right-aligned in the container
ui_swatch_rect :: proc(ctx: ^mu.Context, top: i32, width: i32) -> mu.Rect {
	layout := mu.get_layout(ctx)
	row_height := ctx.style.size.y + ctx.style.padding * 2 + ctx.style.spacing
	bottom := layout.body.y + layout.position.y - ctx.style.spacing + row_height
	return {layout.body.x + layout.body.w - width, top, width, bottom - top}
}

@(private = "file")
ui_push_quad :: proc(
	verts: ^[UI_MAX_VERTICES]Ui_Vertex,
	vertex_count: ^u32,
	batch: ^Ui_Batch,
	rect: mu.Rect,
	uv_min, uv_max: glsl.vec2,
	color: glsl.vec4,
) {
	if vertex_count^ + 6 > UI_MAX_VERTICES do return

	x0, y0 := f32(rect.x), f32(rect.y)
	x1, y1 := f32(rect.x + rect.w), f32(rect.y + rect.h)
	positions := [4]glsl.vec2{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}
	uvs := [4]glsl.vec2{uv_min, {uv_max.x, uv_min.y}, uv_max, {uv_min.x, uv_max.y}}
	order := [6]int{0, 1, 2, 0, 2, 3}

	for idx in order {
		verts[vertex_count^] = {
			pos   = positions[idx],
			uv    = uvs[idx],
			color = color,
		}
		vertex_count^ += 1
	}
	batch.vertex_count += 6
}

// call after mu.end(&ui.ctx), inside an active render pass
ui_render :: proc(ui: ^Ui, cb: vk.CommandBuffer, buffer_idx: u8, width, height: u32) {
	verts := (^[UI_MAX_VERTICES]Ui_Vertex)(ui.vertex_mapped[buffer_idx])
	vertex_count: u32 = 0

	batches := make([dynamic]Ui_Batch, context.temp_allocator)
	append(&batches, Ui_Batch{clip = {extent = {width, height}}})

	atlas_w := f32(mu.DEFAULT_ATLAS_WIDTH)
	atlas_h := f32(mu.DEFAULT_ATLAS_HEIGHT)

	cmd: ^mu.Command
	for variant in mu.next_command_iterator(&ui.ctx, &cmd) {
		switch v in variant {
		case ^mu.Command_Clip:
			r := ui_clip_to_screen(v.rect, width, height)
			last := &batches[len(batches) - 1]
			if last.vertex_count > 0 {
				append(&batches, Ui_Batch{clip = r, first_vertex = vertex_count})
			} else {
				last.clip = r
			}

		case ^mu.Command_Rect:
			white := mu.default_atlas[mu.DEFAULT_ATLAS_WHITE]
			uv := glsl.vec2 {
				(f32(white.x) + f32(white.w) * 0.5) / atlas_w,
				(f32(white.y) + f32(white.h) * 0.5) / atlas_h,
			}
			batch := &batches[len(batches) - 1]
			ui_push_quad(verts, &vertex_count, batch, v.rect, uv, uv, ui_color_to_vec4(v.color))

		case ^mu.Command_Text:
			color := ui_color_to_vec4(v.color)
			x := v.pos.x
			batch := &batches[len(batches) - 1]
			for ch in v.str {
				if ch & 0xc0 == 0x80 do continue
				gi := min(int(ch), 127)
				src := mu.default_atlas[mu.DEFAULT_ATLAS_FONT + gi]
				quad := mu.Rect{x, v.pos.y, src.w, src.h}
				uv_min := glsl.vec2{f32(src.x) / atlas_w, f32(src.y) / atlas_h}
				uv_max := glsl.vec2{f32(src.x + src.w) / atlas_w, f32(src.y + src.h) / atlas_h}
				ui_push_quad(verts, &vertex_count, batch, quad, uv_min, uv_max, color)
				x += src.w
			}

		case ^mu.Command_Icon:
			src := mu.default_atlas[int(v.id)]
			cx := v.rect.x + (v.rect.w - src.w) / 2
			cy := v.rect.y + (v.rect.h - src.h) / 2
			quad := mu.Rect{cx, cy, src.w, src.h}
			uv_min := glsl.vec2{f32(src.x) / atlas_w, f32(src.y) / atlas_h}
			uv_max := glsl.vec2{f32(src.x + src.w) / atlas_w, f32(src.y + src.h) / atlas_h}
			batch := &batches[len(batches) - 1]
			ui_push_quad(
				verts,
				&vertex_count,
				batch,
				quad,
				uv_min,
				uv_max,
				ui_color_to_vec4(v.color),
			)

		case ^mu.Command_Jump:
		}
	}

	if vertex_count == 0 do return

	push := Ui_Push {
		vertex_buffer = ui.vertex_buffers[buffer_idx].device_address,
		screen_size   = {f32(width), f32(height)},
		atlas_texture = ui.atlas_image.bindless_idx,
		atlas_sampler = ui.sampler_idx,
	}
	vk.CmdPushConstants(cb, ui.pipeline.layout, {.VERTEX, .FRAGMENT}, 0, size_of(Ui_Push), &push)
	bind_raster_pipeline(cb, ui.pipeline)

	for batch in batches {
		if batch.vertex_count == 0 do continue
		scissor := batch.clip
		vk.CmdSetScissorWithCount(cb, 1, &scissor)
		vk.CmdDraw(cb, batch.vertex_count, 1, batch.first_vertex, 0)
	}
}
