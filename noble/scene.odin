package noble

import "core:fmt"
import "core:math/linalg/glsl"
import "core:os"
import "core:slice"
import vk "vendor:vulkan"


Scene :: struct {
	triangle_count:      u32,
	draw_count:          u32,
	index_buffer:        Buffer,
	position_buffer:     Buffer,
	draw_data_buffer:    Buffer,
	draw_command_buffer: Buffer,
}

Header :: struct {
	positions_offset:   u32,
	positions_size:     u32,
	normals_offset:     u32,
	normals_size:       u32,
	uvs_offset:         u32,
	uvs_size:           u32,
	indices_offset:     u32,
	indices_size:       u32,
	materials_offset:   u32,
	materials_size:     u32,
	renderables_offset: u32,
	renderables_size:   u32,
}

Material :: struct {
	color: glsl.vec3,
}

Draw_Data :: struct #align (16) {
	transform:     glsl.mat4,
	material_idx:  u32,
	triangle_base: u32,
}

Renderable :: struct {
	transform:    [16]f32,
	material_idx: u32,
	index_offset: u32,
	index_count:  u32,
}

scene_init :: proc(scene: ^Scene, device: ^Device, path: string) {
	fmt.println("[Scene] Loading", path)

	data, err := os.read_entire_file(path, context.temp_allocator)
	if data == nil {
		fmt.println("[Scene] Failed to open", path)
		return
	}

	header := (^Header)(raw_data(data))^
	scene.triangle_count = (header.indices_size / size_of(u32)) / 3
	fmt.println("- triangle count :", scene.triangle_count)

	handle, cb := command_handler_acquire(&device.command_handler)

	temp_pool := make([dynamic]Buffer, context.temp_allocator)

	// positions
	positions_bytes := data[header.positions_offset:header.positions_offset +
	header.positions_size]
	positions := slice.reinterpret([]f32, positions_bytes)
	scene.position_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(positions),
		{
			size = vk.DeviceSize(header.positions_size),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)

	// indices
	indices_bytes := data[header.indices_offset:header.indices_offset + header.indices_size]
	indices := slice.reinterpret([]u32, indices_bytes)
	scene.index_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(indices),
		{
			size = vk.DeviceSize(header.indices_size),
			usage = {.INDEX_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)

	// renderables
	renderables_bytes := data[header.renderables_offset:header.renderables_offset +
	header.renderables_size]
	renderables := ([^]Renderable)(raw_data(renderables_bytes))

	renderable_count := header.renderables_size / size_of(Renderable)
	draw_data := make([]Draw_Data, renderable_count)
	draw_commands := make([]vk.DrawIndexedIndirectCommand, renderable_count)

	triangle_base: u32 = 0
	for i in 0 ..< renderable_count {
		rend := renderables[i]
		draw_data[i] = {
			transform     = make_transform(rend.transform),
			material_idx  = i,
			triangle_base = triangle_base,
		}
		draw_commands[i] = vk.DrawIndexedIndirectCommand {
			firstInstance = 0,
			instanceCount = 1,
			firstIndex    = rend.index_offset,
			indexCount    = rend.index_count,
			vertexOffset  = 0,
		}

		triangle_base += rend.index_count / 3
	}
	defer {
		delete(draw_data)
		delete(draw_commands)
	}

	scene.draw_data_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(draw_data),
		{
			size = vk.DeviceSize(len(draw_data) * size_of(Draw_Data)),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)
	scene.draw_command_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(draw_commands),
		{
			size = vk.DeviceSize(len(draw_commands) * size_of(vk.DrawIndexedIndirectCommand)),
			usage = {.INDIRECT_BUFFER},
			memory = .GPU_ONLY,
		},
	)
	scene.draw_count = u32(len(draw_commands))
	fmt.println("- draw count     :", scene.draw_count)

	command_handler_submit(&device.command_handler, handle, false)
	command_handler_wait(&device.command_handler, handle)

	for &t in temp_pool {
		destroy_buffer(device, &t)
	}
	delete(temp_pool)
}

scene_cleanup :: proc(scene: ^Scene, device: ^Device) {
	destroy_buffer(device, &scene.position_buffer)
	destroy_buffer(device, &scene.index_buffer)
	destroy_buffer(device, &scene.draw_data_buffer)
	destroy_buffer(device, &scene.draw_command_buffer)
}

@(private = "file")
make_transform :: proc(t: [16]f32) -> glsl.mat4 {
	return {
		t[0],
		t[1],
		t[2],
		t[3],
		t[4],
		t[5],
		t[6],
		t[7],
		t[8],
		t[9],
		t[10],
		t[11],
		t[12],
		t[13],
		t[14],
		t[15],
	}
}
