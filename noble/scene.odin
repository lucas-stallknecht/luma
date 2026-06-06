package noble

import "core:fmt"
import "core:math/linalg/glsl"
import "core:os"
import "core:slice"
import vk "vendor:vulkan"


Scene :: struct {
	vertex_count:    u32,
	index_count:     u32,
	index_buffer:    Buffer,
	position_buffer: Buffer,
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

Renderable :: struct {
	transform:    glsl.mat4x4,
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
	scene.vertex_count = header.positions_size / (3 * size_of(f32))
	scene.index_count = header.indices_size / size_of(u32)
	fmt.println("- vertex count    :", scene.vertex_count)
	fmt.println("- indices count    :", scene.index_count)

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
}
