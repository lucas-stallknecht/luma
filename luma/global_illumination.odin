package luma

import "core:fmt"
import "core:math/linalg/glsl"
import "core:os"
import "core:slice"
import vk "vendor:vulkan"

Gi_System :: struct {
	info:                       Gi_System_Info,
	probe_count:                u32,
	grid_spacing:               glsl.vec3,
	probe_position_buffer:      Buffer,
	probe_sh_buffer:            Buffer,
	debug_sphere_vertex_buffer: Buffer,
	debug_sphere_normal_buffer: Buffer,
	debug_sphere_vertex_count:  u32,
}

Gi_System_Info :: struct {
	probe_counts: [3]u32,
	grid_min:     glsl.vec3,
	grid_max:     glsl.vec3,
}

gi_system_init :: proc(
	gi: ^Gi_System,
	device: ^Device,
	probe_debug_path: string,
	info: Gi_System_Info,
) -> bool {
	fmt.println("[Gi] Loading", probe_debug_path)

	data, err := os.read_entire_file(probe_debug_path, context.temp_allocator)
	if err != nil {
		fmt.eprintln("[Gi] Failed to open", probe_debug_path, "-", err)
		return false
	}
	if len(data) < size_of(Header) {
		fmt.eprintfln(
			"[Gi] %q is too small to contain a header (%d bytes)",
			probe_debug_path,
			len(data),
		)
		return false
	}

	header := (^Header)(raw_data(data))^
	if !section_in_bounds(
		   len(data),
		   header.positions_offset,
		   header.positions_size,
		   "Gi",
		   "positions",
	   ) ||
	   !section_in_bounds(len(data), header.normals_offset, header.normals_size, "Gi", "normals") {
		return false
	}

	handle, cb := command_handler_acquire(&device.command_handler)
	temp_pool := make([dynamic]Buffer, context.temp_allocator)

	gi.info = info
	spacings: [3]f32 = {}
	for i in 0 ..< 3 {
		// a single probe along an axis has no spacing; guard the divide
		spacings[i] =
			(info.grid_max[i] - info.grid_min[i]) / f32(info.probe_counts[i] - 1) if info.probe_counts[i] > 1 else 0
	}
	gi.grid_spacing = spacings
	gi.probe_count = info.probe_counts[0] * info.probe_counts[1] * info.probe_counts[2]
	fmt.println("[Gi] Total probe count:", gi.probe_count)

	positions := make([]glsl.vec3, gi.probe_count, context.temp_allocator)
	for z in 0 ..< info.probe_counts[2] {
		for y in 0 ..< info.probe_counts[1] {
			for x in 0 ..< info.probe_counts[0] {
				idx :=
					x + y * info.probe_counts[0] + z * info.probe_counts[0] * info.probe_counts[1]
				positions[idx] = glsl.vec3 {
					info.grid_min.x + f32(x) * spacings.x,
					info.grid_min.y + f32(y) * spacings.y,
					info.grid_min.z + f32(z) * spacings.z,
				}
			}
		}
	}
	gi.probe_position_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(positions),
		{
			size = vk.DeviceSize(int(gi.probe_count) * size_of(glsl.vec3)),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)
	// spherical harmonics coefficients baked on the GPU.
	gi.probe_sh_buffer = create_buffer(
		device,
		{
			size = vk.DeviceSize(int(gi.probe_count) * 4 * size_of(glsl.vec3)),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS, .TRANSFER_DST},
			memory = .GPU_ONLY,
		},
	)
	// zero it so the first bake's feedback read sees a black field instead of garbage
	vk.CmdFillBuffer(cb, gi.probe_sh_buffer.buffer, 0, vk.DeviceSize(vk.WHOLE_SIZE), 0)

	// debug model
	vertex_positions_bytes := data[header.positions_offset:header.positions_offset +
	header.positions_size]
	vertex_positions := slice.reinterpret([]f32, vertex_positions_bytes)
	gi.debug_sphere_vertex_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(vertex_positions),
		{
			size = vk.DeviceSize(header.positions_size),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)
	gi.debug_sphere_vertex_count = u32(len(vertex_positions) / 3)

	vertex_normals_bytes := data[header.normals_offset:header.normals_offset + header.normals_size]
	vertex_normals := slice.reinterpret([]f32, vertex_normals_bytes)
	gi.debug_sphere_normal_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(vertex_normals),
		{
			size = vk.DeviceSize(header.normals_size),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)

	command_handler_submit(&device.command_handler, handle, false)
	command_handler_wait(&device.command_handler, handle)

	for &t in temp_pool {
		destroy_buffer(device, &t)
	}
	delete(temp_pool)

	return true
}

gi_system_cleanup :: proc(gi: ^Gi_System, device: ^Device) {
	destroy_buffer(device, &gi.probe_position_buffer)
	destroy_buffer(device, &gi.probe_sh_buffer)
	destroy_buffer(device, &gi.debug_sphere_vertex_buffer)
	destroy_buffer(device, &gi.debug_sphere_normal_buffer)
}
