package luma

import "base:intrinsics"
import "core:fmt"
import "core:math/linalg/glsl"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"
import stbi "vendor:stb/image"
import vk "vendor:vulkan"

Scene :: struct {
	triangle_count:      u32,
	draw_count:          u32,
	index_buffer:        Buffer,
	position_buffer:     Buffer,
	normal_buffer:       Buffer,
	tangent_buffer:      Buffer,
	uv_buffer:           Buffer,
	material_buffer:     Buffer,
	draw_data_buffer:    Buffer,
	draw_command_buffer: Buffer,
	texture_images:      []Image,
	blas:                []Acceleration_Structure,
	tlas:                Acceleration_Structure,
}

Header :: struct {
	positions_offset:   u32,
	positions_size:     u32,
	normals_offset:     u32,
	normals_size:       u32,
	tangents_offset:    u32,
	tangents_size:      u32,
	uvs_offset:         u32,
	uvs_size:           u32,
	indices_offset:     u32,
	indices_size:       u32,
	textures_offset:    u32,
	textures_size:      u32,
	materials_offset:   u32,
	materials_size:     u32,
	renderables_offset: u32,
	renderables_size:   u32,
}

Material :: struct {
	base_color:                 glsl.vec3,
	base_color_tex_idx:         i32,
	metallic_roughness_tex_idx: i32,
	normal_tex_idx:             i32,
}

Material_GPU :: struct #align (16) {
	base_color:             glsl.vec3,
	base_color_tex:         i32,
	normal_tex:             i32,
	metallic_roughness_tex: i32,
}

Draw_Data :: struct #align (16) {
	transform:     glsl.mat4,
	material_idx:  i32,
	triangle_base: u32,
}

Renderable :: struct {
	transform:    [16]f32,
	material_idx: i32,
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

	// positions + normals + uvs
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
			usage = {
				.STORAGE_BUFFER,
				.SHADER_DEVICE_ADDRESS,
				.ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
			},
			memory = .GPU_ONLY,
		},
	)

	normals_bytes := data[header.normals_offset:header.normals_offset + header.normals_size]
	normals := slice.reinterpret([]f32, normals_bytes)
	scene.normal_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(normals),
		{
			size = vk.DeviceSize(header.normals_size),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)

	tangents_bytes := data[header.tangents_offset:header.tangents_offset + header.tangents_size]
	tangents := slice.reinterpret([]f32, tangents_bytes)
	scene.tangent_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(tangents),
		{
			size = vk.DeviceSize(header.tangents_size),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)

	uvs_bytes := data[header.uvs_offset:header.uvs_offset + header.uvs_size]
	uvs := slice.reinterpret([]f32, uvs_bytes)
	scene.uv_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(uvs),
		{
			size = vk.DeviceSize(header.uvs_size),
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
			usage = {
				.INDEX_BUFFER,
				.SHADER_DEVICE_ADDRESS,
				.ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
			},
			memory = .GPU_ONLY,
		},
	)

	// pre-scan materials to identify which texture indices hold linear (non-color) data,
	// such as normal and metallic-roughness textures, as opposed to sRGB color textures
	materials_bytes := data[header.materials_offset:header.materials_offset +
	header.materials_size]
	materials := slice.reinterpret([]Material, materials_bytes)

	linear_data_indices := make(map[i32]bool, context.temp_allocator)
	for mat in materials {
		if mat.normal_tex_idx >= 0 {
			linear_data_indices[mat.normal_tex_idx] = true
		}
		if mat.metallic_roughness_tex_idx >= 0 {
			linear_data_indices[mat.metallic_roughness_tex_idx] = true
		}
	}

	// textures
	textures_bytes := data[header.textures_offset:header.textures_offset + header.textures_size]
	textures := parse_textures(string(textures_bytes))
	defer delete(textures)

	scene.texture_images = make([]Image, len(textures))
	file_dir := filepath.dir(path, context.temp_allocator)
	for tex_name, i in textures {
		tex_path, _ := filepath.join({file_dir, tex_name}, context.temp_allocator)
		tex_path_cstr := strings.clone_to_cstring(tex_path, context.temp_allocator)

		tex_width, tex_height, tex_channels: i32
		tex_data := stbi.load(tex_path_cstr, &tex_width, &tex_height, &tex_channels, 4)
		if tex_data == nil {
			fmt.println("[Scene] Failed to load texture", tex_path)
			continue
		}
		defer stbi.image_free(tex_data)

		format: vk.Format = .R8G8B8A8_SRGB
		if i32(i) in linear_data_indices {
			format = .R8G8B8A8_UNORM
		}

		scene.texture_images[i] = create_and_upload_image(
			device,
			&temp_pool,
			cb,
			tex_data,
			vk.DeviceSize(tex_width * tex_height * 4),
			{
				width = u32(tex_width),
				height = u32(tex_height),
				format = format,
				usage = {.SAMPLED, .TRANSFER_SRC},
				memory = .GPU_ONLY,
				register_bindless = .Texture,
				mips = true,
			},
		)
	}

	materials_gpu := make([]Material_GPU, len(materials))
	for mat, i in materials {
		materials_gpu[i] = {
			base_color             = mat.base_color,
			base_color_tex         = get_texture_bindless_idx(
				scene.texture_images,
				mat.base_color_tex_idx,
			),
			normal_tex             = get_texture_bindless_idx(
				scene.texture_images,
				mat.normal_tex_idx,
			),
			metallic_roughness_tex = get_texture_bindless_idx(
				scene.texture_images,
				mat.metallic_roughness_tex_idx,
			),
		}
	}
	defer delete(materials_gpu)

	scene.material_buffer = create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(materials_gpu),
		{
			size = vk.DeviceSize(len(materials_gpu) * size_of(Material_GPU)),
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			memory = .GPU_ONLY,
		},
	)

	// renderables, BLAS and TLAS instances
	renderables_bytes := data[header.renderables_offset:header.renderables_offset +
	header.renderables_size]
	renderables := ([^]Renderable)(raw_data(renderables_bytes))

	renderable_count := header.renderables_size / size_of(Renderable)
	draw_data := make([]Draw_Data, renderable_count, context.temp_allocator)
	draw_commands := make(
		[]vk.DrawIndexedIndirectCommand,
		renderable_count,
		context.temp_allocator,
	)

	// position/index buffer uploads above must finish before the BLAS builds below read them
	buffer_barriers(
		cb,
		{
			buffer = &scene.position_buffer,
			src_stage = {.TRANSFER},
			src_access = {.TRANSFER_WRITE},
			dst_stage = {.ACCELERATION_STRUCTURE_BUILD_KHR},
			dst_access = {.ACCELERATION_STRUCTURE_READ_KHR},
		},
		{
			buffer = &scene.index_buffer,
			src_stage = {.TRANSFER},
			src_access = {.TRANSFER_WRITE},
			dst_stage = {.ACCELERATION_STRUCTURE_BUILD_KHR},
			dst_access = {.ACCELERATION_STRUCTURE_READ_KHR},
		},
	)

	blas_triangles := vk.AccelerationStructureGeometryTrianglesDataKHR {
		sType = .ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		vertexFormat = .R32G32B32_SFLOAT,
		vertexData = {deviceAddress = scene.position_buffer.device_address},
		vertexStride = size_of(glsl.vec3),
		maxVertex = u32(len(positions) / 3) - 1,
		indexType = .UINT32,
		indexData = {deviceAddress = scene.index_buffer.device_address},
	}
	blas_geometry := vk.AccelerationStructureGeometryKHR {
		sType = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		geometryType = .TRIANGLES,
		geometry = {triangles = blas_triangles},
		flags = {.NO_DUPLICATE_ANY_HIT_INVOCATION, .OPAQUE},
	}
	scene.blas = make([]Acceleration_Structure, renderable_count)

	tlas_instances := make(
		[]vk.AccelerationStructureInstanceKHR,
		renderable_count,
		context.temp_allocator,
	)

	triangle_base: u32 = 0
	for i in 0 ..< renderable_count {
		rend := renderables[i]
		draw_data[i] = {
			transform     = make_transform(rend.transform),
			material_idx  = rend.material_idx,
			triangle_base = triangle_base,
		}
		draw_commands[i] = vk.DrawIndexedIndirectCommand {
			firstInstance = 0,
			instanceCount = 1,
			firstIndex    = rend.index_offset,
			indexCount    = rend.index_count,
			vertexOffset  = 0,
		}

		triangle_count := rend.index_count / 3
		triangle_base += triangle_count

		// one per renderable / mesh for now
		blas_range_info := vk.AccelerationStructureBuildRangeInfoKHR {
			primitiveOffset = rend.index_offset * size_of(u32),
			primitiveCount  = triangle_count,
		}
		blas_build_info := vk.AccelerationStructureBuildGeometryInfoKHR {
			sType         = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
			type          = .BOTTOM_LEVEL,
			flags         = {.PREFER_FAST_TRACE},
			mode          = .BUILD,
			geometryCount = 1,
			pGeometries   = &blas_geometry,
		}

		scene.blas[i] = build_acceleration_structure(
			device,
			&temp_pool,
			cb,
			blas_build_info,
			blas_range_info,
		)

		tlas_instances[i] = {
			transform                              = to_transform_matrixKHR(&rend.transform),
			instanceCustomIndex                    = i,
			instanceShaderBindingTableRecordOffset = 0,
			flags                                  = .TRIANGLE_CULL_DISABLE_NV,
			mask                                   = 0xFF,
			accelerationStructureReference         = u64(scene.blas[i].device_address),
		}
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

	// TLAS
	tlas_instances_buffer := create_and_upload_buffer(
		device,
		&temp_pool,
		cb,
		raw_data(tlas_instances),
		{
			size = vk.DeviceSize(
				len(tlas_instances) * size_of(vk.AccelerationStructureInstanceKHR),
			),
			memory = .GPU_ONLY,
			usage = {.ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .SHADER_DEVICE_ADDRESS},
		},
	)
	append(&temp_pool, tlas_instances_buffer)
	buffer_barriers(
		cb,
		{
			buffer = &tlas_instances_buffer,
			src_stage = {.TRANSFER},
			src_access = {.TRANSFER_WRITE},
			dst_stage = {.ACCELERATION_STRUCTURE_BUILD_KHR},
			dst_access = {.ACCELERATION_STRUCTURE_READ_KHR},
		},
	)
	geometry_instances := vk.AccelerationStructureGeometryInstancesDataKHR {
		sType = .ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
		data = {deviceAddress = tlas_instances_buffer.device_address},
	}
	tlas_geometry := vk.AccelerationStructureGeometryKHR {
		sType = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		geometryType = .INSTANCES,
		geometry = {instances = geometry_instances},
	}
	tlas_range_info := vk.AccelerationStructureBuildRangeInfoKHR {
		primitiveCount = renderable_count,
	}

	// BLAS builds above must finish before the TLAS build reads them via the instance buffer
	accel_barrier := vk.MemoryBarrier2 {
		sType         = .MEMORY_BARRIER_2,
		srcStageMask  = {.ACCELERATION_STRUCTURE_BUILD_KHR},
		srcAccessMask = {.ACCELERATION_STRUCTURE_WRITE_KHR},
		dstStageMask  = {.ACCELERATION_STRUCTURE_BUILD_KHR},
		dstAccessMask = {.ACCELERATION_STRUCTURE_READ_KHR},
	}
	accel_dep := vk.DependencyInfo {
		sType              = .DEPENDENCY_INFO,
		memoryBarrierCount = 1,
		pMemoryBarriers    = &accel_barrier,
	}
	vk.CmdPipelineBarrier2(cb, &accel_dep)

	tlas_build_info := vk.AccelerationStructureBuildGeometryInfoKHR {
		sType         = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		type          = .TOP_LEVEL,
		flags         = {.PREFER_FAST_TRACE},
		mode          = .BUILD,
		geometryCount = 1,
		pGeometries   = &tlas_geometry,
	}
	scene.tlas = build_acceleration_structure(
		device,
		&temp_pool,
		cb,
		tlas_build_info,
		tlas_range_info,
	)
	write_tlas_descriptor(device, &scene.tlas)

	command_handler_submit(&device.command_handler, handle, false)
	command_handler_wait(&device.command_handler, handle)

	for &t in temp_pool {
		destroy_buffer(device, &t)
	}
	delete(temp_pool)
}

scene_cleanup :: proc(scene: ^Scene, device: ^Device) {
	for &blas in scene.blas {
		destroy_acceleration_structure(device, &blas)
	}
	delete(scene.blas)
	destroy_acceleration_structure(device, &scene.tlas)
	destroy_buffer(device, &scene.position_buffer)
	destroy_buffer(device, &scene.normal_buffer)
	destroy_buffer(device, &scene.tangent_buffer)
	destroy_buffer(device, &scene.uv_buffer)
	destroy_buffer(device, &scene.index_buffer)
	destroy_buffer(device, &scene.material_buffer)
	destroy_buffer(device, &scene.draw_data_buffer)
	destroy_buffer(device, &scene.draw_command_buffer)
	for image in scene.texture_images {
		destroy_image(device, image)
	}
	delete(scene.texture_images)
}

// textures are stored as a comma-separated list of quoted names, e.g. `"a","b"`
@(private = "file")
parse_textures :: proc(s: string) -> []string {
	if len(s) == 0 {
		return {}
	}

	parts := strings.split(s, ",", context.temp_allocator)
	names := make([]string, len(parts))
	for part, i in parts {
		names[i] = strings.trim(part, `"`)
	}
	return names
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

@(private = "file")
get_texture_bindless_idx :: proc(images: []Image, idx: i32) -> i32 {
	if idx == -1 {
		return -1
	}

	image := images[idx]
	return i32(image.bindless_idx) if image.image != 0 else -1
}

@(private = "file")
to_transform_matrixKHR :: proc(m: ^[16]f32) -> (t: vk.TransformMatrixKHR) {
	intrinsics.mem_copy(&t, m, size_of(vk.TransformMatrixKHR))
	return t
}
