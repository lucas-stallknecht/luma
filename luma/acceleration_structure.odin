package luma

import vk "vendor:vulkan"

Acceleration_Structure :: struct {
	handle:         vk.AccelerationStructureKHR,
	buffer:         Buffer,
	device_address: vk.DeviceAddress,
}

create_acceleration_structure :: proc(
	device: ^Device,
	build_size: vk.AccelerationStructureBuildSizesInfoKHR,
	type: vk.AccelerationStructureTypeKHR,
) -> Acceleration_Structure {
	out: Acceleration_Structure

	out.buffer = create_buffer(
		device,
		{
			size = build_size.accelerationStructureSize,
			memory = .GPU_ONLY,
			usage = {.ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS},
		},
	)

	create_info := vk.AccelerationStructureCreateInfoKHR {
		sType  = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		buffer = out.buffer.buffer,
		size   = build_size.accelerationStructureSize,
		type   = type,
	}
	chk(vk.CreateAccelerationStructureKHR(device.device, &create_info, nil, &out.handle))

	address_info := vk.AccelerationStructureDeviceAddressInfoKHR {
		sType                 = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		accelerationStructure = out.handle,
	}
	out.device_address = vk.GetAccelerationStructureDeviceAddressKHR(device.device, &address_info)

	return out
}

destroy_acceleration_structure :: proc(device: ^Device, as: ^Acceleration_Structure) {
	if as.handle != 0 {
		vk.DestroyAccelerationStructureKHR(device.device, as.handle, nil)
	}
	destroy_buffer(device, &as.buffer)
}

// sizes, allocates and records the build command for a single BLAS/TLAS.
build_acceleration_structure :: proc(
	device: ^Device,
	temp_pool: ^[dynamic]Buffer,
	cb: vk.CommandBuffer,
	build_info: vk.AccelerationStructureBuildGeometryInfoKHR,
	range_info: vk.AccelerationStructureBuildRangeInfoKHR,
) -> Acceleration_Structure {
	info := build_info

	max_prim_count := range_info.primitiveCount
	build_size := vk.AccelerationStructureBuildSizesInfoKHR {
		sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
	}
	vk.GetAccelerationStructureBuildSizesKHR(
		device.device,
		.DEVICE,
		&info,
		&max_prim_count,
		&build_size,
	)

	// scratchData.deviceAddress itself must be aligned to minAccelerationStructureScratchOffsetAlignment,
	// so over-allocate and round the address up within the buffer rather than just the size
	scratch_align := vk.DeviceSize(
		device.accel_properties.minAccelerationStructureScratchOffsetAlignment,
	)
	scratch_buffer := create_buffer(
		device,
		{
			size = build_size.buildScratchSize + scratch_align,
			memory = .GPU_ONLY,
			usage = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_STORAGE_KHR},
		},
	)
	append(temp_pool, scratch_buffer)

	out := create_acceleration_structure(device, build_size, info.type)

	info.dstAccelerationStructure = out.handle
	info.scratchData.deviceAddress = align_up(
		scratch_buffer.device_address,
		vk.DeviceAddress(scratch_align),
	)

	range := range_info
	range_ptr: [^]vk.AccelerationStructureBuildRangeInfoKHR = &range
	vk.CmdBuildAccelerationStructuresKHR(cb, 1, &info, &range_ptr)

	return out
}

write_tlas_descriptor :: proc(d: ^Device, tlas: ^Acceleration_Structure) {
	tlas_write := vk.WriteDescriptorSetAccelerationStructureKHR {
		sType                      = .WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
		accelerationStructureCount = 1,
		pAccelerationStructures    = &tlas.handle,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		pNext           = &tlas_write,
		dstSet          = d.rt_descriptor_set,
		dstBinding      = 0,
		dstArrayElement = 0,
		descriptorCount = 1,
		descriptorType  = .ACCELERATION_STRUCTURE_KHR,
	}
	vk.UpdateDescriptorSets(d.device, 1, &write, 0, nil)
}
