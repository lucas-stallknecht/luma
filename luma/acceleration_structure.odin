package luma

import vk "vendor:vulkan"

Acceleration_Structure :: struct {
	handle:         vk.AccelerationStructureKHR,
	buffer:         Buffer,
	device_address: vk.DeviceAddress,
}

create_acceleration_structure :: proc(
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
		type   = info.type,
	}
	chk(vk.CreateAccelerationStructureKHR(device.device, &create_info, nil, &out.handle))

	address_info := vk.AccelerationStructureDeviceAddressInfoKHR {
		sType                 = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		accelerationStructure = out.handle,
	}
	out.device_address = vk.GetAccelerationStructureDeviceAddressKHR(device.device, &address_info)

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

destroy_acceleration_structure :: proc(device: ^Device, as: ^Acceleration_Structure) {
	if as.handle != 0 {
		vk.DestroyAccelerationStructureKHR(device.device, as.handle, nil)
	}
	destroy_buffer(device, &as.buffer)
}
