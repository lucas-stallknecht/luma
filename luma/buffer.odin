package luma

import "base:intrinsics"
import vk "vendor:vulkan"

Buffer :: struct {
	buffer:         vk.Buffer,
	memory:         vk.DeviceMemory,
	device_address: vk.DeviceAddress,
}

// rounds value up to the next multiple of alignment (alignment must be a power of two)
align_up :: proc(value: $T, alignment: T) -> T {
	return (value + alignment - 1) & ~(alignment - 1)
}

Buffer_Create_Desc :: struct {
	size:   vk.DeviceSize,
	usage:  vk.BufferUsageFlags,
	memory: Memory_Preset,
}

create_buffer :: proc(device: ^Device, desc: Buffer_Create_Desc) -> Buffer {
	out: Buffer

	buffer_ci := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = desc.size,
		usage       = desc.usage,
		sharingMode = .EXCLUSIVE,
	}
	chk(vk.CreateBuffer(device.device, &buffer_ci, nil, &out.buffer))

	mem_requirements: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(device.device, out.buffer, &mem_requirements)

	alloc_info := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = mem_requirements.size,
		memoryTypeIndex = find_memory_type(
			device,
			mem_requirements.memoryTypeBits,
			get_memory_flags(desc.memory),
		),
	}
	if .SHADER_DEVICE_ADDRESS in desc.usage {
		alloc_info.pNext = &vk.MemoryAllocateFlagsInfo {
			sType = .MEMORY_ALLOCATE_FLAGS_INFO,
			flags = {.DEVICE_ADDRESS},
		}
	}

	chk(vk.AllocateMemory(device.device, &alloc_info, nil, &out.memory))
	vk.BindBufferMemory(device.device, out.buffer, out.memory, 0)

	if .SHADER_DEVICE_ADDRESS in desc.usage {
		device_address_info := vk.BufferDeviceAddressInfo {
			sType  = .BUFFER_DEVICE_ADDRESS_INFO,
			buffer = out.buffer,
		}
		out.device_address = vk.GetBufferDeviceAddress(device.device, &device_address_info)
	}

	return out
}

destroy_buffer :: proc(device: ^Device, buffer: ^Buffer) {
	if buffer.buffer != 0 {
		vk.DestroyBuffer(device.device, buffer.buffer, nil)
	}
	if buffer.memory != 0 {
		vk.FreeMemory(device.device, buffer.memory, nil)
	}
}

create_and_upload_buffer :: proc(
	device: ^Device,
	temp_pool: ^[dynamic]Buffer,
	cb: vk.CommandBuffer,
	data: rawptr,
	desc: Buffer_Create_Desc,
) -> Buffer {
	desc_copy := desc
	desc_copy.usage += {.TRANSFER_DST}
	out := create_buffer(device, desc_copy)

	staging := create_buffer(
		device,
		{size = desc.size, usage = {.TRANSFER_SRC}, memory = .CPU_UPLOAD},
	)
	// keep staging alive by pushing it into the caller-provided temp pool
	append(temp_pool, staging)

	mapped: rawptr
	vk.MapMemory(device.device, staging.memory, 0, desc.size, {}, &mapped)
	intrinsics.mem_copy(mapped, data, desc.size)
	vk.UnmapMemory(device.device, staging.memory)

	copy_info := vk.BufferCopy {
		size      = desc.size,
		srcOffset = 0,
		dstOffset = 0,
	}
	vk.CmdCopyBuffer(cb, staging.buffer, out.buffer, 1, &copy_info)

	return out
}

Buffer_Barrier :: struct {
	buffer:     ^Buffer,
	src_stage:  vk.PipelineStageFlags2,
	src_access: vk.AccessFlags2,
	dst_stage:  vk.PipelineStageFlags2,
	dst_access: vk.AccessFlags2,
}

buffer_barriers :: proc(cb: vk.CommandBuffer, barriers: ..Buffer_Barrier) {
	vk_barriers := make([]vk.BufferMemoryBarrier2, len(barriers), context.temp_allocator)
	for b, i in barriers {
		vk_barriers[i] = vk.BufferMemoryBarrier2 {
			sType               = .BUFFER_MEMORY_BARRIER_2,
			srcStageMask        = b.src_stage,
			srcAccessMask       = b.src_access,
			dstStageMask        = b.dst_stage,
			dstAccessMask       = b.dst_access,
			srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			buffer              = b.buffer.buffer,
			offset              = 0,
			size                = vk.DeviceSize(vk.WHOLE_SIZE),
		}
	}
	dep := vk.DependencyInfo {
		sType                    = .DEPENDENCY_INFO,
		bufferMemoryBarrierCount = u32(len(vk_barriers)),
		pBufferMemoryBarriers    = raw_data(vk_barriers),
	}
	vk.CmdPipelineBarrier2(cb, &dep)
}
