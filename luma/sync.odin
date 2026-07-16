package luma

import vk "vendor:vulkan"

Image_Barrier :: struct {
	image:      ^Image,
	src_stage:  vk.PipelineStageFlags2,
	src_access: vk.AccessFlags2,
	dst_stage:  vk.PipelineStageFlags2,
	dst_access: vk.AccessFlags2,
}

image_barriers :: proc(cb: vk.CommandBuffer, barriers: ..Image_Barrier) {
	// for image layouts, everything stays in GENERAL
	// swapchain is the only thing that ever leaves GENERAL (needs PRESENT_SRC_KHR)

	vk_barriers := make([]vk.ImageMemoryBarrier2, len(barriers), context.temp_allocator)
	for b, i in barriers {
		vk_barriers[i] = vk.ImageMemoryBarrier2 {
			sType = .IMAGE_MEMORY_BARRIER_2,
			srcStageMask = b.src_stage,
			srcAccessMask = b.src_access,
			dstStageMask = b.dst_stage,
			dstAccessMask = b.dst_access,
			oldLayout = b.image.layout,
			newLayout = .GENERAL,
			image = b.image.image,
			subresourceRange = {
				aspectMask = image_aspect_mask(b.image.format),
				layerCount = max(b.image.array_layers, 1),
				levelCount = b.image.mip_levels,
			},
		}
		b.image.layout = .GENERAL
	}
	dep := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = u32(len(vk_barriers)),
		pImageMemoryBarriers    = raw_data(vk_barriers),
	}
	vk.CmdPipelineBarrier2(cb, &dep)
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
