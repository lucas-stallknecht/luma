package luma

import "base:intrinsics"
import "core:math"
import vk "vendor:vulkan"

Image :: struct {
	image:        vk.Image,
	view:         vk.ImageView,
	memory:       vk.DeviceMemory,
	format:       vk.Format,
	layout:       vk.ImageLayout, // for image_barriers()
	mip_levels:   u32,
	bindless_idx: u32,
}

Image_Create_Desc :: struct {
	width:             u32,
	height:            u32,
	format:            vk.Format,
	usage:             vk.ImageUsageFlags,
	memory:            Memory_Preset,
	// registers into the bindless set on creation, fills Image.bindless_index
	register_bindless: enum {
		None,
		Storage,
		Texture,
	},
	mips:              bool,
}

create_image :: proc(device: ^Device, desc: Image_Create_Desc) -> Image {
	out: Image
	out.format = desc.format
	out.layout = .UNDEFINED

	mip_levels :=
		1 if !desc.mips else u32(math.floor(math.log2(max(f32(desc.width), f32(desc.height))))) + 1

	image_ci := vk.ImageCreateInfo {
		sType       = .IMAGE_CREATE_INFO,
		imageType   = .D2,
		extent      = {desc.width, desc.height, 1},
		mipLevels   = mip_levels,
		arrayLayers = 1,
		usage       = desc.usage,
		format      = desc.format,
		tiling      = .OPTIMAL,
		sharingMode = .EXCLUSIVE,
		samples     = {._1},
	}
	chk(vk.CreateImage(device.device, &image_ci, nil, &out.image))

	mem_requirements: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(device.device, out.image, &mem_requirements)

	alloc_info := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = mem_requirements.size,
		memoryTypeIndex = find_memory_type(
			device,
			mem_requirements.memoryTypeBits,
			get_memory_flags(desc.memory),
		),
	}
	chk(vk.AllocateMemory(device.device, &alloc_info, nil, &out.memory))
	vk.BindImageMemory(device.device, out.image, out.memory, 0)

	view_ci := vk.ImageViewCreateInfo {
		sType = .IMAGE_VIEW_CREATE_INFO,
		image = out.image,
		viewType = .D2,
		format = desc.format,
		subresourceRange = {
			aspectMask = image_aspect_mask(desc.format),
			layerCount = 1,
			levelCount = mip_levels,
		},
	}
	chk(vk.CreateImageView(device.device, &view_ci, nil, &out.view))

	#partial switch desc.register_bindless {
	case .Storage:
		out.bindless_idx = bindless_register_storage_image(device, out.view, out.format)
	case .Texture:
		out.bindless_idx = bindless_register_texture(device, out.view)
	}
	out.mip_levels = mip_levels

	return out
}

create_and_upload_image :: proc(
	device: ^Device,
	temp_pool: ^[dynamic]Buffer,
	cb: vk.CommandBuffer,
	data: rawptr,
	data_size: vk.DeviceSize,
	desc: Image_Create_Desc,
) -> Image {
	desc_copy := desc
	desc_copy.usage += {.TRANSFER_DST}
	out := create_image(device, desc_copy)

	staging := create_buffer(
		device,
		{size = data_size, usage = {.TRANSFER_SRC}, memory = .CPU_UPLOAD},
	)
	append(temp_pool, staging)

	mapped: rawptr
	vk.MapMemory(device.device, staging.memory, 0, data_size, {}, &mapped)
	intrinsics.mem_copy(mapped, data, int(data_size))
	vk.UnmapMemory(device.device, staging.memory)

	image_barriers(cb, {image = &out, dst_stage = {.TRANSFER}, dst_access = {.TRANSFER_WRITE}})

	copy_region := vk.BufferImageCopy {
		imageSubresource = {aspectMask = {.COLOR}, layerCount = 1},
		imageExtent = {desc.width, desc.height, 1},
	}
	vk.CmdCopyBufferToImage(cb, staging.buffer, out.image, .GENERAL, 1, &copy_region)

	// generate mips
	if desc.mips {
		barrier := vk.ImageMemoryBarrier2 {
			sType = .IMAGE_MEMORY_BARRIER_2,
			srcStageMask = {.TRANSFER},
			srcAccessMask = {.TRANSFER_WRITE},
			dstStageMask = {.TRANSFER},
			dstAccessMask = {.TRANSFER_READ},
			oldLayout = out.layout,
			newLayout = .GENERAL,
			image = out.image,
			subresourceRange = {
				aspectMask = image_aspect_mask(out.format),
				levelCount = 1,
				layerCount = 1,
			},
		}
		dep := vk.DependencyInfo {
			sType                   = .DEPENDENCY_INFO,
			imageMemoryBarrierCount = 1,
			pImageMemoryBarriers    = &barrier,
		}
		vk.CmdPipelineBarrier2(cb, &dep)

		mip_width := i32(desc.width)
		mip_height := i32(desc.height)

		for i in 1 ..< out.mip_levels {
			barrier.subresourceRange.baseMipLevel = i - 1
			vk.CmdPipelineBarrier2(cb, &dep)

			blit := vk.ImageBlit {
				srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
				dstOffsets = {
					{0, 0, 0},
					{
						mip_width / 2 if mip_width > 1 else 1,
						mip_height / 2 if mip_height > 1 else 1,
						1,
					},
				},
				srcSubresource = {
					aspectMask = image_aspect_mask(out.format),
					mipLevel = i - 1,
					layerCount = 1,
				},
				dstSubresource = {
					aspectMask = image_aspect_mask(out.format),
					mipLevel = i,
					layerCount = 1,
				},
			}
			vk.CmdBlitImage(cb, out.image, .GENERAL, out.image, .GENERAL, 1, &blit, .LINEAR)

			if mip_width > 1 do mip_width /= 2
			if mip_height > 1 do mip_height /= 2
		}
	}

	return out
}

destroy_image :: proc(device: ^Device, image: Image) {
	if image.view != 0 {
		vk.DestroyImageView(device.device, image.view, nil)
	}
	if image.image != 0 {
		vk.DestroyImage(device.device, image.image, nil)
	}
	if image.memory != 0 {
		vk.FreeMemory(device.device, image.memory, nil)
	}
}

// only depth format we use right now, add more if needed
image_aspect_mask :: proc(format: vk.Format) -> vk.ImageAspectFlags {
	#partial switch format {
	case .D32_SFLOAT, .D16_UNORM:
		return {.DEPTH}
	case:
		return {.COLOR}
	}
}

// everything stays in GENERAL forever, so all you describe here is the
// sync, not the layout dance. swapchain is the only thing that ever leaves
// GENERAL (needs PRESENT_SRC_KHR), see swapchain_barrier_to_present
Image_Barrier :: struct {
	image:      ^Image,
	src_stage:  vk.PipelineStageFlags2,
	src_access: vk.AccessFlags2,
	dst_stage:  vk.PipelineStageFlags2,
	dst_access: vk.AccessFlags2,
}

image_barriers :: proc(cb: vk.CommandBuffer, barriers: ..Image_Barrier) {
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
				layerCount = 1,
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
