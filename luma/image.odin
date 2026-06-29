package luma

import vk "vendor:vulkan"

Image :: struct {
	image:          vk.Image,
	view:           vk.ImageView,
	memory:         vk.DeviceMemory,
	format:         vk.Format,
	layout:         vk.ImageLayout, // for image_barriers()
	bindless_index: u32,
}

Image_Create_Desc :: struct {
	width:             u32,
	height:            u32,
	format:            vk.Format,
	usage:             vk.ImageUsageFlags,
	memory:            Memory_Preset,
	// registers into the bindless set on creation, fills Image.bindless_index
	register_bindless: bool,
}

create_image :: proc(device: ^Device, desc: Image_Create_Desc) -> Image {
	out: Image
	out.format = desc.format
	out.layout = .UNDEFINED

	image_ci := vk.ImageCreateInfo {
		sType       = .IMAGE_CREATE_INFO,
		imageType   = .D2,
		extent      = {desc.width, desc.height, 1},
		mipLevels   = 1,
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
			levelCount = 1,
		},
	}
	chk(vk.CreateImageView(device.device, &view_ci, nil, &out.view))

	if desc.register_bindless {
		out.bindless_index = bindless_register_storage_image(device, out.view, out.format)
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
				levelCount = 1,
				layerCount = 1,
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
