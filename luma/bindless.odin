package luma

import "core:fmt"
import vk "vendor:vulkan"


// one descriptor set for the whole renderer: register a resource once, get an
// index back, and pass that index around in push constants instead of binding sets


MAX_BINDLESS_IMAGES :: 100
MAX_SAMPLERS :: 5
MAX_CUBE_TEXTURES :: 1
MAX_STORAGE_ARRAY_IMAGES :: 1

BINDLESS_SAMPLER_BINDING :: 0
BINDLESS_TEXTURE_BINDING :: 1
BINDLESS_STORAGE_U32_BINDING :: 2
BINDLESS_STORAGE_F32_BINDING :: 3
BINDLESS_STORAGE_RGBA8_BINDING :: 4
BINDLESS_TEXTURE_CUBE_BINDING :: 5
BINDLESS_STORAGE_F32_ARRAY_BINDING :: 6

bindless_init :: proc(d: ^Device) {
	common_binding_flags := vk.DescriptorBindingFlags{.UPDATE_AFTER_BIND, .PARTIALLY_BOUND}
	desc_binding_flags_arr := [?]vk.DescriptorBindingFlags {
		common_binding_flags,
		common_binding_flags,
		common_binding_flags,
		common_binding_flags,
		common_binding_flags,
		common_binding_flags,
		common_binding_flags,
	}
	desc_binding_flags := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
		sType         = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		bindingCount  = len(desc_binding_flags_arr),
		pBindingFlags = raw_data(&desc_binding_flags_arr),
	}
	stage_flags: vk.ShaderStageFlags = {
		.VERTEX,
		.FRAGMENT,
		.COMPUTE,
		.RAYGEN_KHR,
		.MISS_KHR,
		.ANY_HIT_KHR,
	}
	desc_layout_bindings := [?]vk.DescriptorSetLayoutBinding {
		{
			binding = BINDLESS_SAMPLER_BINDING,
			descriptorType = .SAMPLER,
			descriptorCount = MAX_SAMPLERS,
			stageFlags = stage_flags,
		},
		{
			binding = BINDLESS_TEXTURE_BINDING,
			descriptorType = .SAMPLED_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES,
			stageFlags = stage_flags,
		},
		{
			binding = BINDLESS_STORAGE_U32_BINDING,
			descriptorType = .STORAGE_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES,
			stageFlags = stage_flags,
		},
		{
			binding = BINDLESS_STORAGE_F32_BINDING,
			descriptorType = .STORAGE_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES,
			stageFlags = stage_flags,
		},
		{
			binding = BINDLESS_STORAGE_RGBA8_BINDING,
			descriptorType = .STORAGE_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES,
			stageFlags = stage_flags,
		},
		{
			binding = BINDLESS_TEXTURE_CUBE_BINDING,
			descriptorType = .SAMPLED_IMAGE,
			descriptorCount = MAX_CUBE_TEXTURES,
			stageFlags = stage_flags,
		},
		{
			binding = BINDLESS_STORAGE_F32_ARRAY_BINDING,
			descriptorType = .STORAGE_IMAGE,
			descriptorCount = MAX_STORAGE_ARRAY_IMAGES,
			stageFlags = stage_flags,
		},
	}
	desc_layout_ci := vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		pNext        = &desc_binding_flags,
		flags        = {.UPDATE_AFTER_BIND_POOL},
		bindingCount = len(desc_layout_bindings),
		pBindings    = raw_data(&desc_layout_bindings),
	}
	chk(vk.CreateDescriptorSetLayout(d.device, &desc_layout_ci, nil, &d.descriptor_layout))

	descriptor_set_ai := vk.DescriptorSetAllocateInfo {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool     = d.descriptor_pool,
		descriptorSetCount = 1,
		pSetLayouts        = &d.descriptor_layout,
	}
	chk(vk.AllocateDescriptorSets(d.device, &descriptor_set_ai, &d.descriptor_set))
}

bindless_cleanup :: proc(d: ^Device) {
	vk.DestroyDescriptorSetLayout(d.device, d.descriptor_layout, nil)
	vk.DestroyDescriptorPool(d.device, d.descriptor_pool, nil)
}

// figures out which array the format belongs to and writes the descriptor there
bindless_register_storage_image :: proc(d: ^Device, view: vk.ImageView, format: vk.Format) -> u32 {
	binding, slot := storage_image_binding_and_slot(d, format)
	info := vk.DescriptorImageInfo {
		imageView   = view,
		imageLayout = .GENERAL,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = d.descriptor_set,
		dstBinding      = binding,
		dstArrayElement = slot,
		descriptorCount = 1,
		descriptorType  = .STORAGE_IMAGE,
		pImageInfo      = &info,
	}
	vk.UpdateDescriptorSets(d.device, 1, &write, 0, nil)
	return slot
}

bindless_register_texture :: proc(d: ^Device, view: vk.ImageView) -> u32 {
	// pair this with a sampler index and you get a usable texture in the shader

	slot := d.bindless_next.texture
	d.bindless_next.texture += 1
	info := vk.DescriptorImageInfo {
		imageView   = view,
		imageLayout = .GENERAL,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = d.descriptor_set,
		dstBinding      = BINDLESS_TEXTURE_BINDING,
		dstArrayElement = slot,
		descriptorCount = 1,
		descriptorType  = .SAMPLED_IMAGE,
		pImageInfo      = &info,
	}
	vk.UpdateDescriptorSets(d.device, 1, &write, 0, nil)
	return slot
}

bindless_register_texture_cube :: proc(d: ^Device, view: vk.ImageView) -> u32 {
	slot := d.bindless_next.texture_cube
	d.bindless_next.texture_cube += 1
	info := vk.DescriptorImageInfo {
		imageView   = view,
		imageLayout = .GENERAL,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = d.descriptor_set,
		dstBinding      = BINDLESS_TEXTURE_CUBE_BINDING,
		dstArrayElement = slot,
		descriptorCount = 1,
		descriptorType  = .SAMPLED_IMAGE,
		pImageInfo      = &info,
	}
	vk.UpdateDescriptorSets(d.device, 1, &write, 0, nil)
	return slot
}

bindless_register_storage_image_array :: proc(d: ^Device, view: vk.ImageView) -> u32 {
	// view must be a D2_ARRAY covering every layer, unlike the per-image slots above

	slot := d.bindless_next.storage_f32_array
	d.bindless_next.storage_f32_array += 1
	info := vk.DescriptorImageInfo {
		imageView   = view,
		imageLayout = .GENERAL,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = d.descriptor_set,
		dstBinding      = BINDLESS_STORAGE_F32_ARRAY_BINDING,
		dstArrayElement = slot,
		descriptorCount = 1,
		descriptorType  = .STORAGE_IMAGE,
		pImageInfo      = &info,
	}
	vk.UpdateDescriptorSets(d.device, 1, &write, 0, nil)
	return slot
}

bindless_register_sampler :: proc(d: ^Device, sampler: vk.Sampler) -> u32 {
	slot := d.bindless_next.sampler
	d.bindless_next.sampler += 1
	info := vk.DescriptorImageInfo {
		sampler = sampler,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = d.descriptor_set,
		dstBinding      = BINDLESS_SAMPLER_BINDING,
		dstArrayElement = slot,
		descriptorCount = 1,
		descriptorType  = .SAMPLER,
		pImageInfo      = &info,
	}
	vk.UpdateDescriptorSets(d.device, 1, &write, 0, nil)
	return slot
}

@(private = "file")
storage_image_binding_and_slot :: proc(
	d: ^Device,
	format: vk.Format,
) -> (
	binding: u32,
	slot: u32,
) {
	#partial switch format {
	case .R32G32_UINT:
		binding = BINDLESS_STORAGE_U32_BINDING
		slot = d.bindless_next.storage_u32
		d.bindless_next.storage_u32 += 1
	case .R32G32B32A32_SFLOAT:
		binding = BINDLESS_STORAGE_F32_BINDING
		slot = d.bindless_next.storage_f32
		d.bindless_next.storage_f32 += 1
	case .R8G8B8A8_UNORM:
		binding = BINDLESS_STORAGE_RGBA8_BINDING
		slot = d.bindless_next.storage_rgba8
		d.bindless_next.storage_rgba8 += 1
	case:
		fmt.panicf(
			"[Device] No bindless storage array for format %v - add one in luma.glsl and a case here",
			format,
		)
	}
	return
}
