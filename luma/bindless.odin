package luma

import "core:fmt"
import vk "vendor:vulkan"

// one descriptor set for the whole renderer
//
// register a resource once, get back an index, and just pass that index around in push constants
MAX_BINDLESS_IMAGES :: 1000
MAX_SAMPLERS :: 20

BINDLESS_SAMPLER_BINDING :: 0
BINDLESS_TEXTURE_BINDING :: 1
BINDLESS_STORAGE_U32_BINDING :: 2
BINDLESS_STORAGE_RGBA8_BINDING :: 3

bindless_init :: proc(d: ^Device) {
	desc_pool_sizes := [?]vk.DescriptorPoolSize {
		{type = .SAMPLER, descriptorCount = MAX_SAMPLERS},
		{type = .SAMPLED_IMAGE, descriptorCount = MAX_BINDLESS_IMAGES},
		{type = .STORAGE_IMAGE, descriptorCount = MAX_BINDLESS_IMAGES * 2},
	}
	desc_pool_ci := vk.DescriptorPoolCreateInfo {
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		maxSets       = 1,
		poolSizeCount = len(desc_pool_sizes),
		pPoolSizes    = raw_data(&desc_pool_sizes),
		flags         = {.UPDATE_AFTER_BIND},
	}
	chk(vk.CreateDescriptorPool(d.device, &desc_pool_ci, nil, &d.descriptor_pool))

	common_binding_flags := vk.DescriptorBindingFlags{.UPDATE_AFTER_BIND, .PARTIALLY_BOUND}
	desc_binding_flags_arr := [?]vk.DescriptorBindingFlags {
		common_binding_flags,
		common_binding_flags,
		common_binding_flags,
		common_binding_flags | {.VARIABLE_DESCRIPTOR_COUNT},
	}
	desc_binding_flags := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
		sType         = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		bindingCount  = len(desc_binding_flags_arr),
		pBindingFlags = raw_data(&desc_binding_flags_arr),
	}
	desc_layout_bindings := [?]vk.DescriptorSetLayoutBinding {
		{
			binding = BINDLESS_SAMPLER_BINDING,
			descriptorType = .SAMPLER,
			descriptorCount = MAX_SAMPLERS,
			stageFlags = {.VERTEX, .FRAGMENT, .COMPUTE},
		},
		{
			binding = BINDLESS_TEXTURE_BINDING,
			descriptorType = .SAMPLED_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES,
			stageFlags = {.VERTEX, .FRAGMENT, .COMPUTE},
		},
		{
			binding = BINDLESS_STORAGE_U32_BINDING,
			descriptorType = .STORAGE_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES,
			stageFlags = {.VERTEX, .FRAGMENT, .COMPUTE},
		},
		{
			binding = BINDLESS_STORAGE_RGBA8_BINDING,
			descriptorType = .STORAGE_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES,
			stageFlags = {.VERTEX, .FRAGMENT, .COMPUTE},
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

	variable_desc_count: u32 = MAX_BINDLESS_IMAGES
	variable_desc_ai := vk.DescriptorSetVariableDescriptorCountAllocateInfo {
		sType              = .DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
		descriptorSetCount = 1,
		pDescriptorCounts  = &variable_desc_count,
	}
	descriptor_set_ai := vk.DescriptorSetAllocateInfo {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		pNext              = &variable_desc_ai,
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

// pair this with a sampler index and you get a usable texture in the shader
bindless_register_texture :: proc(d: ^Device, view: vk.ImageView) -> u32 {
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
	case .R32_UINT:
		binding = BINDLESS_STORAGE_U32_BINDING
		slot = d.bindless_next.storage_u32
		d.bindless_next.storage_u32 += 1
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
