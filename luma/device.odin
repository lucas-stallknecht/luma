package luma

import "core:fmt"
import "vendor:glfw"
import vk "vendor:vulkan"

chk :: proc(result: vk.Result, location := #caller_location) {
	if result != .SUCCESS {
		fmt.panicf("[Vulkan] Failure %s: %s", location, result)
	}
}

// ────────────────────────────────────────────────────────────────
// Vulkan Context

Device :: struct {
	instance:             vk.Instance,
	physical_device:      vk.PhysicalDevice,
	rt_properties:        vk.PhysicalDeviceRayTracingPipelinePropertiesKHR,
	accel_properties:     vk.PhysicalDeviceAccelerationStructurePropertiesKHR,
	queues:               struct {
		graphics:            vk.Queue,
		graphics_family_idx: u32,
		compute:             vk.Queue,
		compute_family_idx:  u32,
	},
	device:               vk.Device,
	memory_properties:    vk.PhysicalDeviceMemoryProperties,
	descriptor_pool:      vk.DescriptorPool,
	descriptor_layout:    vk.DescriptorSetLayout,
	descriptor_set:       vk.DescriptorSet,
	rt_descriptor_layout: vk.DescriptorSetLayout,
	rt_descriptor_set:    vk.DescriptorSet,
	command_handler:      Command_Handler,
	bindless_next:        struct {
		sampler:           u32,
		texture:           u32,
		texture_cube:      u32,
		storage_u32:       u32,
		storage_f32:       u32,
		storage_rgba8:     u32,
		storage_f32_array: u32,
	},
}

DEFAULT_PHYSICAL_DEVICE_SELECTION_FN :: proc(
	idx: int,
	properties: vk.PhysicalDeviceProperties2,
) -> bool {
	return true
}

Device_Desc :: struct {
	enable_validation:            bool,
	physical_device_selection_fn: Maybe(
		proc(idx: int, properties: vk.PhysicalDeviceProperties2) -> bool,
	),
}

device_init :: proc(d: ^Device, desc: Device_Desc) {
	{
		vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))
		glfw_extensions := glfw.GetRequiredInstanceExtensions()

		layers := [?]cstring{"VK_LAYER_KHRONOS_validation"}

		extensions := make([dynamic]cstring, context.temp_allocator)
		append(&extensions, ..glfw_extensions)

		instance_ci := vk.InstanceCreateInfo {
			sType            = .INSTANCE_CREATE_INFO,
			pApplicationInfo = &{sType = .APPLICATION_INFO, apiVersion = vk.API_VERSION_1_3},
		}
		if desc.enable_validation {
			enabled_features := [?]vk.ValidationFeatureEnableEXT {
				.DEBUG_PRINTF,
				.SYNCHRONIZATION_VALIDATION,
			}
			validation_features := vk.ValidationFeaturesEXT {
				sType                         = .VALIDATION_FEATURES_EXT,
				enabledValidationFeatureCount = len(enabled_features),
				pEnabledValidationFeatures    = raw_data(&enabled_features),
			}
			instance_ci.enabledLayerCount = u32(len(layers))
			instance_ci.ppEnabledLayerNames = raw_data(&layers)
			instance_ci.pNext = &validation_features
			append(&extensions, vk.EXT_VALIDATION_FEATURES_EXTENSION_NAME)
		} else {
			instance_ci.enabledLayerCount = 0
		}

		fmt.println("[Device] Instance extensions", extensions[:])
		instance_ci.enabledExtensionCount = u32(len(extensions))
		instance_ci.ppEnabledExtensionNames = raw_data(extensions)

		chk(vk.CreateInstance(&instance_ci, nil, &d.instance))
		vk.load_proc_addresses_instance(d.instance)
		if vk.EnumeratePhysicalDevices == nil do fmt.panicf("[Device] Failed to load instance functions")
	}

	{
		physical_device_count: u32
		chk(vk.EnumeratePhysicalDevices(d.instance, &physical_device_count, nil))
		if physical_device_count == 0 do fmt.panicf("[Device] No suitable physical devices found")
		physical_devices := make(
			[]vk.PhysicalDevice,
			physical_device_count,
			context.temp_allocator,
		)
		chk(
			vk.EnumeratePhysicalDevices(
				d.instance,
				&physical_device_count,
				raw_data(physical_devices),
			),
		)
		for pd, i in physical_devices {
			d.accel_properties = {
				sType = .PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
			}
			d.rt_properties = {
				sType = .PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
				pNext = &d.accel_properties,
			}
			device_properties := vk.PhysicalDeviceProperties2 {
				sType = .PHYSICAL_DEVICE_PROPERTIES_2,
				pNext = &d.rt_properties,
			}
			vk.GetPhysicalDeviceProperties2(pd, &device_properties)

			selection_fn :=
				desc.physical_device_selection_fn.? or_else DEFAULT_PHYSICAL_DEVICE_SELECTION_FN
			if !selection_fn(i, device_properties) do continue

			d.physical_device = pd
			fmt.printfln("[Device] Selected device: %s", device_properties.properties.deviceName)
			break
		}
		if d.physical_device == nil do fmt.panicf("[Device] No suitable physical device selected")
		vk.GetPhysicalDeviceMemoryProperties(d.physical_device, &d.memory_properties)
	}

	{
		queue_family_count: u32
		vk.GetPhysicalDeviceQueueFamilyProperties(d.physical_device, &queue_family_count, nil)
		queue_families := make(
			[]vk.QueueFamilyProperties,
			queue_family_count,
			context.temp_allocator,
		)
		vk.GetPhysicalDeviceQueueFamilyProperties(
			d.physical_device,
			&queue_family_count,
			raw_data(queue_families),
		)

		for i in 0 ..< queue_family_count {
			if vk.QueueFlag.GRAPHICS in queue_families[i].queueFlags {
				d.queues.graphics_family_idx = i
			}
			if vk.QueueFlag.COMPUTE in queue_families[i].queueFlags {
				d.queues.compute_family_idx = i
			}
		}
	}

	{
		queue_priority: f32 = 1.0
		queue_ci := [?]vk.DeviceQueueCreateInfo {
			{
				sType = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = d.queues.graphics_family_idx,
				queueCount = 1,
				pQueuePriorities = &queue_priority,
			},
			{
				sType = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = d.queues.compute_family_idx,
				queueCount = 1,
				pQueuePriorities = &queue_priority,
			},
		}
		num_queues := 1 if d.queues.graphics_family_idx == d.queues.compute_family_idx else 2

		device_extensions := [?]cstring {
			vk.KHR_SWAPCHAIN_EXTENSION_NAME,
			vk.EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
			vk.KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
			vk.KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
			vk.KHR_RAY_QUERY_EXTENSION_NAME,
			vk.EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_EXTENSION_NAME,
		}
		vk11_features := vk.PhysicalDeviceVulkan11Features {
			sType                = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
			shaderDrawParameters = true,
		}
		vk12_features := vk.PhysicalDeviceVulkan12Features {
			sType                                        = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			pNext                                        = &vk11_features,
			bufferDeviceAddress                          = true,
			timelineSemaphore                            = true,
			descriptorIndexing                           = true,
			runtimeDescriptorArray                       = true,
			shaderSampledImageArrayNonUniformIndexing    = true,
			shaderStorageImageArrayNonUniformIndexing    = true,
			descriptorBindingPartiallyBound              = true,
			descriptorBindingSampledImageUpdateAfterBind = true,
			descriptorBindingStorageImageUpdateAfterBind = true,
			scalarBlockLayout                            = true,
		}
		unused_attachments_features :=
			vk.PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT {
				sType                             = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT,
				pNext                             = &vk12_features,
				dynamicRenderingUnusedAttachments = true,
			}
		vk13_features := vk.PhysicalDeviceVulkan13Features {
				sType                          = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
				pNext                          = &unused_attachments_features,
				synchronization2               = true,
				dynamicRendering               = true,
				shaderDemoteToHelperInvocation = true,
			}
		vk10_features := vk.PhysicalDeviceFeatures {
				samplerAnisotropy        = true,
				geometryShader           = true,
				multiDrawIndirect        = true,
				fragmentStoresAndAtomics = true,
			}
		accel_features := vk.PhysicalDeviceAccelerationStructureFeaturesKHR {
				sType                 = .PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
				pNext                 = &vk13_features,
				accelerationStructure = true,
			}
		ray_query_features := vk.PhysicalDeviceRayQueryFeaturesKHR {
				sType    = .PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
				pNext    = &accel_features,
				rayQuery = true,
			}
		rt_pipeline_features := vk.PhysicalDeviceRayTracingPipelineFeaturesKHR {
				sType = .PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
				pNext = &ray_query_features,
			}

		device_ci := vk.DeviceCreateInfo {
				sType                   = .DEVICE_CREATE_INFO,
				pNext                   = &rt_pipeline_features,
				queueCreateInfoCount    = u32(num_queues),
				pQueueCreateInfos       = raw_data(&queue_ci),
				enabledExtensionCount   = u32(len(device_extensions)),
				ppEnabledExtensionNames = raw_data(&device_extensions),
				pEnabledFeatures        = &vk10_features,
			}
		chk(vk.CreateDevice(d.physical_device, &device_ci, nil, &d.device))
		vk.load_proc_addresses_device(d.device)
		if vk.BeginCommandBuffer == nil do fmt.panicf("[Device] Failed to load device functions")
		vk.GetDeviceQueue(d.device, d.queues.compute_family_idx, 0, &d.queues.compute)
		vk.GetDeviceQueue(d.device, d.queues.graphics_family_idx, 0, &d.queues.graphics)
	}

	command_handler_init(
		&d.command_handler,
		d.device,
		d.queues.graphics,
		d.queues.graphics_family_idx,
	)

	desc_pool_sizes := [?]vk.DescriptorPoolSize {
		{type = .SAMPLER, descriptorCount = MAX_SAMPLERS},
		{type = .SAMPLED_IMAGE, descriptorCount = MAX_BINDLESS_IMAGES + MAX_CUBE_TEXTURES},
		{
			type = .STORAGE_IMAGE,
			descriptorCount = MAX_BINDLESS_IMAGES * 3 + MAX_STORAGE_ARRAY_IMAGES,
		},
		{type = .ACCELERATION_STRUCTURE_KHR, descriptorCount = 1},
	}
	desc_pool_ci := vk.DescriptorPoolCreateInfo {
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		maxSets       = 2,
		poolSizeCount = len(desc_pool_sizes),
		pPoolSizes    = raw_data(&desc_pool_sizes),
		flags         = {.UPDATE_AFTER_BIND},
	}
	chk(vk.CreateDescriptorPool(d.device, &desc_pool_ci, nil, &d.descriptor_pool))

	bindless_init(d)

	// TLAS lives in its own set (1), separate from the bindless one (0)
	rt_desc_layout_bindings := [?]vk.DescriptorSetLayoutBinding {
		{
			binding = 0,
			descriptorType = .ACCELERATION_STRUCTURE_KHR,
			descriptorCount = 1,
			stageFlags = {.COMPUTE},
		},
	}
	rt_desc_layout_ci := vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		flags        = {.UPDATE_AFTER_BIND_POOL},
		bindingCount = len(rt_desc_layout_bindings),
		pBindings    = raw_data(&rt_desc_layout_bindings),
	}
	chk(vk.CreateDescriptorSetLayout(d.device, &rt_desc_layout_ci, nil, &d.rt_descriptor_layout))
	rt_descriptor_set_ai := vk.DescriptorSetAllocateInfo {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool     = d.descriptor_pool,
		descriptorSetCount = 1,
		pSetLayouts        = &d.rt_descriptor_layout,
	}
	chk(vk.AllocateDescriptorSets(d.device, &rt_descriptor_set_ai, &d.rt_descriptor_set))

	free_all(context.temp_allocator)
}

device_cleanup :: proc(d: ^Device) {
	vk.DestroyDescriptorSetLayout(d.device, d.rt_descriptor_layout, nil)
	command_handler_cleanup(&d.command_handler)
	bindless_cleanup(d)
	vk.DestroyDevice(d.device, nil)
	vk.DestroyInstance(d.instance, nil)
}

// ────────────────────────────────────────────────────────────────
// Memory

Memory_Preset :: enum {
	GPU_ONLY,
	CPU_UPLOAD,
}

get_memory_flags :: proc(p: Memory_Preset) -> vk.MemoryPropertyFlags {
	switch p {
	case .GPU_ONLY:
		return {.DEVICE_LOCAL}
	case .CPU_UPLOAD:
		return {.HOST_VISIBLE, .HOST_COHERENT}
	}
	return {.DEVICE_LOCAL}
}

find_memory_type :: proc(
	device: ^Device,
	type_filter: u32,
	properties: vk.MemoryPropertyFlags,
) -> u32 {
	mem_properties := device.memory_properties
	for i in 0 ..< mem_properties.memoryTypeCount {
		if (type_filter & (1 << i) != 0) &&
		   (properties & mem_properties.memoryTypes[i].propertyFlags) == properties {
			return i
		}
	}
	fmt.panicf("[Device] Failed to find suitable memory type")
}
