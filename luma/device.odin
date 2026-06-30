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
	instance:                vk.Instance,
	physical_device:         vk.PhysicalDevice,
	queues:                  struct {
		graphics:            vk.Queue,
		graphics_family_idx: u32,
		compute:             vk.Queue,
		compute_family_idx:  u32,
	},
	device:                  vk.Device,
	available_depth_formats: [dynamic]vk.Format,
	descriptor_pool:         vk.DescriptorPool,
	descriptor_layout:       vk.DescriptorSetLayout,
	descriptor_set:          vk.DescriptorSet,
	command_handler:         Command_Handler,
	bindless_next:           struct {
		sampler:       u32,
		texture:       u32,
		storage_u32:   u32,
		storage_hdr:   u32,
		storage_rgba8: u32,
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
	// instance
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
			instance_ci.enabledLayerCount = u32(len(layers))
			instance_ci.ppEnabledLayerNames = raw_data(&layers)
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

	// physical device
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
			device_properties := vk.PhysicalDeviceProperties2 {
				sType = .PHYSICAL_DEVICE_PROPERTIES_2,
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
	}

	// queues
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

	// logical device
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

		device_extensions := [?]cstring{"VK_KHR_swapchain", "VK_EXT_extended_dynamic_state3"}
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
			descriptorBindingVariableDescriptorCount     = true,
			runtimeDescriptorArray                       = true,
			shaderSampledImageArrayNonUniformIndexing    = true,
			shaderStorageImageArrayNonUniformIndexing    = true,
			shaderUniformBufferArrayNonUniformIndexing   = true,
			descriptorBindingPartiallyBound              = true,
			descriptorBindingUpdateUnusedWhilePending    = true,
			descriptorBindingSampledImageUpdateAfterBind = true,
			descriptorBindingStorageImageUpdateAfterBind = true,
		}
		vk13_features := vk.PhysicalDeviceVulkan13Features {
			sType            = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
			pNext            = &vk12_features,
			synchronization2 = true,
			dynamicRendering = true,
		}
		vk10_features := vk.PhysicalDeviceFeatures {
			samplerAnisotropy                    = true,
			geometryShader                       = true,
			multiDrawIndirect                    = true,
			shaderStorageImageReadWithoutFormat  = true,
			shaderStorageImageWriteWithoutFormat = true,
			fragmentStoresAndAtomics             = true,
		}

		device_ci := vk.DeviceCreateInfo {
			sType                   = .DEVICE_CREATE_INFO,
			pNext                   = &vk13_features,
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
	bindless_init(d)
	free_all(context.temp_allocator)
}

device_cleanup :: proc(d: ^Device) {
	command_handler_cleanup(&d.command_handler)
	bindless_cleanup(d)
	delete(d.available_depth_formats)
	vk.DestroyDevice(d.device, nil)
	vk.DestroyInstance(d.instance, nil)
}

// ────────────────────────────────────────────────────────────────
// Memory

Memory_Preset :: enum {
	GPU_ONLY,
	CPU_UPLOAD,
	CPU_READBACK,
}

get_memory_flags :: proc(p: Memory_Preset) -> vk.MemoryPropertyFlags {
	switch p {
	case .GPU_ONLY:
		return {.DEVICE_LOCAL}

	case .CPU_UPLOAD:
		return {.HOST_VISIBLE, .HOST_COHERENT}

	case .CPU_READBACK:
		return {.HOST_VISIBLE, .HOST_COHERENT}
	}
	return {.DEVICE_LOCAL}
}

find_memory_type :: proc(
	device: ^Device,
	type_filter: u32,
	properties: vk.MemoryPropertyFlags,
) -> u32 {
	mem_properties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(device.physical_device, &mem_properties)

	for i in 0 ..< mem_properties.memoryTypeCount {
		if (type_filter & (1 << i) != 0) &&
		   (properties & mem_properties.memoryTypes[i].propertyFlags) == properties { 	// memory type is suitable
			return i
		}
	}
	fmt.panicf("[Device] Failed to find suitable memory type")
}
