package luma

import "base:intrinsics"
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
		storage_u32:   u32,
		storage_rgba8: u32,
	},
}

MAX_BINDLESS_IMAGES :: 1000
MAX_SAMPLERS :: 20

// glsl requires a format layout qualifier on a storage image that's read
// (not just written)... SO each concrete format gets its own bindless array
// see shaders/luma.glsl for the matching glsl side
BINDLESS_SAMPLER_BINDING :: 0
BINDLESS_STORAGE_U32_BINDING :: 1
BINDLESS_STORAGE_RGBA8_BINDING :: 2
DEFAULT_PHYSICAL_DEVICE_SELECTION_FN :: proc(
	idx: int,
	properties: vk.PhysicalDeviceProperties2,
) -> bool {
	return true
}


Device_Desc :: struct {
	enable_validation:            bool,
	shared_types_file_path:       string,
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

	// physical Device
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

	// logical Device
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
	free_all(context.temp_allocator)


	// bindless Descriptors
	{
		desc_pool_sizes := [?]vk.DescriptorPoolSize {
			{type = .SAMPLER, descriptorCount = MAX_SAMPLERS},
			{type = .STORAGE_IMAGE, descriptorCount = MAX_BINDLESS_IMAGES * 2},
		}
		desc_pool_ci := vk.DescriptorPoolCreateInfo {
			sType         = .DESCRIPTOR_POOL_CREATE_INFO,
			maxSets       = 3,
			poolSizeCount = len(desc_pool_sizes),
			pPoolSizes    = raw_data(&desc_pool_sizes),
			flags         = {.UPDATE_AFTER_BIND},
		}
		chk(vk.CreateDescriptorPool(d.device, &desc_pool_ci, nil, &d.descriptor_pool))

		common_binding_flags := vk.DescriptorBindingFlags{.UPDATE_AFTER_BIND, .PARTIALLY_BOUND}
		desc_binding_flags_arr := [?]vk.DescriptorBindingFlags {
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

}

device_cleanup :: proc(d: ^Device) {
	command_handler_cleanup(&d.command_handler)

	vk.DestroyDescriptorSetLayout(d.device, d.descriptor_layout, nil)
	vk.DestroyDescriptorPool(d.device, d.descriptor_pool, nil)

	delete(d.available_depth_formats)
	vk.DestroyDevice(d.device, nil)
	vk.DestroyInstance(d.instance, nil)
}

// ────────────────────────────────────────────────────────────────
// Objects creation

DEFAULT_PREFERRED_SURFACE_FORMAT :: vk.SurfaceFormatKHR {
	format     = .R8G8B8A8_UNORM,
	colorSpace = .SRGB_NONLINEAR,
}
DEFAULT_PREFERRED_PRESENT_MODE :: vk.PresentModeKHR.IMMEDIATE


Swapchain_Desc :: struct {
	preferred_surface_format: Maybe(vk.SurfaceFormatKHR),
	preferred_present_mode:   Maybe(vk.PresentModeKHR),
}

// swapchain is handled individually in order to
// allow using the abstraction without a window (for offline tasks for example)
create_swapchain :: proc(
	d: ^Device,
	window: glfw.WindowHandle,
	desc: Swapchain_Desc,
) -> (
	swapchain: Swapchain,
) {
	avbailable_surface_formats: []vk.SurfaceFormatKHR
	available_present_modes: []vk.PresentModeKHR
	surface: vk.SurfaceKHR
	surface_caps: vk.SurfaceCapabilitiesKHR
	// surface capabilities
	{
		depth_formats := [?]vk.Format {
			.D32_SFLOAT_S8_UINT,
			.D24_UNORM_S8_UINT,
			.D16_UNORM_S8_UINT,
			.D32_SFLOAT,
			.D16_UNORM,
		}
		d.available_depth_formats = make([dynamic]vk.Format)
		for format in depth_formats {
			format_props := vk.FormatProperties2 {
				sType = .FORMAT_PROPERTIES_2,
			}
			vk.GetPhysicalDeviceFormatProperties2(d.physical_device, format, &format_props)
			if vk.FormatFeatureFlag.DEPTH_STENCIL_ATTACHMENT in
			   format_props.formatProperties.optimalTilingFeatures {
				append(&d.available_depth_formats, format)
				break
			}
		}

		chk(glfw.CreateWindowSurface(d.instance, window, nil, &surface))

		vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(d.physical_device, surface, &surface_caps)
		format_count: u32
		vk.GetPhysicalDeviceSurfaceFormatsKHR(d.physical_device, surface, &format_count, nil)
		vk.GetPhysicalDeviceSurfaceFormatsKHR(d.physical_device, surface, &format_count, nil)
		if format_count == 0 do fmt.panicf("[Swapchain] No surface formats available")
		avbailable_surface_formats = make(
			[]vk.SurfaceFormatKHR,
			format_count,
			context.temp_allocator,
		)
		vk.GetPhysicalDeviceSurfaceFormatsKHR(
			d.physical_device,
			surface,
			&format_count,
			raw_data(avbailable_surface_formats),
		)

		mode_count: u32
		vk.GetPhysicalDeviceSurfacePresentModesKHR(d.physical_device, surface, &mode_count, nil)
		if mode_count == 0 do fmt.panicf("[Swapchain] No present modes available")
		available_present_modes = make([]vk.PresentModeKHR, mode_count, context.temp_allocator)
		vk.GetPhysicalDeviceSurfacePresentModesKHR(
			d.physical_device,
			surface,
			&mode_count,
			raw_data(available_present_modes),
		)
	}

	// swapchain
	{
		/*
		Try getting the preffered format + color space,
		If you can't, try getting the format at least,
		If this also fails, get the first available format
		*/
		surface_format, same := choose_surface_format(
			desc.preferred_surface_format.? or_else DEFAULT_PREFERRED_SURFACE_FORMAT,
			avbailable_surface_formats,
		)
		if !same {
			fmt.printfln(
				"[Device] Preferred surface format was not available. %v has been chosen",
				surface_format,
			)
		}

		// preferred -> immediate -> fifo
		present_mode, same_ := choose_present_mode(
			desc.preferred_present_mode.? or_else DEFAULT_PREFERRED_PRESENT_MODE,
			available_present_modes,
		)
		if !same_ {
			fmt.printfln(
				"[Device] Preferred presentation mode was not available. %v has been chosen",
				present_mode,
			)
		}

		desired_image_count := surface_caps.minImageCount + 1
		image_count := max(surface_caps.maxImageCount, desired_image_count)

		swapchain.device = d

		// inlined swapchain_init
		swapchain.surface = surface
		swapchain.format = surface_format.format
		family_idx := swapchain.device.queues.graphics_family_idx
		swapchain.create_info = vk.SwapchainCreateInfoKHR {
			sType                 = .SWAPCHAIN_CREATE_INFO_KHR,
			surface               = surface,
			minImageCount         = image_count,
			imageFormat           = surface_format.format,
			imageColorSpace       = surface_format.colorSpace,
			imageExtent           = surface_caps.currentExtent,
			imageArrayLayers      = 1,
			imageUsage            = {.COLOR_ATTACHMENT},
			imageSharingMode      = .EXCLUSIVE,
			preTransform          = {.IDENTITY},
			queueFamilyIndexCount = 1,
			pQueueFamilyIndices   = &family_idx,
			compositeAlpha        = {.OPAQUE},
			presentMode           = present_mode,
			clipped               = true,
		}
		chk(
			vk.CreateSwapchainKHR(
				swapchain.device.device,
				&swapchain.create_info,
				nil,
				&swapchain.swapchain,
			),
		)

		// for each swapchain image, store the acquire semaphore and
		// the command buffer submit handle which draws to the image
		swapchain_image_count: u32
		chk(
			vk.GetSwapchainImagesKHR(
				swapchain.device.device,
				swapchain.swapchain,
				&swapchain_image_count,
				nil,
			),
		)
		swapchain.images = make([]vk.Image, swapchain_image_count)
		chk(
			vk.GetSwapchainImagesKHR(
				swapchain.device.device,
				swapchain.swapchain,
				&swapchain_image_count,
				raw_data(swapchain.images),
			),
		)
		swapchain.image_views = make([]vk.ImageView, swapchain_image_count)
		semaphore_ci := vk.SemaphoreCreateInfo {
			sType = .SEMAPHORE_CREATE_INFO,
		}
		swapchain.acquire_semaphores = make([]vk.Semaphore, swapchain_image_count)
		swapchain.timeline_wait_values = make([]u64, swapchain_image_count)
		for i in 0 ..< len(swapchain.images) {
			view_ci := vk.ImageViewCreateInfo {
				sType = .IMAGE_VIEW_CREATE_INFO,
				image = swapchain.images[i],
				viewType = .D2,
				format = surface_format.format,
				subresourceRange = {
					aspectMask = {.COLOR},
					baseMipLevel = 0,
					levelCount = 1,
					baseArrayLayer = 0,
					layerCount = 1,
				},
			}
			chk(
				vk.CreateImageView(
					swapchain.device.device,
					&view_ci,
					nil,
					&swapchain.image_views[i],
				),
			)
			chk(
				vk.CreateSemaphore(
					swapchain.device.device,
					&semaphore_ci,
					nil,
					&swapchain.acquire_semaphores[i],
				),
			)
		}

		timeline_ci := vk.SemaphoreTypeCreateInfo {
			sType         = .SEMAPHORE_TYPE_CREATE_INFO,
			semaphoreType = .TIMELINE,
			initialValue  = 0,
		}
		semaphore_ci.pNext = &timeline_ci
		chk(
			vk.CreateSemaphore(
				swapchain.device.device,
				&semaphore_ci,
				nil,
				&swapchain.timeline_semaphore,
			),
		)
		swapchain.need_acquire = true
	}

	return swapchain
}

create_pipeline_manager :: proc(
	d: ^Device,
	shader_directory: string,
	compile_shader_directory: string,
) -> Pipeline_Manager {
	return {
		device = d,
		raster_pipelines = make(map[string]Raster_Pipeline),
		shader_directory = shader_directory,
		compile_shader_directory = compile_shader_directory,
	}
}

Memory_Preset :: enum {
	GPU_ONLY,
	CPU_UPLOAD,
	CPU_READBACK,
}

Buffer :: struct {
	buffer:         vk.Buffer,
	memory:         vk.DeviceMemory,
	device_address: vk.DeviceAddress,
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

Image :: struct {
	image:          vk.Image,
	view:           vk.ImageView,
	memory:         vk.DeviceMemory,
	format:         vk.Format,
	bindless_index: u32,
}

Image_Create_Desc :: struct {
	width:             u32,
	height:            u32,
	format:            vk.Format,
	usage:             vk.ImageUsageFlags,
	memory:            Memory_Preset,
	// if true (and usage includes .STORAGE), registers the image into the
	// bindless set immediately and fills in Image.bindless_index
	register_bindless: bool,
}

create_image :: proc(device: ^Device, desc: Image_Create_Desc) -> Image {
	out: Image
	out.format = desc.format

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
			layerCount = 1,
			levelCount = 1,
			aspectMask = ({.DEPTH} if desc.format == vk.Format.D32_SFLOAT || desc.format == vk.Format.D16_UNORM else {.COLOR}),
		},
	}
	chk(vk.CreateImageView(device.device, &view_ci, nil, &out.view))

	// register into the bindless set right away so callers can stash the
	// index straight into a push constant, no separate call needed
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

// Registers a sampler into the bindless set. Returns the index to use in
// the GLSL `samplers[]` array.
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

// ────────────────────────────────────────────────────────────────
// Command helpers

bind_pipeline :: proc(cb: vk.CommandBuffer, pipeline: ^Raster_Pipeline) {
	vk.CmdBindPipeline(cb, .GRAPHICS, pipeline.pipeline)
	vk.CmdBindDescriptorSets(cb, .GRAPHICS, pipeline.layout, 0, 1, &pipeline.bindless_set, 0, nil)
}

// ────────────────────────────────────────────────────────────────

@(private = "file")
choose_surface_format :: proc(
	preferred: vk.SurfaceFormatKHR,
	available: []vk.SurfaceFormatKHR,
) -> (
	format: vk.SurfaceFormatKHR,
	same: bool,
) {
	for format in available {
		if preferred.format == format.format && preferred.colorSpace == format.colorSpace {
			return format, true
		}
	}
	for format in available {
		if preferred.format == format.format {
			return format, false
		}
	}
	return available[0], false
}

@(private = "file")
choose_present_mode :: proc(
	preferred: vk.PresentModeKHR,
	available: []vk.PresentModeKHR,
) -> (
	format: vk.PresentModeKHR,
	same: bool,
) {
	for mode in available {
		if mode == preferred {
			return mode, true
		}
	}
	for mode in available {
		if mode == .IMMEDIATE {
			return mode, false
		}
	}
	return .FIFO, false
}

@(private = "file")
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

@(private = "file")
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
