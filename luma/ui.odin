package luma

import imgui "../imgui"
import imgui_glfw "../imgui/imgui_impl_glfw"
import imgui_vk "../imgui/imgui_impl_vulkan"
import vk "vendor:vulkan"

ui_init :: proc(device: ^Device, window: ^Window, swapchain: ^Swapchain) {
	imgui.CreateContext()
	imgui.StyleColorsDark()

	io := imgui.GetIO()
	io.IniFilename = nil

	// we own the GLFW callbacks ourselves (camera look/movement live alongside them), so
	// install_callbacks=false and window.odin forwards events into the backend manually
	imgui_glfw.InitForVulkan(window.glfw_window_ptr, false)

	imgui_vulkan_loader :: proc "c" (
		function_name: cstring,
		user_data: rawptr,
	) -> vk.ProcVoidFunction {
		instance := (^vk.Instance)(user_data)^
		return vk.GetInstanceProcAddr(instance, function_name)
	}
	imgui_vk.LoadFunctions(vk.API_VERSION_1_3, imgui_vulkan_loader, &device.instance)

	image_count := u32(len(swapchain.images))
	init_info := imgui_vk.InitInfo {
		ApiVersion          = vk.API_VERSION_1_3,
		Instance            = device.instance,
		PhysicalDevice      = device.physical_device,
		Device              = device.device,
		QueueFamily         = device.queues.graphics_family_idx,
		Queue               = device.queues.graphics,
		DescriptorPoolSize  = 8, // let the backend create + own its own small descriptor pool
		MinImageCount       = image_count,
		ImageCount          = image_count,
		UseDynamicRendering = true,
	}
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo = vk.PipelineRenderingCreateInfo {
		sType                   = .PIPELINE_RENDERING_CREATE_INFO,
		colorAttachmentCount    = 1,
		pColorAttachmentFormats = cast([^]vk.Format)&swapchain.format,
	}
	imgui_vk.Init(&init_info)
}

ui_cleanup :: proc() {
	imgui_vk.Shutdown()
	imgui_glfw.Shutdown()
	imgui.DestroyContext()
}

ui_new_frame :: proc() {
	imgui_vk.NewFrame()
	imgui_glfw.NewFrame()
	imgui.NewFrame()
}

ui_draw :: proc(cb: vk.CommandBuffer) {
	imgui.Render()
	imgui_vk.RenderDrawData(imgui.GetDrawData(), cb)
}
