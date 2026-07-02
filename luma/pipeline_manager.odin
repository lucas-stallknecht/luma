package luma

import "core:fmt"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"
import vk "vendor:vulkan"

// ────────────────────────────────────────────────────────────────
// Manager

Pipeline_Manager :: struct {
	device:                   ^Device,
	shader_directory:         string,
	compile_shader_directory: string,
	raster_pipelines:         map[string]Raster_Pipeline,
	compute_pipelines:        map[string]Compute_Pipeline,
}

MAX_COLOR_ATTACHMENTS :: 4

create_pipeline_manager :: proc(
	d: ^Device,
	shader_directory: string,
	compile_shader_directory: string,
) -> Pipeline_Manager {
	return {
		device = d,
		raster_pipelines = make(map[string]Raster_Pipeline),
		compute_pipelines = make(map[string]Compute_Pipeline),
		shader_directory = shader_directory,
		compile_shader_directory = compile_shader_directory,
	}
}

pipeline_manager_cleanup :: proc(m: ^Pipeline_Manager) {
	for _, &pipeline in m.raster_pipelines {
		destroy_raster_pipeline(m.device.device, &pipeline)
		raster_info_free(&pipeline.info)
	}
	delete(m.raster_pipelines)
	for _, &pipeline in m.compute_pipelines {
		destroy_compute_pipeline(m.device.device, &pipeline)
		compute_info_free(&pipeline.info)
	}
	delete(m.compute_pipelines)
}

// compiles everything first, only swaps pipelines if it all succeeded -
// one broken shader shouldn't kill the ones that still work
pipeline_reload_all :: proc(m: ^Pipeline_Manager) -> bool {
	raster_entries := make([dynamic]struct {
			name:     string,
			vertex:   []u32,
			fragment: []u32,
		}, context.temp_allocator)
	compute_entries := make([dynamic]struct {
			name: string,
			code: []u32,
		}, context.temp_allocator)

	all_ok := true
	for name, &pipeline in m.raster_pipelines {
		v, v_ok := compile_and_load_spirv(m, pipeline.info.vertex_shader)
		f, f_ok := compile_and_load_spirv(m, pipeline.info.fragment_shader)
		if !(v_ok && f_ok) {
			all_ok = false
			// keep compiling the rest so the user gets full diagnostics
			continue
		}
		append(&raster_entries, struct {
			name:     string,
			vertex:   []u32,
			fragment: []u32,
		}{name = name, vertex = v, fragment = f})
	}
	for name, &pipeline in m.compute_pipelines {
		c, ok := compile_and_load_spirv(m, pipeline.info.shader)
		if !ok {
			all_ok = false
			continue
		}
		append(&compute_entries, struct {
			name: string,
			code: []u32,
		}{name = name, code = c})
	}

	if !all_ok do return false

	vk.DeviceWaitIdle(m.device.device)
	for entry in raster_entries {
		p := &m.raster_pipelines[entry.name]
		p.cached_spirv.vertex = entry.vertex
		p.cached_spirv.fragment = entry.fragment
	}
	for entry in compute_entries {
		p := &m.compute_pipelines[entry.name]
		p.cached_spirv = entry.code
	}

	for _, &pipeline in m.raster_pipelines {
		destroy_raster_pipeline(m.device.device, &pipeline)
		create_raster_pipeline(m, &pipeline)
	}
	for _, &pipeline in m.compute_pipelines {
		destroy_compute_pipeline(m.device.device, &pipeline)
		create_compute_pipeline(m, &pipeline)
	}

	return true
}

// ────────────────────────────────────────────────────────────────
// Raster

Raster_Pipeline :: struct {
	pipeline:     vk.Pipeline,
	layout:       vk.PipelineLayout,
	info:         Raster_Pipeline_Info,
	cached_spirv: struct {
		vertex:   []u32,
		fragment: []u32,
	},
	bindless_set: vk.DescriptorSet,
}

Rasterizer_Info :: struct {
	primitive_topology:  vk.PrimitiveTopology,
	polygon_mode:        vk.PolygonMode,
	cull_mode:           vk.CullModeFlags,
	front_face:          vk.FrontFace,
	line_width:          f32,
	depth_clamp:         bool,
	depth_bias_enable:   bool,
	depth_bias_constant: f32,
	depth_bias_clamp:    f32,
	depth_bias_slope:    f32,
}

Blend_Preset :: enum {
	Additive,
	Alpha,
	Pre_Multiplied,
}

Depth_Test :: struct {
	format:             vk.Format,
	enable_depth_write: bool,
	compare_op:         vk.CompareOp,
}

Raster_Pipeline_Info :: struct {
	vertex_shader:      string,
	fragment_shader:    string,
	color_attachments:  []struct {
		format: vk.Format,
		blend:  Maybe(Blend_Preset),
	}, // up to MAX_COLOR_ATTACHMENTS
	depth_test:         Maybe(Depth_Test),
	raster:             Rasterizer_Info,
	push_constant_size: u32,
	name:               string, // Debug label (used by the manager as the pipeline key)
}

pipeline_manager_add_raster :: proc(
	m: ^Pipeline_Manager,
	info: Raster_Pipeline_Info,
) -> ^Raster_Pipeline {
	assert(info.name != "", "pipeline must have a non-empty name")
	assert(
		info.name not_in m.raster_pipelines,
		"pipeline already registered. Use pipeline_reload_all",
	)
	info2 := raster_info_clone(info)
	pipeline := map_insert(
		&m.raster_pipelines,
		info2.name,
		Raster_Pipeline{info = info2, bindless_set = m.device.descriptor_set},
	)
	create_raster_pipeline(m, pipeline)

	return pipeline
}

pipeline_manager_get_raster :: proc(m: ^Pipeline_Manager, name: string) -> ^Raster_Pipeline {
	return &m.raster_pipelines[name]
}

pipeline_manager_remove_raster :: proc(m: ^Pipeline_Manager, name: string) {
	pipeline, found := &m.raster_pipelines[name]
	if !found do return
	destroy_raster_pipeline(m.device.device, pipeline)
	raster_info_free(&pipeline.info)
	delete_key(&m.raster_pipelines, name)
}

bind_raster_pipeline :: proc(cb: vk.CommandBuffer, pipeline: ^Raster_Pipeline) {
	vk.CmdBindPipeline(cb, .GRAPHICS, pipeline.pipeline)
	vk.CmdBindDescriptorSets(cb, .GRAPHICS, pipeline.layout, 0, 1, &pipeline.bindless_set, 0, nil)
}

@(private = "file")
create_raster_pipeline :: proc(m: ^Pipeline_Manager, pipeline: ^Raster_Pipeline) {
	info := &pipeline.info

	// if cached SPIR-V is empty, try compiling the shaders. When performing
	// a reload we pre-populate cached_spirv, so this avoids recompiling.
	if len(pipeline.cached_spirv.vertex) == 0 {
		vertex_spv, vertex_ok := compile_and_load_spirv(m, info.vertex_shader)
		if vertex_ok {
			pipeline.cached_spirv.vertex = vertex_spv
		}
	}
	if len(pipeline.cached_spirv.fragment) == 0 {
		fragment_spv, fragment_ok := compile_and_load_spirv(m, info.fragment_shader)
		if fragment_ok {
			pipeline.cached_spirv.fragment = fragment_spv
		}
	}

	modules: [2]vk.ShaderModule
	defer {
		for mod in modules {
			if mod != 0 {
				vk.DestroyShaderModule(m.device.device, mod, nil)
			}
		}
	}
	stages: [2]vk.PipelineShaderStageCreateInfo
	push_stage(
		m.device.device,
		Shader_Info{byte_code = pipeline.cached_spirv.vertex},
		.VERTEX,
		&modules[0],
		&stages[0],
	)
	push_stage(
		m.device.device,
		Shader_Info{byte_code = pipeline.cached_spirv.fragment},
		.FRAGMENT,
		&modules[1],
		&stages[1],
	)

	// we pull vertices from device address so we don't need to specify vertex bindings
	vertex_input := vk.PipelineVertexInputStateCreateInfo {
		sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	}
	input_assembly := vk.PipelineInputAssemblyStateCreateInfo {
		sType    = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology = info.raster.primitive_topology,
	}
	raster := vk.PipelineRasterizationStateCreateInfo {
		sType                   = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		depthClampEnable        = b32(info.raster.depth_clamp),
		polygonMode             = info.raster.polygon_mode,
		cullMode                = info.raster.cull_mode,
		frontFace               = info.raster.front_face,
		depthBiasEnable         = b32(info.raster.depth_bias_enable),
		depthBiasConstantFactor = info.raster.depth_bias_constant,
		depthBiasClamp          = info.raster.depth_bias_clamp,
		depthBiasSlopeFactor    = info.raster.depth_bias_slope,
		lineWidth               = max(info.raster.line_width, 1.0),
	}
	multisample := vk.PipelineMultisampleStateCreateInfo {
		sType                = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		rasterizationSamples = {._1},
	}

	depth_stencil := vk.PipelineDepthStencilStateCreateInfo {
		sType = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	}
	if dt, has := info.depth_test.?; has {
		depth_stencil.depthTestEnable = true
		depth_stencil.depthWriteEnable = b32(dt.enable_depth_write)
		depth_stencil.depthCompareOp = dt.compare_op
	}

	color_blend_attachments := make(
		[]vk.PipelineColorBlendAttachmentState,
		len(info.color_attachments),
		context.temp_allocator,
	)
	for ca, i in info.color_attachments {
		blend, has_blend := ca.blend.?
		if !has_blend {
			color_blend_attachments[i] = {
				colorWriteMask = {.R, .G, .B, .A},
			}
			continue
		}
		switch ca.blend {
		case .Alpha:
			color_blend_attachments[i] = vk.PipelineColorBlendAttachmentState {
				blendEnable         = true,
				srcColorBlendFactor = .SRC_ALPHA,
				dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
				colorBlendOp        = .ADD,
				srcAlphaBlendFactor = .ONE,
				dstAlphaBlendFactor = .ZERO,
				alphaBlendOp        = .ADD,
				colorWriteMask      = {.R, .G, .B, .A},
			}
		case .Additive:
			color_blend_attachments[i] = vk.PipelineColorBlendAttachmentState {
				blendEnable         = true,
				srcColorBlendFactor = .ONE,
				dstColorBlendFactor = .ONE,
				colorBlendOp        = .ADD,
				srcAlphaBlendFactor = .ONE,
				dstAlphaBlendFactor = .ONE,
				alphaBlendOp        = .ADD,
				colorWriteMask      = {.R, .G, .B, .A},
			}
		case .Pre_Multiplied:
			color_blend_attachments[i] = vk.PipelineColorBlendAttachmentState {
				blendEnable         = true,
				srcColorBlendFactor = .ONE,
				dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
				colorBlendOp        = .ADD,
				srcAlphaBlendFactor = .ONE,
				dstAlphaBlendFactor = .ONE_MINUS_SRC_ALPHA,
				alphaBlendOp        = .ADD,
				colorWriteMask      = {.R, .G, .B, .A},
			}
		}

	}
	color_blend_ci := vk.PipelineColorBlendStateCreateInfo {
		sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		attachmentCount = u32(len(color_blend_attachments)),
		pAttachments    = raw_data(color_blend_attachments),
	}

	dyn_states := [?]vk.DynamicState{.VIEWPORT_WITH_COUNT, .SCISSOR_WITH_COUNT}
	dynamic_state := vk.PipelineDynamicStateCreateInfo {
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = u32(len(dyn_states)),
		pDynamicStates    = raw_data(&dyn_states),
	}

	// dynamic rendering attachment formats
	color_fmts := make([]vk.Format, len(info.color_attachments), context.temp_allocator)
	for ca, i in info.color_attachments {color_fmts[i] = ca.format}

	rendering := vk.PipelineRenderingCreateInfo {
		sType                   = .PIPELINE_RENDERING_CREATE_INFO,
		colorAttachmentCount    = u32(len(color_fmts)),
		pColorAttachmentFormats = raw_data(color_fmts),
	}
	if dt, has := info.depth_test.?; has {rendering.depthAttachmentFormat = dt.format}

	pc_range := vk.PushConstantRange {
		stageFlags = {.VERTEX, .FRAGMENT},
		size       = info.push_constant_size,
	}
	layout_ci := vk.PipelineLayoutCreateInfo {
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount         = 1,
		pSetLayouts            = &m.device.descriptor_layout,
		pushConstantRangeCount = 1 if info.push_constant_size > 0 else 0,
		pPushConstantRanges    = &pc_range if info.push_constant_size > 0 else nil,
	}
	chk(vk.CreatePipelineLayout(m.device.device, &layout_ci, nil, &pipeline.layout))

	pipeline_ci := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		pNext               = &rendering,
		stageCount          = u32(len(stages)),
		pStages             = raw_data(&stages),
		pVertexInputState   = &vertex_input,
		pInputAssemblyState = &input_assembly,
		pRasterizationState = &raster,
		pMultisampleState   = &multisample,
		pDepthStencilState  = &depth_stencil,
		pColorBlendState    = &color_blend_ci,
		pDynamicState       = &dynamic_state,
		layout              = pipeline.layout,
	}
	chk(vk.CreateGraphicsPipelines(m.device.device, 0, 1, &pipeline_ci, nil, &pipeline.pipeline))
}

@(private = "file")
destroy_raster_pipeline :: proc(device: vk.Device, pipeline: ^Raster_Pipeline) {
	if pipeline.pipeline != 0 {
		vk.DestroyPipeline(device, pipeline.pipeline, nil)
	}
	if pipeline.layout != 0 {
		vk.DestroyPipelineLayout(device, pipeline.layout, nil)
	}
}

@(private = "file")
raster_info_clone :: proc(src: Raster_Pipeline_Info) -> (dst: Raster_Pipeline_Info) {
	dst = src
	dst.vertex_shader = strings.clone(src.vertex_shader)
	dst.fragment_shader = strings.clone(src.fragment_shader)
	dst.color_attachments = slice.clone(src.color_attachments)
	dst.name = strings.clone(src.name)
	return
}

@(private = "file")
raster_info_free :: proc(info: ^Raster_Pipeline_Info) {
	delete(info.vertex_shader)
	delete(info.fragment_shader)
	delete(info.color_attachments)
	delete(info.name)
}

// ────────────────────────────────────────────────────────────────
// Compute

Compute_Pipeline :: struct {
	pipeline:     vk.Pipeline,
	layout:       vk.PipelineLayout,
	info:         Compute_Pipeline_Info,
	cached_spirv: []u32,
	bindless_set: vk.DescriptorSet,
}

Compute_Pipeline_Info :: struct {
	shader:             string,
	push_constant_size: u32,
	name:               string, // Debug label (used by the manager as the pipeline key)
}

pipeline_manager_add_compute :: proc(
	m: ^Pipeline_Manager,
	info: Compute_Pipeline_Info,
) -> ^Compute_Pipeline {
	assert(info.name != "", "pipeline must have a non-empty name")
	assert(
		info.name not_in m.compute_pipelines,
		"pipeline already registered. Use pipeline_reload_all",
	)
	info2 := compute_info_clone(info)
	pipeline := map_insert(
		&m.compute_pipelines,
		info2.name,
		Compute_Pipeline{info = info2, bindless_set = m.device.descriptor_set},
	)
	create_compute_pipeline(m, pipeline)

	return pipeline
}

pipeline_manager_get_compute :: proc(m: ^Pipeline_Manager, name: string) -> ^Compute_Pipeline {
	return &m.compute_pipelines[name]
}

pipeline_manager_remove_compute :: proc(m: ^Pipeline_Manager, name: string) {
	pipeline, found := &m.compute_pipelines[name]
	if !found do return
	destroy_compute_pipeline(m.device.device, pipeline)
	compute_info_free(&pipeline.info)
	delete_key(&m.compute_pipelines, name)
}

bind_compute_pipeline :: proc(cb: vk.CommandBuffer, pipeline: ^Compute_Pipeline) {
	vk.CmdBindPipeline(cb, .COMPUTE, pipeline.pipeline)
	vk.CmdBindDescriptorSets(cb, .COMPUTE, pipeline.layout, 0, 1, &pipeline.bindless_set, 0, nil)
}

@(private = "file")
create_compute_pipeline :: proc(m: ^Pipeline_Manager, pipeline: ^Compute_Pipeline) {
	info := &pipeline.info

	if len(pipeline.cached_spirv) == 0 {
		spv, ok := compile_and_load_spirv(m, info.shader)
		if ok {
			pipeline.cached_spirv = spv
		}
	}

	module: vk.ShaderModule
	defer if module != 0 {
		vk.DestroyShaderModule(m.device.device, module, nil)
	}
	stage: vk.PipelineShaderStageCreateInfo
	push_stage(
		m.device.device,
		Shader_Info{byte_code = pipeline.cached_spirv},
		.COMPUTE,
		&module,
		&stage,
	)

	pc_range := vk.PushConstantRange {
		stageFlags = {.COMPUTE},
		size       = info.push_constant_size,
	}
	layout_ci := vk.PipelineLayoutCreateInfo {
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount         = 1,
		pSetLayouts            = &m.device.descriptor_layout,
		pushConstantRangeCount = 1 if info.push_constant_size > 0 else 0,
		pPushConstantRanges    = &pc_range if info.push_constant_size > 0 else nil,
	}
	chk(vk.CreatePipelineLayout(m.device.device, &layout_ci, nil, &pipeline.layout))

	pipeline_ci := vk.ComputePipelineCreateInfo {
		sType  = .COMPUTE_PIPELINE_CREATE_INFO,
		stage  = stage,
		layout = pipeline.layout,
	}
	chk(vk.CreateComputePipelines(m.device.device, 0, 1, &pipeline_ci, nil, &pipeline.pipeline))
}

@(private = "file")
destroy_compute_pipeline :: proc(device: vk.Device, pipeline: ^Compute_Pipeline) {
	if pipeline.pipeline != 0 {
		vk.DestroyPipeline(device, pipeline.pipeline, nil)
	}
	if pipeline.layout != 0 {
		vk.DestroyPipelineLayout(device, pipeline.layout, nil)
	}
}

@(private = "file")
compute_info_free :: proc(info: ^Compute_Pipeline_Info) {
	delete(info.shader)
	delete(info.name)
}

@(private = "file")
compute_info_clone :: proc(src: Compute_Pipeline_Info) -> (dst: Compute_Pipeline_Info) {
	dst = src
	dst.shader = strings.clone(src.shader)
	dst.name = strings.clone(src.name)
	return
}

// ────────────────────────────────────────────────────────────────
// Shader compiling

Shader_Info :: struct {
	byte_code:   []u32,
	entry_point: cstring,
}

@(private = "file")
push_stage :: proc(
	device: vk.Device,
	si: Shader_Info,
	flag: vk.ShaderStageFlag,
	module: ^vk.ShaderModule,
	stage: ^vk.PipelineShaderStageCreateInfo,
) {
	mci := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(si.byte_code) * size_of(u32),
		pCode    = raw_data(si.byte_code),
	}
	chk(vk.CreateShaderModule(device, &mci, nil, module))
	stage^ = vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {flag},
		module = module^,
		pName  = si.entry_point if si.entry_point != nil else "main",
	}
}

@(private = "file")
compile_and_load_spirv :: proc(m: ^Pipeline_Manager, path: string) -> ([]u32, bool) #optional_ok {
	src_path := strings.concatenate({m.shader_directory, path}, context.temp_allocator)
	dst_path := strings.concatenate(
		{m.compile_shader_directory, path, ".spv"},
		context.temp_allocator,
	)

	dst_dir := filepath.dir(dst_path, context.temp_allocator)
	if !os.exists(dst_dir) {
		if err := os.make_directory_all(dst_dir); err != nil {
			fmt.eprintfln("failed to create shader output directory %q: %v", dst_dir, err)
			return nil, false
		}
	}

	state, stdout, stderr, proc_err := os.process_exec(
		{command = {"glslc", "--target-env=vulkan1.3", src_path, "-o", dst_path}},
		context.temp_allocator,
	)
	if proc_err != nil {
		fmt.eprintfln("failed to run glslc: %v", proc_err)
		return nil, false
	}
	if state.exit_code != 0 {
		fmt.eprintfln("glslc failed with exit code %d", state.exit_code)
		fmt.eprintfln("{}", string(stderr))
		return nil, false
	}

	data, read_err := os.read_entire_file(dst_path, context.temp_allocator)
	if read_err != nil ||
	   data == nil ||
	   len(data) < size_of(u32) ||
	   len(data) % size_of(u32) != 0 {
		fmt.eprintfln("[%s] failed to read SPIR-V: %d", dst_path, read_err)
		return nil, false
	}
	words := mem.slice_data_cast([]u32, data)

	if words[0] != 0x07230203 {
		fmt.eprintfln("[%s] Invalid SPIR-V file", dst_path)
		return nil, false
	}

	return words, true
}
