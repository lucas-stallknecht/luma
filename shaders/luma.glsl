// Common GLSL definitions

#ifndef LUMA_GLSL_INCLUDED
#define LUMA_GLSL_INCLUDED

#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform sampler samplers[];
layout(set = 0, binding = 1) uniform texture2D textures[];

// each storage format needs its own typed array
layout(set = 0, binding = 2, rg32ui) uniform uimage2D images_u32[];
layout(set = 0, binding = 3, rgba32f) uniform image2D images_f32[];
layout(set = 0, binding = 4, rgba8) uniform image2D images_rgba8[];

#define SAMP(idx) samplers[nonuniformEXT(idx)]
#define U32(idx) images_u32[nonuniformEXT(idx)]
#define F32(idx) images_f32[nonuniformEXT(idx)]
#define RGBA8(idx) images_rgba8[nonuniformEXT(idx)]

// glue a texture index and a sampler index together into something we can texture() with
#define TEX(tex_idx, samp_idx) sampler2D(textures[nonuniformEXT(tex_idx)], samplers[nonuniformEXT(samp_idx)])

#endif
