// Common GLSL definitions

#ifndef LUMA_GLSL_INCLUDED
#define LUMA_GLSL_INCLUDED

#extension GL_EXT_nonuniform_qualifier : require

// One array per concrete storage format in use. A format qualifier lets an
// image be both imageLoad'd and imageStore'd freely - add a new array/binding
// here (and a matching case in vulkan.odin's storage_image_binding) the day
// a third format is needed.
layout(set = 0, binding = 0) uniform sampler samplers[];
layout(set = 0, binding = 1, r32ui) uniform uimage2D images_u32[];
layout(set = 0, binding = 2, rgba8) uniform image2D images_rgba8[];

#define SAMP(idx) samplers[nonuniformEXT(idx)]
#define U32(idx) images_u32[nonuniformEXT(idx)]
#define RGBA8(idx) images_rgba8[nonuniformEXT(idx)]

#endif
