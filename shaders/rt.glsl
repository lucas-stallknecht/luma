#ifndef RT_GLSL_INCLUDED
#define RT_GLSL_INCLUDED

#extension GL_EXT_ray_tracing : require

layout(binding = 0, set = 1) uniform accelerationStructureEXT tlas;

#ifdef RT_PAYLOAD_IN
layout(location = 0) rayPayloadInEXT vec3 payload;
#else
layout(location = 0) rayPayloadEXT vec3 payload;
#endif

#endif
