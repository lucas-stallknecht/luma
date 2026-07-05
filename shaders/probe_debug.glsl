#version 460 core

#include "luma.glsl"
#include "types.glsl"
#include "probe_global.glsl"

layout(push_constant) uniform PushConstants
{
    FrameDataBuffer frame_data;
    VertexBuffer vertex_buffer;
    NormalBuffer normal_buffer;
    ProbePositionBuffer probe_position_buffer;
    ProbeSHBuffer probe_sh_buffer;
} push;

#ifdef STAGE_VERTEX

#define SPHERE_SIZE 0.05

layout(location = 0) out vec3 normal;
layout(location = 1) out flat uint probe_idx;

void vert_main() {
    vec3 probe_pos = push.probe_position_buffer.positions[gl_InstanceIndex];
    vec3 v_pos = push.vertex_buffer.positions[gl_VertexIndex] * SPHERE_SIZE + probe_pos;
    vec3 v_normal = push.normal_buffer.normals[gl_VertexIndex];

    gl_Position = push.frame_data.data.proj_view * vec4(v_pos, 1.0);
    normal = v_normal;
    probe_idx = gl_InstanceIndex;
}

#endif

#ifdef STAGE_FRAGMENT

layout(location = 0) in vec3 normal;
layout(location = 1) in flat uint probe_idx;
layout(location = 0) out vec4 frag_color;

void frag_main() {
    vec3 irradiance = sh_irradiance(push.probe_sh_buffer.probes[probe_idx], normalize(normal));
    frag_color = vec4(irradiance, 1.0);
}

#endif
