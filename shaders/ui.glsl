#version 460 core

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "luma.glsl"

struct UiVertex {
    vec2 pos;
    vec2 uv;
    vec4 color;
};

layout(buffer_reference, std430) readonly buffer UiVertexBuffer {
    UiVertex vertices[];
};

layout(push_constant) uniform PushConstants {
    UiVertexBuffer vertex_buffer;
    vec2 screen_size;
    uint atlas_texture;
    uint atlas_sampler;
} push;

#ifdef STAGE_VERTEX

layout(location = 0) out vec2 uv;
layout(location = 1) out vec4 color;

void vert_main() {
    UiVertex v = push.vertex_buffer.vertices[gl_VertexIndex];

    vec2 ndc = (v.pos / push.screen_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    uv = v.uv;
    color = v.color;
}

#endif

#ifdef STAGE_FRAGMENT

layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 frag_color;

void frag_main() {
    float alpha = texture(TEX_UNI(push.atlas_texture, push.atlas_sampler), uv).r;
    frag_color = vec4(color.rgb, color.a * alpha);
}

#endif
