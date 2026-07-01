#version 460 core
#include "luma.glsl"

layout(push_constant) uniform PushConstants {
    uvec2 _vertex_buffer; // unused here, kept only so the layout matches ui.vert's push constants
    vec2 screen_size;
    uint atlas_texture;
    uint atlas_sampler;
} push;

layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 frag_color;

void main() {
    float alpha = texture(TEX(push.atlas_texture, push.atlas_sampler), uv).r;
    frag_color = vec4(color.rgb, color.a * alpha);
}
