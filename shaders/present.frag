#version 460 core
#include "luma.glsl"

layout(push_constant) uniform PushConstants {
    uint visbuffer_idx;
} push;

layout(location = 0) out vec4 frag_color;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    uvec4 pix = imageLoad(U32(push.visbuffer_idx), coord);

    uint id = pix.r & 0xFFFFFFFFu;
    uint r = id & 0xFFu;
    uint g = (id >> 8) & 0xFFu;
    uint b = (id >> 16) & 0xFFu;

    frag_color = vec4(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0);
}
