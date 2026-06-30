#version 460 core
#extension GL_EXT_buffer_reference : require
#include "luma.glsl"

layout(push_constant) uniform PushConstants {
    uint draw_image;
} push;

layout(location = 0) out vec4 frag_color;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec4 hdr_col = imageLoad(F32(push.draw_image), coord);

    frag_color = vec4(hdr_col.rgb, 1.0);
}
