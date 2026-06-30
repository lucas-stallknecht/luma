#version 460 core
#include "luma.glsl"

layout(push_constant) uniform PushConstants {
    uint draw_image;
} push;

layout(location = 0) out vec4 frag_color;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec4 hdr_col = imageLoad(F32(push.draw_image), coord);
    vec3 final_color = pow(hdr_col.rgb, vec3(1.0 / 2.2));

    frag_color = vec4(final_color, 1.0);
}
