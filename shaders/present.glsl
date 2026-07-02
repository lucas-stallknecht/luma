#version 460 core

#include "luma.glsl"

#ifdef STAGE_VERTEX

const vec2 verts[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );

void vert_main() {
    gl_Position = vec4(verts[gl_VertexIndex], 0.0, 1.0);
}

#endif

#ifdef STAGE_FRAGMENT

layout(push_constant) uniform PushConstants {
    uint draw_image;
} push;

layout(location = 0) out vec4 frag_color;

void frag_main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec4 hdr_color = imageLoad(F32(push.draw_image), coord);
    vec3 final_color = pow(hdr_color.rgb, vec3(1.0 / 2.2));

    frag_color = vec4(final_color, 1.0);
}

#endif
