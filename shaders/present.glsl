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
    uint bloom_texture;
    uint bloom_sampler;
    float bloom_intensity;
} push;

layout(location = 0) out vec4 frag_color;

// https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/pbrNeutral.glsl
// Input color is non-negative and resides in the Linear Rec. 709 color space.
// Output color is also Linear Rec. 709, but in the [0, 1] range.
vec3 pbr_neutral_tonemapping(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    const float d = 1. - startCompression;
    float newPeak = 1. - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1. - 1. / (desaturation * (peak - newPeak) + 1.);
    return mix(color, newPeak * vec3(1, 1, 1), g);
}

void frag_main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec4 hdr_color = imageLoad(F32_UNI(push.draw_image), coord);

    vec2 uv = (vec2(coord) + 0.5) / vec2(imageSize(F32_UNI(push.draw_image)));
    vec3 bloom_color = textureLod(TEX_UNI(push.bloom_texture, push.bloom_sampler), uv, 0.0).rgb;
    hdr_color.rgb = mix(hdr_color.rgb, bloom_color, push.bloom_intensity);

    vec3 linear_color = pow(hdr_color.rgb, vec3(1.0 / 2.2));
    vec3 final_color = pbr_neutral_tonemapping(linear_color);

    frag_color = vec4(final_color, 1.0);
}

#endif
