#version 460 core

#include "luma.glsl"

#define TAA_HISTORY_WEIGHT 0.4

layout(push_constant) uniform PushConstants {
    uint draw_image;
    uint prev_taa_image;
    uint sampler_idx;
    uint velocity_image;
    uint taa_image;
} push;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(F32(push.draw_image));
    if (any(greaterThanEqual(coord, size))) return;

    vec3 center = imageLoad(F32(push.draw_image), coord).rgb;

    // colour range of the 3x3 neighborhood. clamping the history into this range keeps
    // stale history close to what's on screen now, which is what stops ghosting
    vec3 box_min = center;
    vec3 box_max = center;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 tap = clamp(coord + ivec2(dx, dy), ivec2(0), size - 1);
            vec3 c = imageLoad(F32(push.draw_image), tap).rgb;
            box_min = min(box_min, c);
            box_max = max(box_max, c);
        }
    }

    vec2 velocity = imageLoad(RG16F(push.velocity_image), coord).rg;
    vec2 uv = (vec2(coord) + 0.5) / vec2(size);
    vec2 history_uv = uv - velocity;

    // fall back to the current frame where there's no history to reproject onto
    float feedback = TAA_HISTORY_WEIGHT;
    if (any(lessThan(history_uv, vec2(0.0))) || any(greaterThanEqual(history_uv, vec2(1.0)))) {
        feedback = 0.0;
    }

    vec3 history = texture(TEX(push.prev_taa_image, push.sampler_idx), history_uv).rgb;
    history = clamp(history, box_min, box_max);

    vec3 result = mix(center, history, feedback);
    imageStore(F32(push.taa_image), coord, vec4(result, 1.0));
}
