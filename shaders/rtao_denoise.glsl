#version 460 core

#include "luma.glsl"
#include "types.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    FrameDataBuffer frame_data;
    uint ao_input;
    uint ao_output;
    uint depth_image;
} push;

#define RADIUS 2
#define SPATIAL_SIGMA 2.0
#define DEPTH_SIGMA 0.5

vec3 world_from_depth(vec2 uv, float depth, mat4 inv_proj_view) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 world = inv_proj_view * ndc;
    return world.xyz / world.w;
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(R8_UNI(push.ao_input));
    if (any(greaterThanEqual(coord, size))) return;

    FrameData frame_data = push.frame_data.data;
    uint samp = frame_data.texture_sampler;

    vec2 center_uv = (vec2(coord) + 0.5) / vec2(size);
    float center_depth = texture(TEX_UNI(push.depth_image, samp), center_uv).r;
    float center_ao = imageLoad(R8_UNI(push.ao_input), coord).r;

    if (center_depth >= 1.0) {
        imageStore(R8_UNI(push.ao_output), coord, vec4(center_ao));
        return;
    }

    vec3 center_pos = world_from_depth(center_uv, center_depth, frame_data.inv_proj_view);

    float sum = 0.0;
    float weight_sum = 0.0;
    for (int y = -RADIUS; y <= RADIUS; y++) {
        for (int x = -RADIUS; x <= RADIUS; x++) {
            ivec2 tap = clamp(coord + ivec2(x, y), ivec2(0), size - 1);
            vec2 tap_uv = (vec2(tap) + 0.5) / vec2(size);

            float tap_depth = texture(TEX_UNI(push.depth_image, samp), tap_uv).r;
            if (tap_depth >= 1.0) continue;
            vec3 tap_pos = world_from_depth(tap_uv, tap_depth, frame_data.inv_proj_view);

            float w_spatial = exp(-float(x * x + y * y) / (2.0 * SPATIAL_SIGMA * SPATIAL_SIGMA));
            float w_depth = exp(-distance(center_pos, tap_pos) / DEPTH_SIGMA);
            float w = w_spatial * w_depth;

            sum += imageLoad(R8_UNI(push.ao_input), tap).r * w;
            weight_sum += w;
        }
    }

    imageStore(R8_UNI(push.ao_output), coord, vec4(sum / max(weight_sum, 1e-5)));
}
