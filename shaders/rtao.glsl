#version 460 core

#include "luma.glsl"
#include "types.glsl"
#include "random.glsl"
#include "utils/visbuffer_utils.glsl"
#include "utils/ray_utils.glsl"

#define NORMAL_BIAS 0.001
#define RTAO_N_SAMPLES 4
#define RTAO_HISTORY_WEIGHT 0.9
#define RTAO_DEPTH_REJECT_THRESHOLD 0.002

layout(binding = 0, set = 1) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    FrameDataBuffer frame_data;
    uint visbuffer;
    uint rtao_image;
    uint prev_rtao_image;
    uint prev_depth_image;
    uint velocity_image;
    IndexBuffer index_buffer;
    VertexBuffer vertex_buffer;
    DrawDataBuffer draw_data_buffer;
    NormalBuffer normal_buffer;
} push;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 ao_size = imageSize(R8_UNI(push.rtao_image));
    if (any(greaterThanEqual(coord, ao_size))) return;

    // rtao_image is quarter-res (or half-res): sample the full-res visbuffer at this texel's center
    ivec2 vis_size = imageSize(U32(push.visbuffer));
    ivec2 vis_coord = ivec2((vec2(coord) + 0.5) * vec2(vis_size) / vec2(ao_size));

    FrameData frame_data = push.frame_data.data;
    VisbufferHit hit = decode_visbuffer_hit(
            push.visbuffer, vis_coord, vis_size,
            push.draw_data_buffer, push.index_buffer, push.vertex_buffer,
            frame_data.inv_proj_view, frame_data.camera_position
        );

    if (!hit.valid) {
        imageStore(R8_UNI(push.rtao_image), coord, vec4(1.0));
        return;
    }

    vec3 normal = interpolate_normal(push.normal_buffer, hit.indices, hit.bary, hit.draw.transform);

    // cosine-weighted samples are rotationally symmetric, so any basis around normal works
    mat3 onb = build_onb(normal);
    vec3 ray_origin = hit.world_pos + normal * NORMAL_BIAS;

    uint ao_seed = hash_uint3(floatBitsToUint(hit.world_pos)) ^ pcg_hash(frame_data.frame_idx);
    float acc = 0.0;
    for (int i = 0; i < RTAO_N_SAMPLES; i++) {
        uint sample_seed = ao_seed ^ (uint(i) * 0x9E3779B9u);
        vec3 local_ray_dir = sample_cosine_weighted_hemisphere(sample_seed);
        vec3 rtao_ray_dir = onb * local_ray_dir;

        acc += float(!trace_occluded(tlas, ray_origin, rtao_ray_dir, frame_data.rtao_radius));
    }
    float ao = acc / float(RTAO_N_SAMPLES);
    ao = pow(ao, frame_data.rtao_pow);

    // reproject last frame's resolved AO with the motion vector
    vec2 velocity = imageLoad(RG16F(push.velocity_image), vis_coord).rg;
    vec2 ao_uv = (vec2(coord) + 0.5) / vec2(ao_size);
    vec2 history_uv = ao_uv - velocity;

    // skip history that lands off-screen. also compare this point's depth last frame with
    // what the depth buffer actually held there: if they differ, the history came from a
    // different surface (disocclusion) and blending it in would smear stale AO
    if (all(greaterThanEqual(history_uv, vec2(0.0))) && all(lessThan(history_uv, vec2(1.0)))) {
        vec4 predicted_prev_clip = frame_data.prev_proj_view * vec4(hit.world_pos, 1.0);
        float predicted_prev_depth = predicted_prev_clip.z / predicted_prev_clip.w;
        float actual_prev_depth = texture(TEX_UNI(push.prev_depth_image, frame_data.texture_sampler), history_uv).r;

        if (abs(predicted_prev_depth - actual_prev_depth) < RTAO_DEPTH_REJECT_THRESHOLD) {
            float history_ao = texture(TEX_UNI(push.prev_rtao_image, frame_data.texture_sampler), history_uv).r;
            ao = mix(ao, history_ao, RTAO_HISTORY_WEIGHT);
        }
    }

    imageStore(R8_UNI(push.rtao_image), coord, vec4(ao));
}
