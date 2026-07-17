#version 460 core

#extension GL_EXT_ray_query : require

#include "luma.glsl"
#include "types.glsl"
#include "random.glsl"
#include "hit_utils.glsl"

#define NORMAL_BIAS 0.001
#define SSAO_N_SAMPLES 4

layout(binding = 0, set = 1) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    FrameDataBuffer frame_data;
    uint visbuffer;
    uint ssao_image;
    IndexBuffer index_buffer;
    VertexBuffer vertex_buffer;
    DrawDataBuffer draw_data_buffer;
    NormalBuffer normal_buffer;
} push;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 ao_size = imageSize(R8_UNI(push.ssao_image));
    if (any(greaterThanEqual(coord, ao_size))) return;

    // ssao_image is quarter-res (or half-res): sample the full-res visbuffer at this texel's center
    ivec2 vis_size = imageSize(U32(push.visbuffer));
    ivec2 vis_coord = ivec2((vec2(coord) + 0.5) * vec2(vis_size) / vec2(ao_size));

    uvec4 pix = imageLoad(U32(push.visbuffer), vis_coord);
    uint triangle_id = pix.r;

    if (triangle_id == 0) {
        imageStore(R8_UNI(push.ssao_image), coord, vec4(1.0));
        return;
    }

    FrameData frame_data = push.frame_data.data;
    DrawData draw = push.draw_data_buffer.draw_data[pix.g];

    uint triangle_idx = triangle_id - 1;
    uvec3 tri = fetch_triangle_indices(push.index_buffer, triangle_idx);

    vec3 p0 = push.vertex_buffer.positions[tri.x];
    vec3 p1 = push.vertex_buffer.positions[tri.y];
    vec3 p2 = push.vertex_buffer.positions[tri.z];

    vec3 p0w = vec3(draw.transform * vec4(p0, 1.0));
    vec3 p1w = vec3(draw.transform * vec4(p1, 1.0));
    vec3 p2w = vec3(draw.transform * vec4(p2, 1.0));

    mat4 inv_proj_view = frame_data.inv_proj_view;
    vec3 camera_position = frame_data.camera_position;

    vec3 bary = pixel_bary(vis_coord, vis_size, inv_proj_view, camera_position, p0w, p1w, p2w);
    vec3 world_pos = bary.x * p0w + bary.y * p1w + bary.z * p2w;

    vec3 n0 = push.normal_buffer.normals[tri.x];
    vec3 n1 = push.normal_buffer.normals[tri.y];
    vec3 n2 = push.normal_buffer.normals[tri.z];
    vec3 normal = normalize(mat3(draw.transform) * (bary.x * n0 + bary.y * n1 + bary.z * n2));

    // cosine-weighted samples are rotationally symmetric, so any basis around normal works
    mat3 onb = build_onb(normal);
    vec3 ray_origin = world_pos + normal * NORMAL_BIAS;

    uint ao_seed = hash_uint3(floatBitsToUint(world_pos));
    float acc = 0.0;
    for (int i = 0; i < SSAO_N_SAMPLES; i++) {
        uint sample_seed = ao_seed ^ (uint(i) * 0x9E3779B9u);
        vec3 local_ray_dir = sample_cosine_weighted_hemisphere(sample_seed);
        vec3 ssao_ray_dir = onb * local_ray_dir;

        acc += float(!trace_occluded(tlas, ray_origin, ssao_ray_dir, frame_data.ssao_radius));
    }
    float ao = acc / float(SSAO_N_SAMPLES);
    ao = pow(ao, frame_data.ssao_pow);

    imageStore(R8_UNI(push.ssao_image), coord, vec4(ao));
}
