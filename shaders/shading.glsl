#version 460 core

#extension GL_EXT_ray_query : require

#include "luma.glsl"
#include "types.glsl"
#include "random.glsl"
#include "brdf.glsl"

#define NORMAL_BIAS 0.001
#define SSAO_N_SAMPLES 4

layout(binding = 0, set = 1) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    FrameDataBuffer frame_data;
    uint visbuffer;
    uint draw_image;
    IndexBuffer index_buffer;
    VertexBuffer vertex_buffer;
    DrawDataBuffer draw_data_buffer;
    NormalBuffer normal_buffer;
    TangentBuffer tangent_buffer;
    UvBuffer uv_buffer;
    MaterialBuffer material_buffer;
} push;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

vec3 id_to_color(uint id) {
    uint h = id;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return vec3(h & 255u, (h >> 8) & 255u, (h >> 16) & 255u) / 255.0;
}

// https://en.wikipedia.org/wiki/Barycentric_coordinate_system
vec3 moller_trumbore(vec3 p0, vec3 p1, vec3 p2, vec3 ray_origin, vec3 ray_dir) {
    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;
    vec3 h = cross(ray_dir, e2);
    float inv_det = 1.0 / dot(e1, h);
    vec3 s = ray_origin - p0;
    float u = inv_det * dot(s, h);
    vec3 q = cross(s, e1);
    float v = inv_det * dot(ray_dir, q);
    return vec3(1.0 - u - v, u, v);
}

vec3 pixel_bary(ivec2 coord, ivec2 size, mat4 inv_proj_view, vec3 camera_position, vec3 p0w, vec3 p1w, vec3 p2w) {
    // reconstruct the world-space ray through this pixel's center
    vec2 ndc = (vec2(coord) + 0.5) / vec2(size) * 2.0 - 1.0;
    vec4 far = inv_proj_view * vec4(ndc, 1.0, 1.0);
    far /= far.w;
    vec3 ray_dir = normalize(far.xyz - camera_position);
    return moller_trumbore(p0w, p1w, p2w, camera_position, ray_dir);
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(U32(push.visbuffer));
    if (any(greaterThanEqual(coord, size))) return;

    uvec4 pix = imageLoad(U32(push.visbuffer), coord);
    uint triangle_id = pix.r;

    if (triangle_id == 0) {
        imageStore(F32(push.draw_image), coord, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    DrawData draw = push.draw_data_buffer.draw_data[pix.g];
    Material material = push.material_buffer.materials[draw.material_idx];
    FrameData frame_data = push.frame_data.data;

    uint triangle_idx = triangle_id - 1;
    uint i0 = push.index_buffer.indices[triangle_idx * 3 + 0];
    uint i1 = push.index_buffer.indices[triangle_idx * 3 + 1];
    uint i2 = push.index_buffer.indices[triangle_idx * 3 + 2];

    vec3 p0 = push.vertex_buffer.positions[i0];
    vec3 p1 = push.vertex_buffer.positions[i1];
    vec3 p2 = push.vertex_buffer.positions[i2];

    vec3 p0w = vec3(draw.transform * vec4(p0, 1.0));
    vec3 p1w = vec3(draw.transform * vec4(p1, 1.0));
    vec3 p2w = vec3(draw.transform * vec4(p2, 1.0));

    mat4 inv_proj_view = frame_data.inv_proj_view;
    vec3 camera_position = frame_data.camera_position;

    vec3 bary = pixel_bary(coord, size, inv_proj_view, camera_position, p0w, p1w, p2w);
    vec3 bary_x1 = pixel_bary(coord + ivec2(1, 0), size, inv_proj_view, camera_position, p0w, p1w, p2w);
    vec3 bary_y1 = pixel_bary(coord + ivec2(0, 1), size, inv_proj_view, camera_position, p0w, p1w, p2w);

    vec3 bary_ddx = bary_x1 - bary;
    vec3 bary_ddy = bary_y1 - bary;

    vec2 uv0 = push.uv_buffer.uvs[i0];
    vec2 uv1 = push.uv_buffer.uvs[i1];
    vec2 uv2 = push.uv_buffer.uvs[i2];

    vec2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
    uv.y = 1.0 - uv.y;

    // derivative of (1 - uv.y) is -(d uv.y), so negate the y component
    vec2 uv_ddx = bary_ddx.x * uv0 + bary_ddx.y * uv1 + bary_ddx.z * uv2;
    vec2 uv_ddy = bary_ddy.x * uv0 + bary_ddy.y * uv1 + bary_ddy.z * uv2;
    uv_ddx.y = -uv_ddx.y;
    uv_ddy.y = -uv_ddy.y;

    vec3 n0 = push.normal_buffer.normals[i0];
    vec3 n1 = push.normal_buffer.normals[i1];
    vec3 n2 = push.normal_buffer.normals[i2];
    // world normal, transformed up front it can serve both normal mapping and SSAO
    vec3 normal = normalize(mat3(draw.transform) * (bary.x * n0 + bary.y * n1 + bary.z * n2));

    vec4 t0 = push.tangent_buffer.tangents[i0];
    vec4 t1 = push.tangent_buffer.tangents[i1];
    vec4 t2 = push.tangent_buffer.tangents[i2];
    vec3 tangent = normalize(mat3(draw.transform) * (bary.x * t0.xyz + bary.y * t1.xyz + bary.z * t2.xyz));
    tangent = normalize(tangent - dot(tangent, normal) * normal);
    vec3 bitangent = cross(normal, tangent) * t0.w;
    mat3 TBN = mat3(tangent, bitangent, normal);

    Surface surface;
    surface.albedo = material.base_color;
    surface.normal = normal;
    surface.roughness = 0.5;
    surface.metallic = 0.0;

    if (material.normal_tex >= 0) {
        vec3 tex_normal = textureGrad(TEX(material.normal_tex, frame_data.texture_sampler), uv, uv_ddx, uv_ddy).rgb;
        tex_normal = tex_normal * 2.0 - 1.0;
        surface.normal = normalize(TBN * tex_normal);
    }

    if (material.base_color_tex >= 0) {
        vec4 tex_color = textureGrad(TEX(material.base_color_tex, frame_data.texture_sampler), uv, uv_ddx, uv_ddy);
        surface.albedo *= tex_color.rgb;
    }
    if (material.metallic_roughness_tex >= 0) {
        vec4 tex_value = textureGrad(TEX(material.metallic_roughness_tex, frame_data.texture_sampler), uv, uv_ddx, uv_ddy);
        surface.roughness = tex_value.g;
        surface.metallic = tex_value.b;
    }

    // directional shadow mapping
    vec3 world_pos = bary.x * p0w + bary.y * p1w + bary.z * p2w;
    vec3 ray_origin = world_pos + surface.normal * NORMAL_BIAS;
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query, tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFFu, ray_origin, 0.001, frame_data.light_dir, 1000.0
    );
    rayQueryProceedEXT(ray_query);
    bool occluder_hit = rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
    float shadow = float(!occluder_hit);

    // ambient occlusion
    uint ao_seed = hash_uint3(floatBitsToUint(world_pos));
    float acc = 0.0;
    for (int i = 0; i < SSAO_N_SAMPLES; i++) {
        uint sample_seed = ao_seed ^ (uint(i) * 0x9E3779B9u);
        vec3 local_ray_dir = sample_cosine_weighted_hemisphere(sample_seed);
        vec3 ssao_ray_dir = TBN * local_ray_dir;

        rayQueryEXT ssao_ray_query;
        rayQueryInitializeEXT(
            ssao_ray_query, tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
            0xFFu, ray_origin, 0.001, ssao_ray_dir, frame_data.ssao_radius
        );
        rayQueryProceedEXT(ssao_ray_query);
        bool ssao_occluder_hit = rayQueryGetIntersectionTypeEXT(ssao_ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
        acc += float(!ssao_occluder_hit);
    }
    float ao = acc / float(SSAO_N_SAMPLES);
    ao = pow(ao, frame_data.ssao_pow);

    vec3 view_dir = normalize(camera_position - world_pos);

    vec3 ambient = frame_data.ambient_color * frame_data.ambient_intensity * surface.albedo;
    vec3 direct_lighting = evaluate_BRDF(surface, view_dir, frame_data.light_dir) * frame_data.light_color * frame_data.light_intensity;
    vec3 color = ao * ambient + shadow * direct_lighting;

    imageStore(F32(push.draw_image), coord, vec4(color, 1.0));
}
