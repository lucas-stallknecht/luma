#ifndef HIT_UTILS_GLSL_INCLUDED
#define HIT_UTILS_GLSL_INCLUDED

#extension GL_EXT_ray_query : require

#include "luma.glsl"
#include "types.glsl"

uvec3 fetch_triangle_indices(IndexBuffer index_buffer, uint triangle_idx) {
    return uvec3(
        index_buffer.indices[triangle_idx * 3 + 0],
        index_buffer.indices[triangle_idx * 3 + 1],
        index_buffer.indices[triangle_idx * 3 + 2]
    );
}

vec2 interpolate_uv(UvBuffer uv_buffer, uvec3 idx, vec3 bary) {
    vec2 uv0 = uv_buffer.uvs[idx.x];
    vec2 uv1 = uv_buffer.uvs[idx.y];
    vec2 uv2 = uv_buffer.uvs[idx.z];
    vec2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
    uv.y = 1.0 - uv.y;
    return uv;
}

vec3 sample_albedo(Material material, uint texture_sampler, vec2 uv, vec2 uv_ddx, vec2 uv_ddy) {
    vec3 albedo = material.base_color;
    if (material.base_color_tex >= 0) {
        albedo *= textureGrad(TEX(material.base_color_tex, texture_sampler), uv, uv_ddx, uv_ddy).rgb;
    }
    return albedo;
}

bool trace_occluded(accelerationStructureEXT tlas, vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query, tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFFu, origin, 0.001, dir, t_max
    );
    rayQueryProceedEXT(ray_query);
    return rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

#endif
