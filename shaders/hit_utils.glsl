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

vec3 sample_albedo(Material material, uint texture_sampler, vec2 uv, vec2 uv_ddx, vec2 uv_ddy) {
    vec3 albedo = material.base_color;
    if (material.base_color_tex >= 0) {
        albedo *= textureGrad(TEX(material.base_color_tex, texture_sampler), uv, uv_ddx, uv_ddy).rgb;
    }
    return albedo;
}

// Duff et al., "Building an Orthonormal Basis, Revisited"
mat3 build_onb(vec3 n) {
    float sign = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;
    vec3 t = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    vec3 bt = vec3(b, sign + n.y * n.y * a, -n.y);
    return mat3(t, bt, n);
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
