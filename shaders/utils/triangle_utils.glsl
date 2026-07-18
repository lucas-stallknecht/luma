#ifndef TRIANGLE_UTILS_GLSL_INCLUDED
#define TRIANGLE_UTILS_GLSL_INCLUDED

#include "../types.glsl"

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

vec2 interpolate_uv_derivative(UvBuffer uv_buffer, uvec3 idx, vec3 bary_d) {
    vec2 uv0 = uv_buffer.uvs[idx.x];
    vec2 uv1 = uv_buffer.uvs[idx.y];
    vec2 uv2 = uv_buffer.uvs[idx.z];
    vec2 d = bary_d.x * uv0 + bary_d.y * uv1 + bary_d.z * uv2;
    d.y = -d.y; // because derivative of (1 - uv.y) is -(d uv.y)
    return d;
}

vec3 interpolate_normal(NormalBuffer normal_buffer, uvec3 idx, vec3 bary, mat4 transform) {
    vec3 n0 = normal_buffer.normals[idx.x];
    vec3 n1 = normal_buffer.normals[idx.y];
    vec3 n2 = normal_buffer.normals[idx.z];
    return normalize(mat3(transform) * (bary.x * n0 + bary.y * n1 + bary.z * n2));
}

#endif
