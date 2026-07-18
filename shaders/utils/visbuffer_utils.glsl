#ifndef VISBUFFER_UTILS_GLSL_INCLUDED
#define VISBUFFER_UTILS_GLSL_INCLUDED

#include "../luma.glsl"
#include "../types.glsl"
#include "triangle_utils.glsl"

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

// the fetch + transform + bary reconstruction below, bundled up so it isn't retyped per shader
struct VisbufferHit {
    bool valid;
    uint triangle_id;
    uint draw_id;
    DrawData draw;
    uvec3 indices;
    vec3 p0w, p1w, p2w;
    vec3 bary;
    vec3 world_pos;
};

VisbufferHit decode_visbuffer_hit(
    uint visbuffer_idx, ivec2 coord, ivec2 size,
    DrawDataBuffer draw_data_buffer, IndexBuffer index_buffer, VertexBuffer vertex_buffer,
    mat4 inv_proj_view, vec3 camera_position
) {
    VisbufferHit hit;
    uvec4 pix = imageLoad(U32(visbuffer_idx), coord);
    hit.triangle_id = pix.r;
    hit.valid = hit.triangle_id != 0u;
    if (!hit.valid) return hit;

    hit.draw_id = pix.g;
    hit.draw = draw_data_buffer.draw_data[hit.draw_id];
    hit.indices = fetch_triangle_indices(index_buffer, hit.triangle_id - 1u);

    vec3 p0 = vertex_buffer.positions[hit.indices.x];
    vec3 p1 = vertex_buffer.positions[hit.indices.y];
    vec3 p2 = vertex_buffer.positions[hit.indices.z];
    hit.p0w = vec3(hit.draw.transform * vec4(p0, 1.0));
    hit.p1w = vec3(hit.draw.transform * vec4(p1, 1.0));
    hit.p2w = vec3(hit.draw.transform * vec4(p2, 1.0));

    hit.bary = pixel_bary(coord, size, inv_proj_view, camera_position, hit.p0w, hit.p1w, hit.p2w);
    hit.world_pos = hit.bary.x * hit.p0w + hit.bary.y * hit.p1w + hit.bary.z * hit.p2w;
    return hit;
}

struct BaryDerivatives {
    vec3 ddx, ddy;
};

// feeds textureGrad's mip selection for material texture sampling
BaryDerivatives visbuffer_bary_derivatives(VisbufferHit hit, ivec2 coord, ivec2 size, mat4 inv_proj_view, vec3 camera_position) {
    vec3 bary_x1 = pixel_bary(coord + ivec2(1, 0), size, inv_proj_view, camera_position, hit.p0w, hit.p1w, hit.p2w);
    vec3 bary_y1 = pixel_bary(coord + ivec2(0, 1), size, inv_proj_view, camera_position, hit.p0w, hit.p1w, hit.p2w);
    BaryDerivatives d;
    d.ddx = bary_x1 - hit.bary;
    d.ddy = bary_y1 - hit.bary;
    return d;
}

#endif
