#version 460 core

#extension GL_EXT_ray_query : require

#include "luma.glsl"
#include "types.glsl"
#include "random.glsl"
#include "brdf.glsl"
#include "hit_utils.glsl"
#include "probe_global.glsl"

#define NORMAL_BIAS 0.001
#define SSAO_N_SAMPLES 4

#define DEBUG_VIEW_NONE 0
#define DEBUG_VIEW_ALBEDO 1
#define DEBUG_VIEW_NORMAL 2
#define DEBUG_VIEW_UV 3
#define DEBUG_VIEW_METALLIC 4
#define DEBUG_VIEW_ROUGHNESS 5
#define DEBUG_VIEW_AO 6
#define DEBUG_VIEW_SHADOW 7
#define DEBUG_VIEW_GI 8
#define DEBUG_VIEW_VIS 9

#define DEBUG_VIEW DEBUG_VIEW_NONE

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
    ProbeSHBuffer probe_sh_buffer;
    ProbePositionBuffer probe_position_buffer;
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

    FrameData frame_data = push.frame_data.data;

    if (triangle_id == 0) {
        vec2 ndc = (vec2(coord) + 0.5) / vec2(size) * 2.0 - 1.0;
        vec4 far = frame_data.inv_proj_view * vec4(ndc, 1.0, 1.0);
        far /= far.w;
        vec3 ray_dir = normalize(far.xyz - frame_data.camera_position);

        vec3 sky = texture(TEXCUBE_UNI(frame_data.sky_cubemap, frame_data.texture_sampler), ray_dir).rgb;
        imageStore(F32_UNI(push.draw_image), coord, vec4(sky, 1.0));
        return;
    }

    DrawData draw = push.draw_data_buffer.draw_data[pix.g];
    Material material = push.material_buffer.materials[draw.material_idx];

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

    vec3 bary = pixel_bary(coord, size, inv_proj_view, camera_position, p0w, p1w, p2w);
    vec3 bary_x1 = pixel_bary(coord + ivec2(1, 0), size, inv_proj_view, camera_position, p0w, p1w, p2w);
    vec3 bary_y1 = pixel_bary(coord + ivec2(0, 1), size, inv_proj_view, camera_position, p0w, p1w, p2w);

    vec3 bary_ddx = bary_x1 - bary;
    vec3 bary_ddy = bary_y1 - bary;

    vec2 uv0 = push.uv_buffer.uvs[tri.x];
    vec2 uv1 = push.uv_buffer.uvs[tri.y];
    vec2 uv2 = push.uv_buffer.uvs[tri.z];

    vec2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
    uv.y = 1.0 - uv.y;

    // derivative of (1 - uv.y) is -(d uv.y), so negate the y component
    vec2 uv_ddx = bary_ddx.x * uv0 + bary_ddx.y * uv1 + bary_ddx.z * uv2;
    vec2 uv_ddy = bary_ddy.x * uv0 + bary_ddy.y * uv1 + bary_ddy.z * uv2;
    uv_ddx.y = -uv_ddx.y;
    uv_ddy.y = -uv_ddy.y;

    vec3 n0 = push.normal_buffer.normals[tri.x];
    vec3 n1 = push.normal_buffer.normals[tri.y];
    vec3 n2 = push.normal_buffer.normals[tri.z];
    // world normal, transformed up front it can serve both normal mapping and SSAO
    vec3 normal = normalize(mat3(draw.transform) * (bary.x * n0 + bary.y * n1 + bary.z * n2));

    vec4 t0 = push.tangent_buffer.tangents[tri.x];
    vec4 t1 = push.tangent_buffer.tangents[tri.y];
    vec4 t2 = push.tangent_buffer.tangents[tri.z];
    vec3 tangent = normalize(mat3(draw.transform) * (bary.x * t0.xyz + bary.y * t1.xyz + bary.z * t2.xyz));
    tangent = normalize(tangent - dot(tangent, normal) * normal);
    vec3 bitangent = cross(normal, tangent) * t0.w;
    mat3 TBN = mat3(tangent, bitangent, normal);

    Surface surface;
    surface.albedo = sample_albedo(material, frame_data.texture_sampler, uv, uv_ddx, uv_ddy);
    surface.normal = normal;
    surface.roughness = 0.5;
    surface.metallic = 0.0;

    if (material.normal_tex >= 0) {
        vec3 tex_normal = textureGrad(TEX(material.normal_tex, frame_data.texture_sampler), uv, uv_ddx, uv_ddy).rgb;
        tex_normal.xy = tex_normal.xy * 2.0 - 1.0;
        // BC5/ATI2-compressed normal maps only store xy, reconstruct z instead of trusting the blue channel
        // this fixed Bistro
        tex_normal.z = sqrt(clamp(1.0 - dot(tex_normal.xy, tex_normal.xy), 0.0, 1.0));
        surface.normal = normalize(TBN * tex_normal);
    }

    if (material.metallic_roughness_tex >= 0) {
        vec4 tex_value = textureGrad(TEX(material.metallic_roughness_tex, frame_data.texture_sampler), uv, uv_ddx, uv_ddy);
        surface.roughness = tex_value.g;
        surface.metallic = tex_value.b;
    }

    // directional shadow mapping
    vec3 world_pos = bary.x * p0w + bary.y * p1w + bary.z * p2w;
    vec3 ray_origin = world_pos + surface.normal * NORMAL_BIAS;
    float shadow = float(!trace_occluded(tlas, ray_origin, frame_data.light_dir, 1000.0));

    // ambient occlusion
    uint ao_seed = hash_uint3(floatBitsToUint(world_pos));
    float acc = 0.0;
    for (int i = 0; i < SSAO_N_SAMPLES; i++) {
        uint sample_seed = ao_seed ^ (uint(i) * 0x9E3779B9u);
        vec3 local_ray_dir = sample_cosine_weighted_hemisphere(sample_seed);
        vec3 ssao_ray_dir = TBN * local_ray_dir;

        acc += float(!trace_occluded(tlas, ray_origin, ssao_ray_dir, frame_data.ssao_radius));
    }
    float ao = acc / float(SSAO_N_SAMPLES);
    ao = pow(ao, frame_data.ssao_pow);

    // indirect diffuse from the probe grid
    vec3 view_dir = normalize(camera_position - world_pos);
    vec3 indirect_irradiance = sample_probe_irradiance(
            push.probe_sh_buffer, push.probe_position_buffer, world_pos, surface.normal, view_dir,
            frame_data.grid_min, frame_data.grid_spacing, frame_data.probe_counts
        );

    #if DEBUG_VIEW != DEBUG_VIEW_NONE
    vec3 debug_color;
    #if DEBUG_VIEW == DEBUG_VIEW_ALBEDO
    debug_color = surface.albedo;
    #elif DEBUG_VIEW == DEBUG_VIEW_NORMAL
    debug_color = surface.normal * 0.5 + 0.5;
    #elif DEBUG_VIEW == DEBUG_VIEW_UV
    debug_color = vec3(uv, 0.0);
    #elif DEBUG_VIEW == DEBUG_VIEW_METALLIC
    debug_color = vec3(surface.metallic);
    #elif DEBUG_VIEW == DEBUG_VIEW_ROUGHNESS
    debug_color = vec3(surface.roughness);
    #elif DEBUG_VIEW == DEBUG_VIEW_AO
    debug_color = vec3(ao);
    #elif DEBUG_VIEW == DEBUG_VIEW_SHADOW
    debug_color = vec3(shadow);
    #elif DEBUG_VIEW == DEBUG_VIEW_GI
    debug_color = indirect_irradiance;
    #elif DEBUG_VIEW == DEBUG_VIEW_VIS
    debug_color = id_to_color(triangle_id);
    #endif
    imageStore(F32_UNI(push.draw_image), coord, vec4(debug_color, 1.0));
    return;
    #endif

    vec3 indirect_lighting = surface.albedo / PI * indirect_irradiance;
    vec3 direct_lighting = evaluate_BRDF(surface, view_dir, frame_data.light_dir) * frame_data.light_color * frame_data.light_intensity;

    vec3 color = ao * indirect_lighting + shadow * direct_lighting;

    imageStore(F32_UNI(push.draw_image), coord, vec4(color, 1.0));
}
