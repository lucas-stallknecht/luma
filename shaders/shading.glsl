#version 460 core

#include "luma.glsl"
#include "types.glsl"
#include "random.glsl"
#include "brdf.glsl"
#include "utils/visbuffer_utils.glsl"
#include "utils/material_utils.glsl"
#include "utils/ray_utils.glsl"
#include "probe_global.glsl"

#define NORMAL_BIAS 0.001

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
    uint rtao_image;
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

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(U32(push.visbuffer));
    if (any(greaterThanEqual(coord, size))) return;

    FrameData frame_data = push.frame_data.data;

    VisbufferHit hit = decode_visbuffer_hit(
            push.visbuffer, coord, size,
            push.draw_data_buffer, push.index_buffer, push.vertex_buffer,
            frame_data.inv_proj_view, frame_data.camera_position
        );

    if (!hit.valid) {
        vec2 ndc = (vec2(coord) + 0.5) / vec2(size) * 2.0 - 1.0;
        vec4 far = frame_data.inv_proj_view * vec4(ndc, 1.0, 1.0);
        far /= far.w;
        vec3 ray_dir = normalize(far.xyz - frame_data.camera_position);

        vec3 sky = texture(TEXCUBE_UNI(frame_data.sky_cubemap, frame_data.texture_sampler), ray_dir).rgb;
        imageStore(F32_UNI(push.draw_image), coord, vec4(sky, 1.0));
        return;
    }

    Material material = push.material_buffer.materials[hit.draw.material_idx];

    BaryDerivatives bary_d = visbuffer_bary_derivatives(
            hit, coord, size, frame_data.inv_proj_view, frame_data.camera_position
        );

    vec2 uv = interpolate_uv(push.uv_buffer, hit.indices, hit.bary);
    vec2 uv_ddx = interpolate_uv_derivative(push.uv_buffer, hit.indices, bary_d.ddx);
    vec2 uv_ddy = interpolate_uv_derivative(push.uv_buffer, hit.indices, bary_d.ddy);

    // world normal, transformed up front it can serve both normal mapping and RTAO
    vec3 normal = interpolate_normal(push.normal_buffer, hit.indices, hit.bary, hit.draw.transform);

    vec4 t0 = push.tangent_buffer.tangents[hit.indices.x];
    vec4 t1 = push.tangent_buffer.tangents[hit.indices.y];
    vec4 t2 = push.tangent_buffer.tangents[hit.indices.z];
    vec3 tangent = normalize(mat3(hit.draw.transform) * (hit.bary.x * t0.xyz + hit.bary.y * t1.xyz + hit.bary.z * t2.xyz));
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
    vec3 ray_origin = hit.world_pos + surface.normal * NORMAL_BIAS;
    float shadow = float(!trace_occluded(tlas, ray_origin, frame_data.light_dir, 1000.0));

    // ambient occlusion
    vec2 screen_uv = (vec2(coord) + 0.5) / vec2(size);
    float ao = texture(TEX_UNI(push.rtao_image, frame_data.texture_sampler), screen_uv).r;

    // indirect diffuse from the probe grid
    vec3 view_dir = normalize(frame_data.camera_position - hit.world_pos);
    vec3 indirect_irradiance = sample_probe_irradiance(
            push.probe_sh_buffer, push.probe_position_buffer, hit.world_pos, surface.normal, view_dir,
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
    debug_color = id_to_color(hit.triangle_id);
    #endif
    imageStore(F32_UNI(push.draw_image), coord, vec4(debug_color, 1.0));
    return;
    #endif

    vec3 indirect_lighting = surface.albedo / PI * indirect_irradiance;
    vec3 direct_lighting = evaluate_BRDF(surface, view_dir, frame_data.light_dir) * frame_data.light_color * frame_data.light_intensity;

    vec3 color = ao * indirect_lighting + shadow * direct_lighting;

    imageStore(F32_UNI(push.draw_image), coord, vec4(color, 1.0));
}
