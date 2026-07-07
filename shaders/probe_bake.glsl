#version 460 core

#extension GL_EXT_ray_query : require

#include "luma.glsl"
#include "types.glsl"
#include "random.glsl"
#include "probe_global.glsl"
#include "hit_utils.glsl"

#define WORKGROUP_SIZE 64
#define SAMPLES_PER_PROBE 256
#define SAMPLES_PER_THREAD (SAMPLES_PER_PROBE / WORKGROUP_SIZE)
#define NORMAL_BIAS 0.001
#define HYSTERESIS 0.9

layout(binding = 0, set = 1) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    FrameDataBuffer frame_data;
    IndexBuffer index_buffer;
    NormalBuffer normal_buffer;
    UvBuffer uv_buffer;
    DrawDataBuffer draw_data_buffer;
    MaterialBuffer material_buffer;
    ProbePositionBuffer probe_position_buffer;
    ProbeSHBuffer probe_sh_buffer;
} push;

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// per-thread SH partials, reduced across the workgroup
shared vec3 s_coeff0[WORKGROUP_SIZE];
shared vec3 s_coeff1[WORKGROUP_SIZE];
shared vec3 s_coeff2[WORKGROUP_SIZE];
shared vec3 s_coeff3[WORKGROUP_SIZE];

vec3 trace(vec3 origin, vec3 dir, FrameData frame_data) {
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(ray_query, tlas, gl_RayFlagsOpaqueEXT, 0xFFu, origin, 0.001, dir, 1000.0);
    rayQueryProceedEXT(ray_query);

    if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return texture(TEXCUBE_UNI(frame_data.sky_cubemap, frame_data.texture_sampler), dir).rgb;
    }

    uint draw_idx = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true));
    uint primitive_idx = uint(rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true));
    DrawData draw = push.draw_data_buffer.draw_data[draw_idx];
    Material material = push.material_buffer.materials[draw.material_idx];

    uint triangle_idx = draw.triangle_base + primitive_idx;
    uvec3 tri = fetch_triangle_indices(push.index_buffer, triangle_idx);

    vec2 hit_bary = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);
    vec3 bary = vec3(1.0 - hit_bary.x - hit_bary.y, hit_bary.x, hit_bary.y);

    vec2 uv = interpolate_uv(push.uv_buffer, tri, bary);
    vec3 albedo = sample_albedo(material, frame_data.texture_sampler, uv, vec2(0.0), vec2(0.0));
    // push bounce albedo toward white to fake lost bounces
    albedo = pow(albedo, vec3(1.0 / frame_data.albedo_boost));

    vec3 n0 = push.normal_buffer.normals[tri.x];
    vec3 n1 = push.normal_buffer.normals[tri.y];
    vec3 n2 = push.normal_buffer.normals[tri.z];
    vec3 normal = normalize(mat3(draw.transform) * (bary.x * n0 + bary.y * n1 + bary.z * n2));
    // face the ray, else the bias pushes hit_pos into the surface
    if (dot(normal, dir) > 0.0) {
        normal = -normal;
    }

    float hit_t = rayQueryGetIntersectionTEXT(ray_query, true);
    vec3 hit_pos = origin + dir * hit_t + normal * NORMAL_BIAS;

    vec3 radiance = vec3(0.0);

    // shadowed direct sun
    float n_dot_l = dot(normal, frame_data.light_dir);
    if (n_dot_l > 0.0 && !trace_occluded(tlas, hit_pos, frame_data.light_dir, 1000.0)) {
        radiance += albedo * (n_dot_l / PI)
                * frame_data.light_color * frame_data.light_intensity;
    }

    // indirect from last bake's field, it goes one deeper bounce per bake
    vec3 indirect_irradiance = sample_probe_irradiance(
            push.probe_sh_buffer, push.probe_position_buffer, hit_pos, normal, -dir,
            frame_data.grid_min, frame_data.grid_spacing, frame_data.probe_counts
        );
    radiance += albedo / PI * indirect_irradiance;

    return radiance;
}

void main() {
    // one workgroup per probe; dispatch is (1, probe_count, 1)
    uint probe_idx = gl_WorkGroupID.y;
    uint lid = gl_LocalInvocationID.x;
    FrameData frame_data = push.frame_data.data;

    vec3 probe_pos = push.probe_position_buffer.positions[probe_idx];

    // one rotation per probe, shared by the whole workgroup (single point set).
    // fold frame_data.frame_idx into the seed to refine across frames.
    uint seed = hash_uint3(uvec3(probe_idx, 0u, 0x9E3779B9u));
    mat3 rotation = random_rotation(seed);

    // strided directions: lid, lid + 64, ... accumulate in locals, shared isn't zeroed
    vec3 coeff0 = vec3(0.0);
    vec3 coeff1 = vec3(0.0);
    vec3 coeff2 = vec3(0.0);
    vec3 coeff3 = vec3(0.0);
    for (uint s = 0u; s < uint(SAMPLES_PER_THREAD); s++) {
        uint i = lid + s * uint(WORKGROUP_SIZE);
        vec3 dir = rotation * fibonacci_sphere(i, uint(SAMPLES_PER_PROBE));
        vec3 radiance = trace(probe_pos, dir, frame_data);

        vec4 basis = sh_basis(dir);
        coeff0 += radiance * basis.x;
        coeff1 += radiance * basis.y;
        coeff2 += radiance * basis.z;
        coeff3 += radiance * basis.w;
    }

    s_coeff0[lid] = coeff0;
    s_coeff1[lid] = coeff1;
    s_coeff2[lid] = coeff2;
    s_coeff3[lid] = coeff3;
    barrier();

    // reduce partials down to lid 0
    for (uint stride = uint(WORKGROUP_SIZE) / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            s_coeff0[lid] += s_coeff0[lid + stride];
            s_coeff1[lid] += s_coeff1[lid + stride];
            s_coeff2[lid] += s_coeff2[lid + stride];
            s_coeff3[lid] += s_coeff3[lid + stride];
        }
        barrier();
    }

    if (lid == 0u) {
        // MC normalization, uniform pdf = 1 / (4*PI)
        float norm = 4.0 * PI / float(SAMPLES_PER_PROBE);

        // blend into the stored field; trace() reads it back, so bakes converge to multi-bounce
        ProbeSH prev = push.probe_sh_buffer.probes[probe_idx];
        ProbeSH result;
        result.coeffs[0] = mix(s_coeff0[0] * norm, prev.coeffs[0], HYSTERESIS);
        result.coeffs[1] = mix(s_coeff1[0] * norm, prev.coeffs[1], HYSTERESIS);
        result.coeffs[2] = mix(s_coeff2[0] * norm, prev.coeffs[2], HYSTERESIS);
        result.coeffs[3] = mix(s_coeff3[0] * norm, prev.coeffs[3], HYSTERESIS);
        push.probe_sh_buffer.probes[probe_idx] = result;
    }
}
