#ifndef PROBE_GLOBAL_GLSL_INCLUDED
#define PROBE_GLOBAL_GLSL_INCLUDED

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#ifndef PI
#define PI 3.14159265359
#endif

layout(buffer_reference, buffer_reference_align = 4, scalar) readonly buffer ProbePositionBuffer {
    vec3 positions[];
};

struct ProbeSH {
    vec3 coeffs[4];
};
layout(buffer_reference, scalar) buffer ProbeSHBuffer {
    ProbeSH probes[];
};

vec4 sh_basis(vec3 w) {
    return vec4(0.282095f, 0.488603f * w.x, 0.488603f * w.y, 0.488603f * w.z);
}

// radiance SH -> irradiance for a surface facing n (cosine-lobe convolution)
// https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
vec3 sh_irradiance(ProbeSH sh, vec3 n) {
    const float A0 = PI;
    const float A1 = 2.0 * PI / 3.0;

    vec4 basis = sh_basis(n);
    vec3 irradiance = A0 * sh.coeffs[0] * basis.x
            + A1 * (sh.coeffs[1] * basis.y + sh.coeffs[2] * basis.z + sh.coeffs[3] * basis.w);
    return max(irradiance, vec3(0.0));
}

int probe_linear_idx(ivec3 coord, ivec3 counts) {
    return coord.x + coord.y * counts.x + coord.z * counts.x * counts.y;
}

// indirect irradiance at world_pos, trilinear blend of the 8 surrounding probes
// TODO: add visibility, backface and other weight changes
// weights sum to 1 atm, make sure to normalize it when it does not
vec3 sample_probe_irradiance(
    ProbeSHBuffer sh_buffer,
    vec3 world_pos,
    vec3 normal,
    vec3 grid_min,
    vec3 grid_spacing,
    ivec3 probe_counts
) {
    // world pos -> fractional probe index (distance from grid origin, in probe spacings)
    // then take the lower-corner probe and the leftover fraction is the blend to the next
    vec3 grid_coord = (world_pos - grid_min) / grid_spacing;

    ivec3 base_coord = ivec3(floor(grid_coord));
    vec3 frac = grid_coord - vec3(base_coord);

    vec3 irradiance = vec3(0.0);
    float total_weight = 0.0;

    for (int i = 0; i < 8; i++) {
        // bit j of i = lower (0) / upper (1) probe on axis j
        ivec3 corner = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 coord = clamp(base_coord + corner, ivec3(0), probe_counts - 1);

        vec3 axis_weight = mix(1.0 - frac, frac, vec3(corner));
        float weight = axis_weight.x * axis_weight.y * axis_weight.z;

        int idx = probe_linear_idx(coord, probe_counts);
        irradiance += weight * sh_irradiance(sh_buffer.probes[idx], normal);
        total_weight += weight;
    }

    return irradiance / max(total_weight, 1e-4);
}

#endif
