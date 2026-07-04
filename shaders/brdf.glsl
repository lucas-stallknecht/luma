#ifndef BRDF_GLSL_INCLUDED
#define BRDF_GLSL_INCLUDED

#ifndef PI
#define PI 3.14159265359
#endif

#define MIN_ROUGHNESS 1e-3
#define MIN_N_DOT_V 1e-5

struct Surface {
    vec3 albedo;
    float roughness;
    vec3 normal;
    float metallic;
};

// Frostbite BRDF
// https://www.ea.com/news/moving-frostbite-to-pb

vec3 f_schlick(vec3 f0, float f90, float cos_theta) {
    return f0 + (f90 - f0) * pow(1.0 - cos_theta, 5.0f);
}

float v_smith_GGX_correlated(float n_dot_l, float n_dot_v, float alpha_G) {
    float alpha_G2 = alpha_G * alpha_G;
    float lambda_GGXV = n_dot_l * sqrt((-n_dot_v * alpha_G2 + n_dot_v) * n_dot_v + alpha_G2);
    float lambda_GGXL = n_dot_v * sqrt((-n_dot_l * alpha_G2 + n_dot_l) * n_dot_l + alpha_G2);

    return 0.5 / (lambda_GGXV + lambda_GGXL);
}

float d_GGX(float n_dot_h, float m) {
    float m2 = m * m;
    float f = (n_dot_h * m2 - n_dot_h) * n_dot_h + 1.0;
    return m2 / (f * f);
}

vec3 evaluate_specular(vec3 f0, float f90, float n_dot_v, float n_dot_l, float n_dot_h, float l_dot_h, float linear_roughness) {
    float roughness = max(linear_roughness * linear_roughness, MIN_ROUGHNESS);

    vec3 f = f_schlick(f0, f90, l_dot_h);
    float vis = v_smith_GGX_correlated(n_dot_l, n_dot_v, roughness);
    float d = d_GGX(n_dot_h, roughness);

    return d * f * vis;
}

vec3 evaluate_disney_diffuse(float n_dot_v, float n_dot_l, float l_dot_h, float linear_roughness) {
    float energy_bias = mix(0.0, 0.5, linear_roughness);
    float energy_factor = mix(1.0, 1.0 / 1.51, linear_roughness);
    float fd90 = energy_bias + 2.0 * l_dot_h * l_dot_h * linear_roughness;
    vec3 f0 = vec3(1.0);
    float light_scatter = f_schlick(f0, fd90, n_dot_l).r;
    float view_scatter = f_schlick(f0, fd90, n_dot_v).r;

    return vec3(light_scatter * view_scatter * energy_factor);
}

vec3 evaluate_BRDF(Surface surface, vec3 v, vec3 l) {
    vec3 n = surface.normal;
    float n_dot_v = abs(dot(n, v)) + MIN_N_DOT_V;
    vec3 h = normalize(v + l);

    float n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    float n_dot_l = clamp(dot(n, l), 0.0, 1.0);
    float l_dot_h = clamp(dot(l, h), 0.0, 1.0);

    vec3 f0 = mix(vec3(0.04), surface.albedo, surface.metallic);
    float f90 = 1.0;

    vec3 fs = evaluate_specular(
            f0,
            f90,
            n_dot_v,
            n_dot_l,
            n_dot_h,
            l_dot_h,
            surface.roughness
        );
    vec3 fd = evaluate_disney_diffuse(
            n_dot_v,
            n_dot_l,
            l_dot_h,
            surface.roughness
        );

    // metals have no diffuse part, only the dielectric part scatters
    vec3 diffuse = fd * surface.albedo * (1.0 - surface.metallic);

    return (diffuse + fs) * n_dot_l / PI;
}

#endif
