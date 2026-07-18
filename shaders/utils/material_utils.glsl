#ifndef MATERIAL_UTILS_GLSL_INCLUDED
#define MATERIAL_UTILS_GLSL_INCLUDED

#include "../luma.glsl"
#include "../types.glsl"

vec3 sample_albedo(Material material, uint texture_sampler, vec2 uv, vec2 uv_ddx, vec2 uv_ddy) {
    vec3 albedo = material.base_color;
    if (material.base_color_tex >= 0) {
        albedo *= textureGrad(TEX(material.base_color_tex, texture_sampler), uv, uv_ddx, uv_ddy).rgb;
    }
    return albedo;
}

#endif
