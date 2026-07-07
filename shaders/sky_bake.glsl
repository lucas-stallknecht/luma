#version 460 core

#include "luma.glsl"
#include "types.glsl"

layout(push_constant) uniform PushConstants {
    FrameDataBuffer frame_data;
    uint cubemap_image;
} push;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// procedural atmosphere + clouds, ported from https://github.com/wwwtyro/glsl-atmosphere

const float SKY_BR = 0.0025;
const float SKY_BM = 0.0003;
const float SKY_G = 0.9800;
const vec3 SKY_NITROGEN = vec3(0.650, 0.570, 0.475);
const vec3 SKY_KR = SKY_BR / pow(SKY_NITROGEN, vec3(4.0));
const vec3 SKY_KM = SKY_BM / pow(SKY_NITROGEN, vec3(0.84));

const mat3 SKY_FBM_M = mat3(0.0, 1.60, 1.20, -1.6, 0.72, -0.96, -1.2, -0.96, 1.28);

float sky_hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

float sky_noise(vec3 x) {
    vec3 f = fract(x);
    float n = dot(floor(x), vec3(1.0, 157.0, 113.0));
    return mix(mix(mix(sky_hash(n + 0.0), sky_hash(n + 1.0), f.x),
            mix(sky_hash(n + 157.0), sky_hash(n + 158.0), f.x), f.y),
        mix(mix(sky_hash(n + 113.0), sky_hash(n + 114.0), f.x),
            mix(sky_hash(n + 270.0), sky_hash(n + 271.0), f.x), f.y), f.z);
}

float sky_fbm(vec3 p) {
    float f = 0.0;
    f += sky_noise(p) / 2.0;
    p = SKY_FBM_M * p * 1.1;
    f += sky_noise(p) / 4.0;
    p = SKY_FBM_M * p * 1.2;
    f += sky_noise(p) / 6.0;
    p = SKY_FBM_M * p * 1.3;
    f += sky_noise(p) / 12.0;
    p = SKY_FBM_M * p * 1.4;
    f += sky_noise(p) / 24.0;
    return f;
}

// ray_dir/sun_dir must be normalized
// cirrus/cumulus in [0, 1] control cloud coverage
vec3 sky_color(vec3 ray_dir, vec3 sun_dir, float time, float cirrus, float cumulus, float cloud_noise_scale, float cloud_noise_speed) {
    // there's no modeled ground, so bend rays below the horizon back up to keep the
    // scattering math (which divides by pos.y) stable, then darken the result after
    float horizon = ray_dir.y;
    vec3 pos = ray_dir;
    pos.y = max(pos.y, 0.02);

    float mu = dot(pos, sun_dir);
    float rayleigh = 3.0 / (8.0 * 3.14159265) * (1.0 + mu * mu);
    vec3 mie = (SKY_KR + SKY_KM * (1.0 - SKY_G * SKY_G) / (2.0 + SKY_G * SKY_G) / pow(1.0 + SKY_G * SKY_G - 2.0 * SKY_G * mu, 1.5)) / (SKY_BR + SKY_BM);

    vec3 day_extinction = exp(-exp(-((pos.y + sun_dir.y * 4.0) * (exp(-pos.y * 16.0) + 0.1) / 80.0) / SKY_BR) * (exp(-pos.y * 16.0) + 0.1) * SKY_KR / SKY_BR) * exp(-pos.y * exp(-pos.y * 8.0) * 4.0) * exp(-pos.y * 2.0) * 4.0;
    vec3 night_extinction = vec3(1.0 - exp(sun_dir.y)) * 0.2;
    vec3 extinction = mix(day_extinction, night_extinction, clamp(-sun_dir.y * 0.2 + 0.5, 0.0, 1.0));
    vec3 color = rayleigh * mie * extinction;

    // cirrus clouds
    float cirrus_density = smoothstep(1.0 - cirrus, 1.0, sky_fbm(pos / pos.y * 2.0 * cloud_noise_scale + time * 0.05 * cloud_noise_speed)) * 0.3;
    color = mix(color, extinction * 4.0, cirrus_density * pos.y);

    // cumulus clouds
    for (int i = 0; i < 3; i++) {
        float density = smoothstep(1.0 - cumulus, 1.0, sky_fbm((0.7 + float(i) * 0.01) * pos / pos.y * cloud_noise_scale + time * 0.3 * cloud_noise_speed));
        color = mix(color, extinction * density * 5.0, min(density, 1.0) * pos.y);
    }

    color += sky_noise(pos * 1000.0) * 0.01;

    // fade to a dim ground tone below the horizon
    vec3 ground = extinction * 0.05;
    color = mix(ground, color, smoothstep(-0.1, 0.02, horizon));

    return max(color, 0.0);
}

// matches the Vulkan cube face order/orientation (+X, -X, +Y, -Y, +Z, -Z)
vec3 cube_face_direction(uint face, vec2 uv) {
    switch (face) {
        case 0:
        return normalize(vec3(1.0, -uv.y, -uv.x));
        case 1:
        return normalize(vec3(-1.0, -uv.y, uv.x));
        case 2:
        return normalize(vec3(uv.x, 1.0, uv.y));
        case 3:
        return normalize(vec3(uv.x, -1.0, -uv.y));
        case 4:
        return normalize(vec3(uv.x, -uv.y, 1.0));
        default:
        return normalize(vec3(-uv.x, -uv.y, -1.0));
    }
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    uint face = gl_GlobalInvocationID.z;

    ivec2 size = imageSize(F32_ARRAY_UNI(push.cubemap_image)).xy;
    if (any(greaterThanEqual(coord, size))) return;

    vec2 uv = (vec2(coord) + 0.5) / vec2(size) * 2.0 - 1.0;
    vec3 ray_dir = cube_face_direction(face, uv);

    FrameData frame_data = push.frame_data.data;
    vec3 color = sky_color(
            ray_dir, frame_data.light_dir, frame_data.time,
            frame_data.cirrus, frame_data.cumulus, frame_data.cloud_noise_scale, frame_data.cloud_noise_speed
        );

    imageStore(F32_ARRAY_UNI(push.cubemap_image), ivec3(coord, int(face)), vec4(color, 1.0));
}
