#version 460 core

#include "luma.glsl"

layout(push_constant) uniform PushConstants {
    uint src_texture;
    uint src_sampler;
    float src_lod;
    uint dst_image;
    uint dst_width;
    uint dst_height;
} push;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

float luma(vec3 col) {
    return dot(col, vec3(0.2126f, 0.7152f, 0.0722f));
}

float karis_average(vec3 col) {
    float l = luma(pow(col, vec3(1.0 / 2.2)));
    return 1.0 / (1.0 + l);
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (coord.x >= int(push.dst_width) || coord.y >= int(push.dst_height)) return;

    vec2 uv = (vec2(coord) + 0.5) / vec2(push.dst_width, push.dst_height);
    #define SRC TEX_UNI(push.src_texture, push.src_sampler)

    vec2 texel = 1.0 / vec2(textureSize(SRC, int(push.src_lod)));
    float x = texel.x;
    float y = texel.y;

    vec3 a = textureLod(SRC, vec2(uv.x - 2 * x, uv.y + 2 * y), push.src_lod).rgb;
    vec3 b = textureLod(SRC, vec2(uv.x, uv.y + 2 * y), push.src_lod).rgb;
    vec3 c = textureLod(SRC, vec2(uv.x + 2 * x, uv.y + 2 * y), push.src_lod).rgb;

    vec3 d = textureLod(SRC, vec2(uv.x - 2 * x, uv.y), push.src_lod).rgb;
    vec3 e = textureLod(SRC, vec2(uv.x, uv.y), push.src_lod).rgb;
    vec3 f = textureLod(SRC, vec2(uv.x + 2 * x, uv.y), push.src_lod).rgb;

    vec3 g = textureLod(SRC, vec2(uv.x - 2 * x, uv.y - 2 * y), push.src_lod).rgb;
    vec3 h = textureLod(SRC, vec2(uv.x, uv.y - 2 * y), push.src_lod).rgb;
    vec3 i = textureLod(SRC, vec2(uv.x + 2 * x, uv.y - 2 * y), push.src_lod).rgb;

    vec3 j = textureLod(SRC, vec2(uv.x - x, uv.y + y), push.src_lod).rgb;
    vec3 k = textureLod(SRC, vec2(uv.x + x, uv.y + y), push.src_lod).rgb;
    vec3 l = textureLod(SRC, vec2(uv.x - x, uv.y - y), push.src_lod).rgb;
    vec3 m = textureLod(SRC, vec2(uv.x + x, uv.y - y), push.src_lod).rgb;

    vec3 downsample;
    vec3 groups[5];
    if (int(push.src_lod) == 0) {
        groups[0] = (a + b + d + e) * (0.125 / 4.0);
        groups[1] = (b + c + e + f) * (0.125 / 4.0);
        groups[2] = (d + e + g + h) * (0.125 / 4.0);
        groups[3] = (e + f + h + i) * (0.125 / 4.0);
        groups[4] = (j + k + l + m) * (0.5 / 4.0);
        groups[0] *= karis_average(groups[0]);
        groups[1] *= karis_average(groups[1]);
        groups[2] *= karis_average(groups[2]);
        groups[3] *= karis_average(groups[3]);
        groups[4] *= karis_average(groups[4]);
        downsample = groups[0] + groups[1] + groups[2] + groups[3] + groups[4];
    }
    else {
        downsample = e * 0.125;
        downsample += (a + c + g + i) * 0.03125;
        downsample += (b + d + f + h) * 0.0625;
        downsample += (j + k + l + m) * 0.125;
    }

    imageStore(F32_UNI(push.dst_image), coord, vec4(downsample, 1.0));
}
