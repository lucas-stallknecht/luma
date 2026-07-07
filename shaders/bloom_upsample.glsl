#version 460 core

#include "luma.glsl"

layout(push_constant) uniform PushConstants {
    uint src_texture;
    uint src_sampler;
    float src_lod;
    float dst_lod;
    uint dst_image;
    uint dst_width;
    uint dst_height;
    float filter_radius;
} push;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (coord.x >= int(push.dst_width) || coord.y >= int(push.dst_height)) return;

    vec2 uv = (vec2(coord) + 0.5) / vec2(push.dst_width, push.dst_height);
    #define SRC TEX(push.src_texture, push.src_sampler)

    float x = push.filter_radius;
    float y = push.filter_radius;

    vec3 a = textureLod(SRC, vec2(uv.x - x, uv.y + y), push.src_lod).rgb;
    vec3 b = textureLod(SRC, vec2(uv.x, uv.y + y), push.src_lod).rgb;
    vec3 c = textureLod(SRC, vec2(uv.x + x, uv.y + y), push.src_lod).rgb;

    vec3 d = textureLod(SRC, vec2(uv.x - x, uv.y), push.src_lod).rgb;
    vec3 e = textureLod(SRC, vec2(uv.x, uv.y), push.src_lod).rgb;
    vec3 f = textureLod(SRC, vec2(uv.x + x, uv.y), push.src_lod).rgb;

    vec3 g = textureLod(SRC, vec2(uv.x - x, uv.y - y), push.src_lod).rgb;
    vec3 h = textureLod(SRC, vec2(uv.x, uv.y - y), push.src_lod).rgb;
    vec3 i = textureLod(SRC, vec2(uv.x + x, uv.y - y), push.src_lod).rgb;

    vec3 upsampled = e * 4.0;
    upsampled += (b + d + f + h) * 2.0;
    upsampled += (a + c + g + i);
    upsampled *= 1.0 / 16.0;

    vec3 base = textureLod(SRC, uv, push.dst_lod).rgb;
    imageStore(F32(push.dst_image), coord, vec4(base + upsampled, 1.0));
}
