#ifndef TYPES_GLSL_INCLUDED
#define TYPES_GLSL_INCLUDED

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct DrawData {
    mat4 transform;
    int material_idx;
    uint triangle_base;
};

struct Material {
    vec3 base_color;
    int base_color_tex;
    int normal_tex;
    int metallic_roughness_tex;
};

struct FrameData {
    mat4 proj_view;
    mat4 inv_proj_view;
    vec3 camera_position;
    uint texture_sampler;
    vec3 light_dir;
    float albedo_boost;
    vec3 light_color;
    float light_intensity;
    vec3 grid_min;
    uint probe_count;
    vec3 grid_spacing;
    uint frame_idx;
    ivec3 probe_counts;
    float ssao_pow;
    float ssao_radius;
    float time;
    float cirrus;
    float cumulus;
    float cloud_noise_scale;
    float cloud_noise_speed;
    uint sky_cubemap;
};

layout(buffer_reference, std430) readonly buffer FrameDataBuffer {
    FrameData data;
};

layout(buffer_reference, std430) readonly buffer IndexBuffer {
    uint indices[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) readonly buffer VertexBuffer {
    vec3 positions[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) readonly buffer NormalBuffer {
    vec3 normals[];
};
layout(buffer_reference, std430) readonly buffer TangentBuffer {
    vec4 tangents[];
};
layout(buffer_reference, buffer_reference_align = 16) readonly buffer UvBuffer {
    vec2 uvs[];
};

layout(buffer_reference, std430) readonly buffer DrawDataBuffer {
    DrawData draw_data[];
};
layout(buffer_reference, std430) readonly buffer MaterialBuffer {
    Material materials[];
};

#endif
