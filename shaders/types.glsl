#ifndef TYPES_GLSL_INCLUDED
#define TYPES_GLSL_INCLUDED

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

layout(buffer_reference, std430) readonly buffer IndexBuffer {
    uint indices[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) readonly buffer VertexBuffer {
    vec3 positions[];
};
layout(buffer_reference, std430) readonly buffer DrawDataBuffer {
    DrawData draw_data[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) readonly buffer NormalBuffer {
    vec3 normals[];
};
layout(buffer_reference, buffer_reference_align = 16) readonly buffer UvBuffer {
    vec2 uvs[];
};
layout(buffer_reference, std430) readonly buffer MaterialBuffer {
    Material materials[];
};

#endif
