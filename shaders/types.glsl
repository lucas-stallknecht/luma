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

#endif
