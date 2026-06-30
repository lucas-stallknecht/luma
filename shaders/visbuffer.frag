#version 460 core
#include "luma.glsl"
#include "types.glsl"

layout(push_constant) uniform PushConstants
{
    mat4 proj_view_matrix;
    VertexBuffer vertex_buffer;
    DrawDataBuffer draw_data_buffer;
    UvBuffer uv_buffer;
    MaterialBuffer material_buffer;
    uint texture_sampler;
} push;

layout(location = 0) in flat uint triangle_base;
layout(location = 1) in flat uint draw_id;
layout(location = 2) in vec2 uv;
layout(location = 3) in flat uint material_idx;
layout(location = 0) out uvec4 frag_color;

void main() {
    uint id = triangle_base + gl_PrimitiveID + 1;
    Material material = push.material_buffer.materials[material_idx];

    if (material.base_color_tex >= 0) {
        float alpha = texture(TEX(material.base_color_tex, push.texture_sampler), uv).a;
        if (alpha < 0.5) {
            discard;
        }
    }

    frag_color = uvec4(id, draw_id, 0u, 0u);
}
