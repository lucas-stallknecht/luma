#version 460 core
#include "types.glsl"

layout(push_constant) uniform PushConstants
{
    FrameDataBuffer frame_data;
    VertexBuffer vertex_buffer;
    DrawDataBuffer draw_data_buffer;
    UvBuffer uv_buffer;
    MaterialBuffer material_buffer;
} push;

layout(location = 0) out flat uint triangle_base;
layout(location = 1) out flat uint draw_id;
layout(location = 2) out vec2 uv;
layout(location = 3) out flat uint material_idx;

void main() {
    vec3 pos = push.vertex_buffer.positions[gl_VertexIndex];
    DrawData draw = push.draw_data_buffer.draw_data[gl_DrawID];

    gl_Position = push.frame_data.data.proj_view_matrix * draw.transform * vec4(pos, 1.0);
    triangle_base = draw.triangle_base;
    draw_id = uint(gl_DrawID);
    uv = push.uv_buffer.uvs[gl_VertexIndex];
    material_idx = draw.material_idx;
}
