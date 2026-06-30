#version 460 core
#extension GL_EXT_buffer_reference : require
#include "types.glsl"

layout(buffer_reference, std430) readonly buffer VertexBuffer {
    float positions[];
};

layout(buffer_reference, std430) readonly buffer DrawDataBuffer {
    DrawData draw_data[];
};

layout(push_constant) uniform PushConstants
{
    mat4 proj_view_matrix;
    VertexBuffer vertex_buffer;
    DrawDataBuffer draw_data_buffer;
} push;

layout(location = 0) out flat uint triangle_base;
layout(location = 1) out flat uint draw_id;

void main() {
    vec3 pos = vec3(
            push.vertex_buffer.positions[gl_VertexIndex * 3 + 0],
            push.vertex_buffer.positions[gl_VertexIndex * 3 + 1],
            push.vertex_buffer.positions[gl_VertexIndex * 3 + 2]
        );
    DrawData draw = push.draw_data_buffer.draw_data[gl_DrawID];
    gl_Position = push.proj_view_matrix * draw.transform * vec4(pos, 1.0);
    triangle_base = draw.triangle_base;
    draw_id = uint(gl_DrawID);
}
