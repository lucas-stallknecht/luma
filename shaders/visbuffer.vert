#version 460 core
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430) readonly buffer VertexBuffer {
    float positions[];
};

layout(push_constant) uniform PushConstants
{
    mat4 model_matrix;
    mat4 proj_view_matrix;
    VertexBuffer vertex_buffer;
} push;

void main() {
    vec3 pos = vec3(
            push.vertex_buffer.positions[gl_VertexIndex * 3 + 0],
            push.vertex_buffer.positions[gl_VertexIndex * 3 + 1],
            push.vertex_buffer.positions[gl_VertexIndex * 3 + 2]
        );
    gl_Position = push.proj_view_matrix * push.model_matrix * vec4(pos, 1.0);
}
