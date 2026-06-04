#version 460 core

layout(push_constant) uniform push_constant
{
    mat4 proj_view_matrix;
} push;

void main() {
    vec2 positions[3] = vec2[](
            vec2(0.0, 0.5),
            vec2(-0.5, -0.5),
            vec2(0.5, -0.5)
        );

    gl_Position = push.proj_view_matrix * vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
