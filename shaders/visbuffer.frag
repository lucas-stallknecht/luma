#version 460 core

layout(location = 0) out vec4 frag_color;

void main() {
    frag_color = vec4(gl_PrimitiveID / 11.0, 0.0, 0.0, 1.0);
}
