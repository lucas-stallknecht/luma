#version 460 core

layout(location = 0) in flat uint triangle_base;
layout(location = 0) out uvec4 frag_color;

void main() {
    uint id = triangle_base + gl_PrimitiveID + 1;
    frag_color = uvec4(id, 0.0, 0.0, 0.0);
}
