#version 460 core

layout(location = 0) in flat uint triangle_base;
layout(location = 0) out vec4 frag_color;

void main() {
    uint id = triangle_base + gl_PrimitiveID;

    // integer hash (fast, decent distribution)
    id ^= id >> 16;
    id *= 0x7feb352d;
    id ^= id >> 15;
    id *= 0x846ca68b;
    id ^= id >> 16;

    // convert to RGB
    vec3 color = vec3(
            float(id & 255u) / 255.0,
            float((id >> 8) & 255u) / 255.0,
            float((id >> 16) & 255u) / 255.0
        );

    frag_color = vec4(color, 1.0);
}
