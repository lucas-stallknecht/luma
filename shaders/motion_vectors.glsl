#version 460 core

#include "luma.glsl"
#include "types.glsl"
#include "utils/visbuffer_utils.glsl"

layout(push_constant) uniform PushConstants {
    FrameDataBuffer frame_data;
    uint visbuffer;
    uint velocity_image;
    IndexBuffer index_buffer;
    VertexBuffer vertex_buffer;
    DrawDataBuffer draw_data_buffer;
    vec2 jitter;
    vec2 prev_jitter;
} push;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(U32(push.visbuffer));
    if (any(greaterThanEqual(coord, size))) return;

    FrameData frame_data = push.frame_data.data;
    VisbufferHit hit = decode_visbuffer_hit(
            push.visbuffer, coord, size,
            push.draw_data_buffer, push.index_buffer, push.vertex_buffer,
            frame_data.inv_proj_view, frame_data.camera_position
        );

    if (!hit.valid) {
        imageStore(RG16F(push.velocity_image), coord, vec4(0.0));
        return;
    }

    // no per-draw prev-transform yet, so this only captures camera motion
    vec4 curr_clip = frame_data.proj_view * vec4(hit.world_pos, 1.0);
    vec4 prev_clip = frame_data.prev_proj_view * vec4(hit.world_pos, 1.0);

    vec2 curr_ndc = curr_clip.xy / curr_clip.w;
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;

    // world_pos was found with the jittered inv_proj_view, so projecting it back already
    // removes the current frame's jitter and leaves only the previous frame's. adding each
    // frame's jitter back cancels it out, giving a velocity with no jitter (TAA and RTAO need this)
    curr_ndc += push.jitter;
    prev_ndc += push.prev_jitter;

    // NDC spans [-1, 1], UV spans [0, 1], so a delta there is half as large in UV space
    vec2 velocity = (curr_ndc - prev_ndc) * 0.5;

    imageStore(RG16F(push.velocity_image), coord, vec4(velocity, 0.0, 0.0));
}
