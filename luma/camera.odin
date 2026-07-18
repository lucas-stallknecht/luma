package luma

import "core:math"
import la "core:math/linalg"
import "core:math/linalg/glsl"

TAA_JITTER_SAMPLES :: 8

Camera :: struct {
	near:             f32,
	far:              f32,
	move_speed:       f32,
	look_sensitivity: f32,
	fov:              f32,
	position:         glsl.vec3,
	rotation:         glsl.quat,
	proj:             glsl.mat4,
}

create_default_camera :: proc() -> Camera {
	return Camera {
		near = 0.01,
		far = 100.0,
		move_speed = 4.0,
		look_sensitivity = 0.2,
		fov = 60.0,
		position = glsl.vec3{5.0, 1.0, -0.3},
		rotation = la.quaternion_from_euler_angle_y_f32(math.PI / 2),
		proj = glsl.mat4(1.0),
	}
}

camera_update_proj :: proc(cam: ^Camera, aspect_ratio: f32) {
	cam.proj = la.matrix4_perspective_f32(
		math.to_radians_f32(cam.fov),
		aspect_ratio,
		cam.near,
		cam.far,
		true, // reversed-z
	)
	// Vulkan's clip space has Y pointing down, unlike OpenGL
	cam.proj[1][1] *= -1.0
}

// a subpixel shift so TAA samples the scene slightly differently each frame. the Halton
// sequence spreads the shifts out evenly
camera_taa_jitter :: proc(frame_idx: u32) -> glsl.vec2 {
	// start at 1: Halton's sample 0 is zero
	i := frame_idx % TAA_JITTER_SAMPLES + 1
	return {halton(i, 2) - 0.5, halton(i, 3) - 0.5}
}

// applies the jitter to the projection so the whole image shifts by that amount. the matrix
// stays invertible, so world positions can still be reconstructed from it
camera_jittered_proj :: proc(cam: ^Camera, jitter_ndc: glsl.vec2) -> glsl.mat4 {
	proj := cam.proj
	// these two cells add a screen-space shift of jitter_ndc
	proj[2][0] += jitter_ndc.x
	proj[2][1] += jitter_ndc.y
	return proj
}

camera_get_view :: proc(cam: ^Camera) -> glsl.mat4 {
	rot := la.matrix4_from_quaternion(la.quaternion_inverse(cam.rotation))
	trans := la.matrix4_translate_f32(-cam.position)
	return rot * trans
}

camera_get_forward :: proc(cam: ^Camera) -> glsl.vec3 {
	return la.mul(cam.rotation, glsl.vec3{0.0, 0.0, -1.0})
}

camera_get_up :: proc(cam: ^Camera) -> glsl.vec3 {
	return la.mul(cam.rotation, glsl.vec3{0.0, 1.0, 0.0})
}

camera_get_right :: proc(cam: ^Camera) -> glsl.vec3 {
	return la.mul(cam.rotation, glsl.vec3{1.0, 0.0, 0.0})
}

camera_move_forward :: proc(cam: ^Camera, d: f32) {
	cam.position += camera_get_forward(cam) * d * cam.move_speed
}

camera_move_up :: proc(cam: ^Camera, d: f32) {
	cam.position += camera_get_up(cam) * d * cam.move_speed
}

camera_move_right :: proc(cam: ^Camera, d: f32) {
	cam.position += camera_get_right(cam) * d * cam.move_speed
}

camera_rotate :: proc(cam: ^Camera, delta: glsl.vec2) {
	yaw := -delta.x * (cam.look_sensitivity * 0.01)
	pitch := -delta.y * (cam.look_sensitivity * 0.01)

	// yaw around world up, pitch around local right, to avoid roll
	yaw_quat := la.quaternion_angle_axis_f32(yaw, glsl.vec3{0.0, 1.0, 0.0})
	cam.rotation = yaw_quat * cam.rotation
	pitch_quat := la.quaternion_angle_axis_f32(pitch, camera_get_right(cam))
	cam.rotation = pitch_quat * cam.rotation

	cam.rotation = la.normalize(cam.rotation)
}

@(private = "file")
halton :: proc(index: u32, base: u32) -> f32 {
	f: f32 = 1.0
	r: f32 = 0.0
	i := index
	for i > 0 {
		f /= f32(base)
		r += f * f32(i % base)
		i /= base
	}
	return r
}
