package noble

import "core:math"
import la "core:math/linalg"
import "core:math/linalg/glsl"

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

create_camera :: proc() -> Camera {
	rotation: glsl.quat
	rotation.w = 1
	return Camera {
		near = 0.01,
		far = 100.0,
		move_speed = 4.0,
		look_sensitivity = 0.2,
		fov = 60.0,
		position = glsl.vec3{0.0, 0.0, 2.0},
		rotation = rotation,
		proj = glsl.mat4(1.0),
	}
}

camera_update_proj :: proc(cam: ^Camera, aspect_ratio: f32) {
	cam.proj = la.matrix4_perspective_f32(
		math.to_radians_f32(cam.fov),
		aspect_ratio,
		cam.near,
		cam.far,
		true // reversed-z
	)
	// vulkan specific
	cam.proj[1][1] *= -1.0
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

	// world up
	yaw_quat := la.quaternion_angle_axis_f32(yaw, glsl.vec3{0.0, 1.0, 0.0})
	cam.rotation = yaw_quat * cam.rotation
	// local right
	pitch_quat := la.quaternion_angle_axis_f32(pitch, camera_get_right(cam))
	cam.rotation = pitch_quat * cam.rotation

	cam.rotation = la.normalize(cam.rotation)
}
