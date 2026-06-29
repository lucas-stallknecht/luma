package luma

import "core:fmt"
import "core:math/linalg/glsl"
import "vendor:glfw"

MouseState :: enum {
	Not_Captured,
	First_Captured,
	Fully_Captured,
}

Window :: struct {
	width:               u32,
	height:              u32,
	glfw_window_ptr:     glfw.WindowHandle,
	resized:             bool,
	minimized:           bool,
	pressed_keys:        [512]bool,
	mouse_state:         MouseState,
	last_mouse_position: glsl.vec2,
	mouse_delta:         glsl.vec2,
}

window_init :: proc(win: ^Window, width: u32 = 1600, height: u32 = 900) -> bool {
	win.width = width
	win.height = height
	win.mouse_state = .Not_Captured

	if !glfw.Init() {
		fmt.println("[Window] GLFW initialization failed")
		return false
	}

	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	glfw.WindowHint(glfw.RESIZABLE, glfw.FALSE)

	win.glfw_window_ptr = glfw.CreateWindow(
		i32(win.width),
		i32(win.height),
		"Luma Renderer",
		nil,
		nil,
	)
	if win.glfw_window_ptr == nil {
		fmt.println("[Window] Failed to create GLFW window")
		glfw.Terminate()
		return false
	}

	glfw.SetWindowUserPointer(win.glfw_window_ptr, win)

	glfw.SetWindowSizeCallback(win.glfw_window_ptr, size_callback)
	glfw.SetKeyCallback(win.glfw_window_ptr, key_callback)
	glfw.SetCursorPosCallback(win.glfw_window_ptr, cursor_pos_callback)
	glfw.SetMouseButtonCallback(win.glfw_window_ptr, mouse_button_callback)

	if glfw.RawMouseMotionSupported() {
		glfw.SetInputMode(win.glfw_window_ptr, glfw.RAW_MOUSE_MOTION, 1)
	}

	fmt.printfln("[Window] Created %dx%d", win.width, win.height)
	return true
}

window_cleanup :: proc(win: ^Window) {
	if win.glfw_window_ptr != nil {
		glfw.DestroyWindow(win.glfw_window_ptr)
		glfw.Terminate()
		win.glfw_window_ptr = nil
	}
}

window_should_close :: proc(win: ^Window) -> bool {
	return bool(glfw.WindowShouldClose(win.glfw_window_ptr))
}

window_update :: proc(win: ^Window) {
	glfw.PollEvents()
}

window_consume_mouse_delta :: proc(win: ^Window) -> glsl.vec2 {
	d := win.mouse_delta
	win.mouse_delta = {}
	return d
}

@(private = "file")
size_callback :: proc "c" (window: glfw.WindowHandle, w, h: i32) {
	win := (^Window)(glfw.GetWindowUserPointer(window))
	win.resized = true
	win.width = u32(w)
	win.height = u32(h)
	win.minimized = (w == 0 || h == 0)
}

@(private = "file")
key_callback :: proc "c" (window: glfw.WindowHandle, key, scancode, action, mods: i32) {
	win := (^Window)(glfw.GetWindowUserPointer(window))
	if key >= 0 && key < i32(len(win.pressed_keys)) {
		win.pressed_keys[key] = (action == glfw.PRESS || action == glfw.REPEAT)
	}
}

@(private = "file")
cursor_pos_callback :: proc "c" (window: glfw.WindowHandle, xpos, ypos: f64) {
	win := (^Window)(glfw.GetWindowUserPointer(window))
	if win.mouse_state == .Not_Captured do return

	new_pos := glsl.vec2{f32(xpos), f32(ypos)}

	if win.mouse_state == .First_Captured {
		win.last_mouse_position = new_pos
		win.mouse_state = .Fully_Captured
		return
	}
	win.mouse_delta += new_pos - win.last_mouse_position
	win.last_mouse_position = new_pos
}

@(private = "file")
mouse_button_callback :: proc "c" (window: glfw.WindowHandle, button, action, mods: i32) {
	win := (^Window)(glfw.GetWindowUserPointer(window))
	if button != glfw.MOUSE_BUTTON_RIGHT do return

	if action == glfw.PRESS {
		win.mouse_state = .First_Captured
		glfw.SetInputMode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
	} else if action == glfw.RELEASE {
		win.mouse_state = .Not_Captured
		win.mouse_delta = {}
		glfw.SetInputMode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
	}
}
