package luma

import "base:runtime"
import "core:fmt"
import "core:math/linalg/glsl"
import "core:unicode/utf8"
import "vendor:glfw"
import mu "vendor:microui"

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
	mouse_position:      glsl.vec2,
	ui_ctx:              ^mu.Context, // when set, GLFW input is also forwarded to microui
}

window_bind_ui :: proc(win: ^Window, ctx: ^mu.Context) {
	win.ui_ctx = ctx
}

window_init :: proc(win: ^Window, width: u32 = 1920, height: u32 = 1080) -> bool {
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
	glfw.SetScrollCallback(win.glfw_window_ptr, scroll_callback)
	glfw.SetCharCallback(win.glfw_window_ptr, char_callback)

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

	if win.ui_ctx != nil {
		context = runtime.default_context()
		if mu_key, ok := mu_key_map(key); ok {
			if action == glfw.PRESS || action == glfw.REPEAT {
				mu.input_key_down(win.ui_ctx, mu_key)
			} else if action == glfw.RELEASE {
				mu.input_key_up(win.ui_ctx, mu_key)
			}
		}
	}
}

@(private = "file")
cursor_pos_callback :: proc "c" (window: glfw.WindowHandle, xpos, ypos: f64) {
	win := (^Window)(glfw.GetWindowUserPointer(window))
	new_pos := glsl.vec2{f32(xpos), f32(ypos)}
	win.mouse_position = new_pos

	// captured mode reports an unbounded virtual cursor, not meaningful for UI hit-testing
	if win.ui_ctx != nil && win.mouse_state != .Fully_Captured {
		context = runtime.default_context()
		mu.input_mouse_move(win.ui_ctx, i32(new_pos.x), i32(new_pos.y))
	}

	if win.mouse_state == .Not_Captured do return

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

	if win.ui_ctx != nil {
		if mu_btn, ok := mu_mouse_button(button); ok {
			context = runtime.default_context()
			x, y := i32(win.mouse_position.x), i32(win.mouse_position.y)
			// always forward releases so a button can't get stuck down in ctx.mouse_down_bits
			if action == glfw.PRESS && win.mouse_state != .Fully_Captured {
				mu.input_mouse_down(win.ui_ctx, x, y, mu_btn)
			} else if action == glfw.RELEASE {
				mu.input_mouse_up(win.ui_ctx, x, y, mu_btn)
			}
		}
	}

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

@(private = "file")
scroll_callback :: proc "c" (window: glfw.WindowHandle, xoffset, yoffset: f64) {
	win := (^Window)(glfw.GetWindowUserPointer(window))
	if win.ui_ctx == nil do return
	context = runtime.default_context()
	mu.input_scroll(win.ui_ctx, 0, i32(-yoffset * 30))
}

@(private = "file")
char_callback :: proc "c" (window: glfw.WindowHandle, codepoint: rune) {
	win := (^Window)(glfw.GetWindowUserPointer(window))
	if win.ui_ctx == nil do return
	context = runtime.default_context()
	buf, n := utf8.encode_rune(codepoint)
	mu.input_text(win.ui_ctx, string(buf[:n]))
}

// RIGHT is the camera-look button and deliberately unmapped, see mouse_button_callback
@(private = "file")
mu_mouse_button :: proc "contextless" (glfw_button: i32) -> (mu.Mouse, bool) {
	switch glfw_button {
	case glfw.MOUSE_BUTTON_LEFT:
		return .LEFT, true
	case glfw.MOUSE_BUTTON_MIDDLE:
		return .MIDDLE, true
	}
	return {}, false
}

@(private = "file")
mu_key_map :: proc "contextless" (glfw_key: i32) -> (mu.Key, bool) {
	switch glfw_key {
	case glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT:
		return .SHIFT, true
	case glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL:
		return .CTRL, true
	case glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT:
		return .ALT, true
	case glfw.KEY_BACKSPACE:
		return .BACKSPACE, true
	case glfw.KEY_DELETE:
		return .DELETE, true
	case glfw.KEY_ENTER:
		return .RETURN, true
	case glfw.KEY_LEFT:
		return .LEFT, true
	case glfw.KEY_RIGHT:
		return .RIGHT, true
	case glfw.KEY_HOME:
		return .HOME, true
	case glfw.KEY_END:
		return .END, true
	case glfw.KEY_A:
		return .A, true
	case glfw.KEY_X:
		return .X, true
	case glfw.KEY_C:
		return .C, true
	case glfw.KEY_V:
		return .V, true
	}
	return {}, false
}
