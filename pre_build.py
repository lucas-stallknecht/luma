import importlib.util
import os
import sys

# luma only needs the GLFW + Vulkan imgui backends (see luma/window.odin).
# imgui/build.py hardcodes its backend list as a module-level global rather
# than a CLI flag, so we import it and override that global before running.
WANTED_BACKENDS = ["vulkan", "glfw"]

root_dir = os.path.dirname(os.path.abspath(__file__))
imgui_dir = os.path.join(root_dir, "imgui")
build_py = os.path.join(imgui_dir, "build.py")

def main():
	spec = importlib.util.spec_from_file_location("imgui_build", build_py)
	imgui_build = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = imgui_build
	spec.loader.exec_module(imgui_build)

	imgui_build.wanted_backends = WANTED_BACKENDS

	cwd = os.getcwd()
	os.chdir(imgui_dir)
	try:
		imgui_build.main()
	finally:
		os.chdir(cwd)

if __name__ == "__main__":
	main()
