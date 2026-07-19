# luma

luma is a toy renderer written in Odin and Vulkan. Beyond the "vendor" libraries that ship with Odin, its only dependency is [Dear ImGui](https://github.com/ocornut/imgui) (via [odin-imgui](https://gitlab.com/L-4/odin-imgui)) for the debug UI, vendored as a git submodule and built once with a small setup script. It's mostly a place for me to try out rendering techniques I want to understand.

Scenes are exported from Blender via the `tools/blender_export` Python script, straight into a binary format the renderer reads directly.

![](misc/preview.png)

## Frame flow

Each frame is a render graph rebuilt from scratch, with barriers inferred from every pass' declared resource usage. Passes flow top-down below. Arrows are labelled with the resource that creates the dependency, and the dashed edge into Present is the TAA-off bypass.

![](misc/frame_flow.png)

## Graphics techniques

- **GPU-driven rendering**: an early pass at it. Draw data lives on the GPU, though there's no GPU frustum culling yet.
- **Rasterized visbuffer with shading in compute.** This will be proven useful once there are multiple point lights to shade.
- **Frostbite BRDF shading**, plus the usual essentials like normal mapping.
- **Ray traced directional hard shadows.** Soft shadows will follow.
- **Ray traced ambient occlusion**, denoised using temporal reprojection and bilateral filtering.
- **Temporal anti-aliasing** using already present motion vectors and subpixel camera jitter.
- **Physically correct bloom.**
- **DDGI** for global illumination, using irradiance probes. Radiance is stored as spherical harmonics, with the coefficients computed by shooting rays around each probe and integrating with Monte Carlo. Bounces accumulate over frames. The per-probe depth with octahedral mapping isn't implemented yet.

  ![](misc/gi_demo.avif)
  ![](misc/probes.png)

- **A sky shader** baked into a cubemap. Not mine, borrowed.

## Vulkan features

- Shader reloading with managed pipelines
- Bindless architecture
- Ray Tracing
- On-demand command handling

## Running a release build

Download the latest Windows release, unzip it, and run `luma_windows.exe`. Nothing else to install.

You'll need a GPU with ray tracing support and an up to date driver. This has only been tested on an AMD Radeon RX 9070 XT on Windows, so other GPUs may not work.

On launch it will ask you to pick a window size and a GPU in the console before opening the render window.

## Controls

- **Right mouse button** (hold): look around
- **WASD**: move
- **Space** / **Left Ctrl**: move up / down
- **R**: reload shaders on the fly

## Building from source

You'll need the [Odin compiler](https://odin-lang.org/), the [Vulkan SDK](https://vulkan.lunarg.com/), Python, and Git installed, along with a GPU that supports Vulkan ray tracing (`VK_KHR_ray_tracing_pipeline`). With those in place:

```sh
git submodule update --init
python pre_build.py   # clones and builds the Dear ImGui backends used by imgui/build.py, once
mkdir build
odin run luma -out:build/luma
```
