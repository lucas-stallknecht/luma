# luma

luma is a toy renderer written in Odin and Vulkan. It has no external dependencies beyond the "vendor" libraries that ship with Odin, so it builds without any setup. It's mostly a place for me to try out rendering techniques I want to understand.

Scenes are exported from Blender via the `tools/blender_export` Python script, straight into a binary format the renderer reads directly.

![](misc/preview.png)

## Graphics techniques

- **GPU-driven rendering**: an early pass at it. Draw data lives on the GPU, though there's no GPU frustum culling yet.
- **Rasterized visbuffer with shading in compute.** This will be proven useful once there are multiple point lights to shade.

  ![](misc/visbuffer.png)

- **Frostbite BRDF shading**, plus the usual essentials like normal mapping.
- **Ray traced directional hard shadows and ambient occlusion.** Soft shadows and denoising will follow once motion vectors are in.
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

## Building

You'll need the [Odin compiler](https://odin-lang.org/) and the [Vulkan SDK](https://vulkan.lunarg.com/) installed, along with a GPU that supports Vulkan ray tracing (`VK_KHR_ray_tracing_pipeline`). With those in place:

```sh
mkdir build
odin run luma -out:build/luma
./build/luma
```
