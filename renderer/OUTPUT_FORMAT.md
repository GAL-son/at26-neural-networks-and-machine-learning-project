# Renderer output format

What the renderer writes to disk, so the training suite can read it without going through the C++.
Source of truth: `src/IO/ExrFrameWriter.cpp`, `resources/shaders/source/geometry.*`, `ground_truth.*`, `ssaa_downsample.comp`.

## Files and directories

```
<outputDir>/<sceneFolder>_<objStem>/frame_000000.exr
                                    frame_000001.exr
                                    ...
                                    frame_<N-1>.exr
```

- `outputDir` and `scenesDir` come from `config.json`. They are resolved **relative to the config file's directory**.
  Scene paths are resolved relative to `scenesDir`.
- The folder name is `<parent folder of .obj>_<.obj stem>`, e.g. `sponza/sponza.obj` -> `sponza_sponza/`,
  `bedroom/iscv2.obj` -> `bedroom_iscv2/`.
- `N` = number of frames in the scene's `animation.json`. Without that file it is a generated orbit of **600** frames.
- File index = animation frame index (`%06u`, starting at 0) = frame number in the Blender preview.
- One multi-layer EXR per frame, single part, scanline, **ZIP** compression.
- Frames are written by a background thread. A write error is only printed to stderr, so
  **check that the indices are contiguous** when ingesting.
- **Frame rate: every sequence is assumed to run at 60 fps**, so frame `i` is at `t = i / 60` s and a 600-frame sequence lasts 10 s.
The renderer has no notion of time and just renders one camera pose per frame. The 60 fps is set when the camera path is made:
`FPS = 60` in `preprocesing/blender_camera_path.py`, stored as `"fps"` in `animation.json` (the renderer ignores it).  The generated orbit (no `animation.json`) does one full loop in 600 frames, which is also 10 s at 60 fps.


## Image

| Property | Value |
|---|---|
| Resolution | 1920 x 1080 (`Engine::kRenderExtent`) |
| Origin | Row 0 = **top** of the image, column 0 = left (Vulkan framebuffer = EXR line order) |
| Pixel centre | `uv = ((x + 0.5) / W, (y + 0.5) / H)` |
| Camera | Perspective, vertical FOV 80°, aspect 16:9, near 0.1, far 10000 |
| Scene | Static. Only the camera moves. There is no model matrix, so object space = world space (Y up) |
| Lighting | **None.** Every colour buffer is unlit albedo |
| Frame rate | 60 fps (assumed, see above) |

## Channels

| EXR channel(s) | Type | GPU source | Content | Range | Background (no geometry) |
|---|---|---|---|---|---|
| `albedo.R/G/B` | HALF | `R8G8B8A8_UNORM` | Aliased albedo, **linear** | [0, 1], steps of 1/255 | 0.1 |
| `normal.X/Y/Z` | HALF | `R16G16B16A16_SFLOAT` | Aliased **world-space** normal, unit length | [-1, 1] | (0, 0, 0) |
| `motion.X/Y` | HALF | `R16G16_SFLOAT` | Aliased screen-space motion, UV units | ~[-1, 1] | (0, 0) |
| `depth.Z` | FLOAT | `D32_SFLOAT` | Aliased **linear view-space Z** (distance along camera forward) | [0.1, 10000] | 10000 (far plane) |
| `ground-truth.R/G/B` | HALF | `R16G16B16A16_SFLOAT` | SSAA x16 albedo, **linear** | [0, 1] | 0.1 |

Alpha is not stored for any layer.

### Albedo (`albedo.*`)
- Textured materials: diffuse texture sampled as `R8G8B8A8_SRGB`, so the values are decoded to **linear**.
  Untextured materials: MTL `Kd` used as-is.
- The render target is 8-bit UNORM (linear, no sRGB encoding), so values are quantised to `k/255`, then stored as half.
- For display, apply the sRGB transfer function.
- Material opacity and texture alpha are **ignored** (no blending, no alpha test). Cut-out geometry such as leaves renders as solid quads.
- Textures have **no mipmaps** (bilinear only), so the aliased input also has texture aliasing/shimmer, not just edge aliasing.

### Ground truth (`ground-truth.*`)
- Same shader as albedo (`shadeAlbedo`), same camera, same frame, **no jitter**.
- Rendered at 4x resolution per axis (7680 x 4320) into an 8-bit UNORM target, then reduced with a
  4x4 **box filter** (plain mean of 16 samples, in linear space). Sample grid for output pixel `(x, y)`:
  super-pixels `[4x, 4x+3] x [4y, 4y+3]`.
- Stored as half. Values are multiples of about 1/(255*16).
- The header attribute `ssaaFactor` = 4 (per axis).

### Normals (`normal.*`)
- Interpolated vertex normals from the OBJ, renormalised per pixel. Normal maps are not applied.
- OBJ faces without normals get `(0, 1, 0)`.
- Back-face culling is off and back faces are **not** flipped, so a normal can point away from the camera.
- **Geometry mask:** `mask = |normal| > 0.5`. Background is exactly `(0, 0, 0)`.

### Motion vectors (`motion.*`)
- `motion = uv_current - uv_previous`, computed per pixel as `(ndc_curr - ndc_prev) * 0.5` from the current and previous view-projection.
- Units: fraction of image size. **+X = right, +Y = down.** Pixel units: `motion * (W, H)`.
- Where the pixel was in the previous frame: `uv_prev = uv - motion`.
- Camera motion only, since the scene is static. It can be recomputed from `depth` and the two matrices in the header.
- **Frame 0 has zero motion everywhere** (`prevViewProjection == viewProjection`).
- Background has 0 motion, even when the camera rotates.\
- The displacement covers **one frame (1/60 s)``. Velocity in UV/s = 'motion * 60'. Sequences made with different fps will have larger or smaller motion per frame.

### Depth (`depth.Z`)
- Linearised from the hardware depth `d` (standard Z, 0 = near, 1 = far, `GLM_FORCE_DEPTH_ZERO_TO_ONE`):
  `z = near * far / (far - d * (far - near))`.
- This is **planar view-space Z**, not Euclidean distance to the camera (the header string says "distance", but it is Z).
- Background = `farPlane` (10000). Use the normal mask, not a depth threshold, to find geometry.

## Header attributes

| Attribute | Type | Meaning |
|---|---|---|
| `frameIndex` | int | Same as the number in the file name |
| `ssaaFactor` | int | SSAA factor **per axis** (4 -> 16 samples) |
| `nearPlane`, `farPlane` | float | 0.1, 10000 |
| `viewProjection` | m44f | `P * V` of this frame (Vulkan clip space, Y flipped, Z in [0, 1]) |
| `prevViewProjection` | m44f | `P * V` of the previous frame (equal to `viewProjection` at frame 0) |
| `motionConvention` | string | Human-readable reminder of the motion convention |
| `depthConvention` | string | Human-readable reminder of the depth convention |

**Matrix layout:** the glm (column-major) data is copied byte-for-byte into the Imath matrix.
If a reader returns it as a 4x4 array `A` with `A[i][j]` = Imath `x[i][j]`, then `A` is the **transpose**
of the mathematical matrix:
`M = A.T`, `clip = M @ [x, y, z, 1]` (equivalently `clip = [x, y, z, 1] @ A`), `ndc = clip.xyz / clip.w`, `uv = ndc.xy * 0.5 + 0.5`.
Check this once on real data, e.g. reproject a pixel using its depth and compare against `motion`.

## Size

- 1920 x 1080 = 2,073,600 px x 12 channels.
- At native precision (11 x half + 1 x float) that is 26 B/px, about **54 MB per frame and about 32 GB per 600-frame sequence** uncompressed.
  As float32 it is about 100 MB per frame and about 60 GB per sequence. Don't load a whole sequence into RAM; use patches, crops or memory maps.



