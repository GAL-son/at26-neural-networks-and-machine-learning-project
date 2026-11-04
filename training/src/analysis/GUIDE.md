# Dataset analysis guide

How to read the output of the analysis scripts for any captured scene.
For every metric: what it measures, which values are normal, which are suspicious, and what to do about them.

The value ranges below are **rules of thumb**, not hard limits. They are meant to tell "fine", "look closer"
and "something is broken" apart. Always confirm a suspicion visually in the contact sheets.

## Running

```powershell
python training/analysis/run_full_analysis.py --quick   # pipeline check, a few minutes
python training/analysis/run_full_analysis.py           # full run
python training/analysis/run_full_analysis.py --only analyze temporal
```

| Output | Written by | Section |
|---|---|---|
| console summary per scene | all steps | — |
| `results/logs/<step>.log` | runner | full console output, including tracebacks |
| `inventory.json`, `inventory_<scene>.csv`, `inventory_timeline.png` | `inventory.py` | [1](#1-inventory) |
| `contact_sheets/*.png` | `contact_sheet.py` | [2](#2-contact-sheets) |
| `analysis.json`, `normalization.json`, `region_share.png`, `error_vs_edge_distance.png`, `edge_step_length.png`, `crop_energy.png`, `distributions.png` | `analyze.py` | [3](#3-analyze) |
| `temporal.json`, `temporal_autocorrelation.png`, `temporal_flicker.png` | `temporal.py` | [4](#4-temporal) |

`--quick` results are only good for checking that everything runs. Never quote them.

## General rules

- **Read in order.** Inventory problems invalidate everything after them. A wrong region mask (seen in the contact sheets)
  invalidates every edge metric.
- **Compare methods within one scene, not numbers across scenes.** Scenes differ in texture detail, geometry and motion.
  A 30 dB scene and a 42 dB scene are not "worse" and "better", just different. Never average PSNR over scenes.
- **Terms used below:**
  - **Input**: the aliased 1-sample albedo.
  - **GT**: the 4×4 supersampled albedo.
  - **Edge zone**: pixels within 1.5 px of a depth jump, normal crease or background border.
  - **Interior**: all other geometry.
- Errors are measured on **sRGB** values in 0–1 unless stated otherwise. RMSE × 255 ≈ error in grey levels.

---

## 1. Inventory

Integrity of every frame. Every value should be clean before you look at anything else.

### Structure

| Field | Expected | Suspicious / broken | Action |
|---|---|---|---|
| `frames` | Equals the frame count of the scene's `animation.json` | Fewer frames | Renderer stopped early or the writer dropped frames. Re-render the scene. |
| `missing_indices` | `[]` | Any entry | Re-render. Until then, don't build temporal pairs across the gap and don't let the split cross it. |
| `header_index_mismatch` | `[]` | Any entry | The writer saved frames out of order. Motion vectors would point at the wrong previous frame. Fix before temporal work. |
| `resolutions` | One value | Several values | Frames recorded at different window sizes (windowed mode). Re-render headless. |
| `ssaa` | One value, the same in all scenes | Differs between scenes | The GT quality differs, and so does what "correct" means. Don't mix such scenes in one training set. |
| `near`, `far` | One value per scene | Several values | Settings changed mid-recording. Re-render. |

### Values

| Field | Expected | Suspicious / broken | Action |
|---|---|---|---|
| `nonfinite_values` | 0 | > 0 | One NaN in a crop makes the loss NaN. Usual causes: degenerate triangles (normals) or broken depth. Fix the mesh. As a stopgap, apply `nan_to_num` in the loader. |
| `max_normal_len_err` | ≤ 0.002 (half-float rounding) | 0.002–0.05 | The mesh has unnormalised normals. Renormalise in the loader. |
| | | > 0.05 or ≈ 1 | Missing or zero normals in the OBJ. Edge detection and the normal input are both unreliable. Re-export with normals. |
| `gt_range` | Inside 0–1, often exactly 0 and 1 | Below 0 or above 1 | Unexpected for albedo: blending or format bug. Investigate before training. |

If `gt_range` reaches exactly 0 and 1, don't end the network with a sigmoid, which can never output 0 or 1. Use a linear output and clamp at inference.

### Content

| Field | Normal | Look closer | Problem |
|---|---|---|---|
| `near_static_pairs` (share of frames) | 0–5% | 5–20%: camera often pauses without looking around | > 20%: many duplicates. Regenerate the path or drop duplicates. |
| `mostly_background_frames` | 0 | 1–5%: a few views into the void | > 5%: the path leaves the geometry. Fix the path or drop those frames. |

### Motion

| `motion_mean_px` | Meaning |
|---|---|
| < 2 | Very calm. The previous frame contributes few new sub-pixel samples, so the temporal model will gain little. |
| 2–15 | Typical walking and looking around. |
| 15–40 | Fast: running, quick turns. |
| > 40 | Extreme. Check that the path isn't jittery, e.g. a wrong `UNITS_PER_METER` or a camera very close to geometry. |

| `motion_p99_px_max` | Meaning |
|---|---|
| < 50 | No large flicks. |
| 50–200 | Fast glances. History is partly invalid in those frames. |
| > 200 | Extreme flicks: large disocclusions. That's fine as a stress test, but check it is intended. |

Motion does not make single-frame anti-aliasing harder, because no motion blur is rendered. It sets the difficulty for the temporal model.

**If motion is very high and the median depth (section 3.6) is small,** the camera moves fast *relative to the scene's scale*. Check the scene units used in Blender.

### Cost

| Field | Normal | Problem |
|---|---|---|
| `dataset_gb` per frame | About 15–30 MB at 1080p; simpler scenes compress better | Much smaller: frames possibly empty. Check the contact sheets. |
| `mean_load_s` | 0.1–0.3 s from an SSD | > 0.5 s: HDD or network drive. Training will be I/O-bound, so copy the data to an SSD or cache it as `.npy`. |
| `full_pass_decode_min` | Minutes | Time to decode every frame once. Plan crop extraction around it: cut 8–16 crops per decoded frame. |

### `inventory_timeline.png`

- **Motion panel:**
  - short spikes are glances;
  - long plateaus are sustained fast movement;
  - a uniformly high line suggests jitter in the path.
- **Background panel:** spikes are frames looking outside the scene. Compare with the `mostly_background_frames` count.

---

## 2. Contact sheets

One sheet per sampled frame, eight panels:

1. input
2. GT
3. |GT − input|
4. regions (dark = background, grey = interior, orange = edge zone)
5. world normals
6. log depth
7. motion magnitude
8. zoom on the worst 64 px window: input on the left, GT on the right

Check every scene before reading section 3.

| Check | Healthy | Problem | Consequence and action |
|---|---|---|---|
| Orange edge zone | Thin lines along silhouettes and creases | Whole surfaces orange | The edge detector over-fires (faceted normals, depth noise), so every edge metric is measured on the wrong pixels. Raise `crease_deg` (e.g. 45) or `depth_tol` (e.g. 0.05) in `common.geometric_edges` and re-run `analyze.py`. |
| | | Visible edges without orange | The detector misses them, e.g. texture-only edges or very small depth steps. Expected for texture edges; lower `depth_tol` only if real geometry is missed. |
| Zoom: input vs GT | Same content; GT smoother on edges and fine texture | Content shifted between the halves | Input and GT are misaligned. This is a renderer bug, and training on it is meaningless. Stop and fix it. |
| Difference panel | Bright on edges and detailed textures, dark on flat areas | Uniformly bright flat areas | Colour or brightness mismatch between the passes. Compare the mean albedo and GT in section 3.6. |
| Geometry | Closed surfaces | Holes cut through nearby objects | Near-plane clipping: the camera gets closer than the near plane. The GT has the same holes, so it is learnable, but it's a limitation. Check `depth p1` in section 3.6. |
| Foliage, fences, chains | Cut-out shapes | Opaque rectangles | The renderer ignores alpha. The GT is identical, so it's consistent but unrealistic. Note it as a limitation. |
| Normals panel | Smooth colour on curved surfaces | Faceted patches, black holes | Flat-shaded or missing normals. Expect over-firing edge detection. |
| Motion panel | Smooth gradients following the geometry | Noise, or abrupt jumps inside one surface | Broken motion vectors. Confirm with the convention test in section 4. |

---

## 3. Analyze

Statistics over evenly spaced frames (default 40 per scene).

### 3.1 Where the error is: `regions`, `region_share.png`

The difference between input and GT is what the network has to learn. The split tells you *what kind* of problem the scene is.

**Edge zone pixel share (`regions.edge.pixel_share`):**

| Value | Meaning |
|---|---|
| 3–15% | Normal |
| 15–30% | Dense geometry (furniture, foliage, scanned meshes), or the detector over-fires. Verify in the contact sheets. |
| > 30% | Almost certainly over-firing. Fix the thresholds first (section 2). |

**Edge share of squared error (`regions.edge.error_share_srgb`):**

| Value | Scene type | Consequence |
|---|---|---|
| > 60% | Edge-dominated: classic geometric aliasing | The edge-weighted loss and edge metrics carry the result. |
| 30–60% | Mixed | Both edges and textures matter. |
| < 30% | Texture-dominated: shimmering detailed textures (the renderer has no mipmaps) | The network will mostly learn texture filtering. Expect a strong blur baseline (3.2). |

**Background share of error:** below 1% is expected. Above 2% means a pass mismatch, or silhouettes classified as background. Check the contact sheets.

**L1 share (linear) vs squared share (sRGB):**
- If they're similar, the colour space of the loss matters little.
- If the linear share is much lower in dark regions, an sRGB loss is important to avoid neglecting dark areas.

**Edge/interior error density (`error_density_edge_vs_interior`)**, the error of an edge pixel relative to an interior pixel:

| Value | Meaning | Edge weight λ in `L1 + λ·L1_edge` |
|---|---|---|
| < 2 | Edges hardly differ from interiors (texture-dominated, or a wrong mask) | 1, or no edge term |
| 2–10 | Typical | about density / 2, i.e. 1–5 |
| > 10 | Error strongly concentrated on edges | 5, plus importance-sampled crops |

### 3.2 Baselines: `baselines`

- `identity` is the input compared directly with the GT: the "no anti-aliasing" score.
- `blur_σ` is the input blurred with a Gaussian of σ pixels: the cheapest anti-aliasing.
- **The best blur is the real baseline.** A model that doesn't beat it has learned nothing a fixed filter can't do.

**Identity PSNR (`psnr_all`):**

| Value | Meaning |
|---|---|
| > 40 dB | Mild aliasing: smooth textures, few edges. Gains in dB will be small, so judge by edge PSNR and the flicker ratio. |
| 32–40 dB | Typical |
| 25–32 dB | Heavy aliasing, usually high-frequency textures without mipmaps |
| < 25 dB | Extreme. First rule out misalignment (contact sheet zoom). |

**Edge PSNR** is typically 3–6 dB below the full-frame PSNR. The edge zone is where the errors are.

**Best blur gain (best `psnr_all` − identity `psnr_all`):**

| Value | Meaning |
|---|---|
| < 1 dB | Mostly edge aliasing. Blur barely helps, which is a good scene to show that a network does better than filtering. |
| 1–4 dB | Typical |
| > 4 dB | Texture-dominated. Blur is a strong baseline; compare carefully on edge PSNR and interior SSIM. |

**Which blur wins:**
- **The smallest σ tested:** the true optimum may be smaller still. Add lower values to `BLUR_SIGMAS` before quoting the baseline.
- **The largest σ tested:** the GT is much smoother than the input. Plausible for very noisy textures, otherwise suspicious (check alignment).

**Identity SSIM:**

| Value | Meaning |
|---|---|
| > 0.98 | Mild |
| 0.90–0.98 | Typical |
| < 0.90 | Heavy |

If blur raises PSNR but *lowers* interior SSIM, it is destroying real texture detail. A good model must improve both.

**Rough targets for a trained model,** relative to the best blur, per scene:

| Result | Meaning |
|---|---|
| ≤ 0 dB | Failed |
| +0.5–1 dB | Useful |
| +1–3 dB | Good |
| > +3 dB | Excellent |

Report edge PSNR separately. A model can tie the blur on the full frame and still win clearly on edges.

### 3.3 Error radius: `error_radius_px`, `error_vs_edge_distance.png`

This is the distance from the nearest geometric edge within which 50%, 90% and 99% of the squared error lies. The value 32 means 32 px or more.

**50% radius:**

| Value | Meaning |
|---|---|
| 0–1 px | Edge-dominated |
| 2–8 px | Mixed |
| > 8 px | Texture-dominated |

90% and 99% at 32 is normal whenever there is any texture error. It doesn't mean the edges are 32 px wide. For the edge-weighted loss, keep the 1.5 px edge zone regardless.

### 3.4 Staircases: `edge_error_on_steps_longer_than`, `edge_step_length.png`

Edges close to horizontal or vertical form staircases with long steps. To smooth one, the network needs to see roughly one step length of context.

**Share of edge error on steps longer than 32 px:**

| Value | U-Net |
|---|---|
| < 15% | 4 levels are enough (theoretical receptive field about 200 px; the effective one is much smaller). |
| 15–30% | 5 levels |
| > 30% | 5 levels plus dilated convolutions in the bottleneck. Keep crops at 256 or larger. |

**Artefact:**
- With 1° angle bins, `>64px` and `>128px` are always identical. Both contain only the 0° bin, i.e. exactly axis-aligned edges.
- For those edges, a single frame holds no information about their sub-pixel position. Only temporal input can improve them, so they're a natural argument for the temporal model.

### 3.5 Crops: `crops`, `crop_energy.png`

This describes what a random 256×256 training sample contains.

**`negligible_share`** (crops with less than 5% of the average error):

| Value | Meaning |
|---|---|
| < 5% | Error everywhere (dense texture). Uniform sampling is fine. |
| 5–20% | Error-weighted sampling helps. |
| 20–50% | Many empty crops. Use error-weighted sampling. |
| > 50% | Mostly flat scene. Little to learn; prefer adding other scenes. |

**`top20_error_share`:**

| Value | Meaning |
|---|---|
| 20–40% | Even spread |
| 40–70% | Concentrated |
| > 70% | Very concentrated: a few regions hold almost all the signal. |

In the curve, the dotted diagonal is a perfectly even spread. The further the curve bows above it, the more concentrated the error.

**`sqrt_sampling.error_per_crop_vs_uniform`** (drawing crops with probability ∝ √error):



