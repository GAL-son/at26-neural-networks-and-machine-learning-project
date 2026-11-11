# Configs
JSON files read by the Python suite in `src/`. Relative paths are resolved **relative to the config file's directory**.

## `train.json` (dataset and splits)
| Key | Default | Meaning |
|---|---|---|
| `capturesDir` | `"../../data/captures"` | Folder with one sub-folder per captured scene (`<sceneFolder>_<objStem>/frame_*.exr`) |
| `expectedFrames` | 600 | Frames per sequence. Used to detect missing frames at the end. `null` = don't check |
| `splits.train` / `.val` / `.test` | `[]` | Capture folder names, e.g. `"sponza_sponza"`. A scene may be in one split only |

Scenes are split as whole sequences, never by frame or patch. The split is decided from the analysis results.

## `analysis.json`
| Key | Default | Meaning |
|---|---|---|
| `dataset` | `"train.json"` | Dataset config to use |
| `scenes` | `"all"` | `"all"` (everything in `capturesDir`), `"train"` / `"val"` / `"test"`, or a list of folder names |
| `stride` | 1 | Take a clip every `stride` frames |
| `clip` | 1 | Consecutive frames per clip. Use ≥ 2 with `stride` > 1 to keep (t-1, t) pairs for `temporal` |
| `limit` | `null` | Max frames per scene |
| `device` | `"auto"` | `auto` / `rocm` (= `cuda`) / `directml` / `cpu`. `auto` tries ROCm/CUDA, then DirectML, then CPU |
| `workers` | 4 | EXR decoding processes. 0 = decode in the main process |
| `prefetch` | 2 | Frames queued per worker (~27 MB each) |
| `outputDir` | `"../../reports/analysis"` | Each run writes to `<outputDir>/<timestamp>/` |
| `analyses.<name>` | | Parameters of one analysis. `"enabled": false` turns it off |

### `analyses.integrity`
| Key | Default | Meaning |
|---|---|---|
| `normalTolerance` | 0.01 | Max allowed `abs(length(normal) - 1)` on geometry |
| `reprojection` | `true` | Compare stored motion with motion recomputed from depth and the header matrices |
| `reprojectionThresholdPx` | 0.25 | Mean reprojection error (pixels) below which motion counts as consistent |

### `analyses.consistency`
| Key | Default | Meaning |
|---|---|---|
| `blockSize` | 8 | Tile size for comparing block averages of GT and input |
| `brightnessTolerance` | 0.02 | Max allowed deviation of the GT / input brightness ratio from 1, per channel |
| `blockErrorThreshold` | 0.02 | Max mean absolute difference of block averages (linear) |
| `shiftThresholdPx` | 0.05 | Max allowed sub-pixel shift between GT and input |
| `clipThreshold` | 0.01 | Max share of input pixels at the top 8-bit value (1.0) |

### `analyses.residual`
| Key | Default | Meaning |
|---|---|---|
| `changeThreshold` | 2/255 | Per-pixel sRGB difference above which a pixel counts as changed by anti-aliasing |
| `depthTolerance`, `creaseDegrees` | 0.02, 30 | Geometric edge detection: relative 1/z step, normal angle |
| `patchSize` | 256 | Patch size for the per-patch error distribution |
| `flatPatchPsnr` | 45 | Patches whose input already reaches this PSNR count as flat (nothing to learn) |

### `analyses.temporal`
| Key | Default | Meaning |
|---|---|---|
| `matrixLayout` | `"transposed"` | `"transposed"` (M = stored.T, as documented) or `"asStored"`; use what `integrity` reports |
| `occlusionTolerance` | 0.02 | Relative depth mismatch that marks a pixel's history as invalid (disocclusion) |
| `staticMotionPx`, `fastMotionPx` | 0.1, 4 | Mean motion (px / frame) thresholds of the static / slow / fast scene category |