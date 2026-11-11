# Configs


JSON files read by the Python suite in `src/`. Relative paths are resolved **relative to the config file's directory**.


## `dataset.json`


| Key | Default | Meaning |
|---|---|---|
| `capturesDir` | `"../data/captures"` | Folder with one sub-folder per captured scene (`<sceneFolder>_<objStem>/frame_*.exr`) |
| `expectedFrames` | 600 | Frames per sequence. Used to detect missing frames at the end. `null` = don't check |
| `splits.train` / `.val` / `.test` | `[]` | Capture folder names, e.g. `"sponza_sponza"`. A scene may be in one split only |


Scenes are split as whole sequences, never by frame or patch. The split is decided from the analysis results.


## `analysis.json`


| Key | Default | Meaning |
|---|---|---|
| `dataset` | `"dataset.json"` | Dataset config to use |
| `scenes` | `"all"` | `"all"` (everything in `capturesDir`), `"train"` / `"val"` / `"test"`, or a list of folder names |
| `stride` | 1 | Take a clip every `stride` frames |
| `clip` | 1 | Consecutive frames per clip. Use ≥ 2 with `stride` > 1 to keep (t-1, t) pairs for temporal analyses |
| `limit` | `null` | Max frames per scene |
| `device` | `"auto"` | `auto` / `rocm` (= `cuda`) / `directml` / `cpu`. `auto` tries ROCm/CUDA, then DirectML, then CPU |
| `workers` | 4 | EXR decoding processes. 0 = decode in the main process |
| `prefetch` | 2 | Frames queued per worker (~27 MB each) |
| `outputDir` | `"../reports/analysis"` | Each run writes to `<outputDir>/<timestamp>/` |
| `analyses.<name>` | | Parameters of one analysis. `"enabled": false` turns it off |


### `analyses.integrity`


| Key | Default | Meaning |
|---|---|---|
| `normalTolerance` | 0.01 | Max allowed `abs(length(normal) - 1)` on geometry |
| `reprojection` | `true` | Compare stored motion with motion recomputed from depth and the header matrices |
| `reprojectionThresholdPx` | 0.25 | Mean reprojection error (pixels) below which motion counts as consistent |




