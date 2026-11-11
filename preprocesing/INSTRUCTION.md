# Preprocessing: camera paths in Blender

`blender_camera_path.py` turns a curve drawn in Blender into a player-like camera animation
and writes it as `animation.json` next to the scene's `.obj`. The renderer picks that file up
automatically when it loads the scene. Without it, the renderer falls back to a generated orbit.

Each frame gets a position, yaw and pitch. Movement follows the curve with a few distinct speeds
(idle / walk / run / sprint) and smooth acceleration. On top of that come slow pans, quick glances,
head bob and hand jitter. All of it is generated from Perlin noise and is fully reproducible for a given `SEED`.

## Requirements

- Blender 4.x or 5.x. numpy ships with Blender, so nothing needs to be installed.
- The scene as an `.obj` inside the renderer's scenes directory, for example `scenes/bedroom/iscv2.obj`.

## Workflow

### 1. Import the scene

File → Import → Wavefront (.obj), keeping the defaults: **Forward −Z, Up Y, Scale 1.0**.

### 2. Check scale and orientation

- **Scale.** OBJ files carry no units. Select something you know (a door, a chair) and read
  N → Item → Dimensions. A door around 2.1 means metres (`UNITS_PER_METER = 1`).
  A door around 210 means centimetres (`UNITS_PER_METER = 100`).
- **Orientation.** The floor must be flat on Blender's XY plane, with Z pointing up.
  If the scene lies on its side:
  1. Rotate it, then apply with Ctrl+A → All Transforms.
  2. **Re-export the `.obj`** over the original: File → Export → Wavefront, Forward −Z, Up Y,
     Selection Only (meshes), Materials on, Path Mode Relative.

  > The renderer loads the `.obj`, not the `.blend`. Fixing orientation in Blender without
  > re-exporting puts the camera path and the geometry in different coordinate systems.

### 3. Draw the route

1. Set the 3D cursor to eye height. N → View → 3D Cursor → Location Z = floor height + 1.7 m
   (or 170 in a cm scene). You can also draw on the floor and set `EYE_HEIGHT_M = 1.7` instead.
2. Shift+A → Curve → Bézier. Tab into Edit Mode, then A and X → Vertices to empty it.
3. Numpad 7 for top view. Use Alt+Z (X-ray) or Alt+B (clip region) if the ceiling is in the way.
4. Pick the **Draw** tool from the toolbar. Set Type: Bézier, Depth: Cursor, Tolerance about 5–10 px.
5. Drag the route. **The start of the stroke is frame 0.**
6. Clean up as needed:
   - **G** to move points; **G Z** for stairs or other floors;
   - **V → Automatic** for smooth handles;
   - **Alt+C** for a closed loop;
   - Segment → Switch Direction to reverse.
7. In Object Data properties:
   - set **Resolution Preview U** to 32;
   - keep **Extrude** and **Bevel Depth** at 0.

Route tips:

- Stay at least about 0.5 m from walls. The camera turns up to ±110° and bobs slightly.
- Sharp corners are fine. The look direction smooths through them.
- The script needs exactly **one spline per curve object**. Join strokes with F, or use one curve per run.

### 4. Run the script

1. **Save the `.blend` in the scene folder**, next to the `.obj`. `OUTPUT_PATH = "//animation.json"`
   is relative to the `.blend`.
2. Select the curve so it is the active object, or set `CURVE_NAME`.
3. Open the Scripting tab, open `preprocesing/blender_camera_path.py`, adjust the settings and run it (Alt+P).
4. Window → Toggle System Console shows the summary:
   ```
   [camera_path] curve length 42.10 units, 600 frames - 10.0s at 60 fps
   [camera_path] gait shares idle 10%, walk 46%, run 29%, sprint 15%
   [camera_path] max yaw change 7.8 deg/frame
   [camera_path] speed scale x0.84 (natural speeds at FRAME_COUNT ~504)
   [camera_path] wrote 600 frames to .../scenes/bedroom/animation.json
   ```

### 5. Preview

The script creates `AA_previewCam`, keyframed on frames `0 .. N-1`. These numbers match the
renderer's `frame_%06u.exr`. The preview uses the renderer's FOV (80° vertical), 16:9 aspect and clip range.
Press Numpad 0 and scrub the timeline to check that the camera stays inside the scene.

Re-running the script overwrites `animation.json` and the preview keys.

### 6. Check in the renderer

1. Load the scene. The log should print `loaded N frames from .../animation.json`.
2. Compare one frame, for example `frame_000200.exr`, with the Blender preview at frame 200.
   The viewpoints should match. A mirrored or 90°-rotated view means the `.obj` was exported
   with axes other than Forward −Z / Up Y.

## Settings

All settings are at the top of the script.

### General

| Setting | Default | Meaning |
|---|---|---|
| `CURVE_NAME` | `""` | Curve to use. Empty means the active object. |
| `REVERSE` | `False` | Walk the curve backwards. |
| `OUTPUT_PATH` | `"//animation.json"` | `//` means the folder of the saved `.blend`. |
| `FRAME_COUNT` | 600 | Number of frames, which is the number of EXRs the renderer records. |
| `FPS` | 60 | Virtual frame rate. Controls how far the camera moves per frame, and so the size of the motion vectors. |
| `SEED` | 1 | Different seed, different but similar-feeling path. |
| `UNITS_PER_METER` | 1.0 | 100 for centimetre scenes. |
| `FIT_TO_SPLINE` | `True` | Scale all speeds so the N frames cover exactly the whole curve. |
| `EYE_HEIGHT_M` | 0.0 | Added to the curve height. |

### Movement

| Setting | Default | Meaning |
|---|---|---|
| `GAITS` | idle 0 / walk 2 / run 5 / sprint 7.5 m/s | `(name, speed, share of time)`. Order matters: changes only go to neighbouring gaits. |
| `GAIT_HZ` | 0.12 | How often the gait tends to change. |
| `MIN_GAIT_S` | 1.0 | Shortest time in one gait. |
| `ACCEL_TIME` / `DECEL_TIME` | 0.6 / 0.35 s | Smoothing when speeding up and slowing down. |
| `SPEED_JITTER` | 0.06 | Small speed variation. |
| `BOB_M`, `STRIDE_M` | 0.035, 1.6 | Head bob at top speed, and step length. |
| `LATERAL_WANDER_M` | 0.0 | Sideways drift off the curve. Watch out near walls. |

### Looking around

| Setting | Default | Meaning |
|---|---|---|
| `LOOKAHEAD_M`, `TURN_TIME` | 1.5, 0.25 s | Heading aims ahead along the curve and follows corners with this delay. |
| `BASE_PITCH_DEG`, `SLOPE_FOLLOW` | −4, 0.5 | Default slight downward look; how much the camera follows ramps and stairs. |
| `ACTIVITY_HZ` | 0.05 | "Mood": alternates between calm stretches and looking around a lot. |
| `PAN_YAW_DEG`, `PAN_PITCH_DEG`, `PAN_HZ` | 35, 8, 0.08 | Slow pans. |
| `GLANCE_HZ` | 0.35 | About how many glances per second at full activity. |
| `GLANCE_YAW_DEG`, `GLANCE_PITCH_DEG` | 110, 20 | Maximum glance size. |
| `GLANCE_HOLD_S` | (0.4, 2.0) | How long a glance is held. |
| `GLANCE_TIME`, `RETURN_TIME` | 0.08, 0.35 s | Flick speed, and the turn back to the path. |
| `RUN_LOOK_DAMPING` | 0.6 | Less looking around when sprinting. |
| `JITTER_DEG` | 0.3 | Hand jitter. |
| `PITCH_LIMIT_DEG` | 75 | Pitch clamp. The renderer clamps at 89. |

### Preview

| Setting | Default | Meaning |
|---|---|---|
| `PREVIEW_CAMERA` | `"AA_previewCam"` | Empty string means no preview camera. |
| `RENDER_FOV_Y_DEG`, `RENDER_NEAR`, `RENDER_FAR`, `RENDER_RESOLUTION` | 80, 0.1, 10000, 1920×1080 | Must match the renderer's camera in `Engine.cpp`. |

### Tuning tips

- **Calmer footage:** raise `GLANCE_TIME` and lower `GLANCE_HZ` and `PAN_YAW_DEG`.
- **Harder footage (large motion vectors):** lower `FPS` to 30, or lower `GLANCE_TIME`.
- **Speeds look unnatural:** the summary prints a `speed scale`. Values far from 1 mean the
  curve is too long or too short for `FRAME_COUNT`. Set `FRAME_COUNT` to the suggested natural value.

## Output format

```json
{
  "generator": "camera_path.py",
  "curve": "CameraRoute",
  "seed": 1,
  "fps": 60,
  "frames": [
    {"position": [1.2, 1.7, -3.4], "yaw": -92.5, "pitch": -4.1},
    ...
  ]
}
```

- `position`: renderer coordinates (Y up). Converted from Blender as `(x, z, -y)`.
- `yaw`, `pitch`: degrees. Yaw 0 = +X and 90 = +Z; positive pitch looks up.
- The renderer only reads `frames`. The other keys are metadata.
- One `animation.json` per folder. All `.obj` files in that folder share it.

## Troubleshooting

| Message / symptom | Fix |
|---|---|
| `Select the movement curve (or set CURVE_NAME)` | The active object is not a curve. |
| `Save the .blend next to the scene .obj first ...` | The `.blend` is unsaved and `OUTPUT_PATH` is relative. |
| `Curve has bevel/extrude geometry` | Set Extrude and Bevel Depth to 0. |
| `Curve must contain exactly one spline` | Join the strokes (select two end points, then F), or split them into separate curves. |
| `Generated motion never moves` | All frames came out idle. Change `SEED` or lower the idle share in `GAITS`. |
| Camera inside walls in the preview | Move the curve away from walls, and lower `LATERAL_WANDER_M` and `BOB_M`. |
| Renderer view doesn't match the preview | Re-export the `.obj` with Forward −Z / Up Y. Check `UNITS_PER_METER` and the FOV / near / far settings. |
| Head bob invisible, or huge | `UNITS_PER_METER` is wrong. |


