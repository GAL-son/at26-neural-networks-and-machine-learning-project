"""
Camera path generator for AI anti-aliaser renderer;

Workflow (Blender 4.x/ 5.x)
    1. Import scene .obj (Forward -Z, Up Y)
    2. Fix scene orientation if needed (reexport .obj)
    3. Draw one curve for the route (eye level of the floor with EYE_HEIGHT_M set)
    4. Select the curve, open this script in Scripting tab, adjust settings and run the script

Script generates animation.json sidecar file for renderer camera path.
"""

import json
import math
import os
import random

import bpy
import numpy as np
from mathutils import Vector, noise  # Blender specific

CURVE_NAME = ""
REVERSE = False
OUTPUT_PATH = "//animation.json"
FRAME_COUNT = 600
FPS = 60
SEED = 1

UNITS_PER_METER = 1.0
FIT_TO_SPLINE = True
EYE_HEIGHT_M = 0.0

GAITS = [
    ("idle", 0.0, 0.10),
    ("walk", 2.0, 0.45),
    ("run", 5.0, 0.30),
    ("sprint", 7.5, 0.15),
]
GAIT_HZ = 0.12
MIN_GAIT_S = 1.0
ACCEL_TIME = 0.6
DECEL_TIME = 0.35
SPEED_JITTER = 0.06

LOOKAHEAD_M = 1.5
TURN_TIME = 0.25
BASE_PITCH_DEG = -0.4
SLOPE_FOLLOW = 0.5

ACTIVITY_HZ = 0.05
PAN_YAW_DEG = 35.0
PAN_PITCH_DEG = 8.0
PAN_HZ = 0.08

GLANCE_HZ = 0.35
GLANCE_YAW_DEG = 110.0
GLANCE_PITCH_DEG = 20.0
GLANCE_HOLD_S = (0.4, 2.0)
GLANCE_TIME = 0.08
RETURN_TIME = 0.35
RUN_LOOK_DAMPING = 0.6

JITTER_DEG = 0.3
BOB_M = 0.035
STRIDE_M = 1.6
LATERAL_WANDER_M = 0.0
PITCH_LIMIT_DEG = 75.0

PREVIEW_CAMERA = "AA_previewCam"
RENDER_FOV_Y_DEG = 80.0
RENDER_RESOLUTION = (1920, 1080)

def lerp(a, b, t):
    return a + (b - a) * t

def smooth_damp(current, target, velocity, smooth_time, dt):
    omega = 2.0 / max(smooth_time, 1e-4)
    x = omega * dt
    decay = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    temp = (velocity - omega * change) * dt
    velocity = (velocity - omega * temp) * decay
    return target + (change + temp) * decay, velocity

def fbm_track(rng, t, hz, octaves=3):
    ox, oy, oz = (rng.uniform(-100, 100) for _ in range(3))
    out = np.zeros(len(t))
    amp, freq, norm = 1.0, hz, 0.0
    for octave in range(octaves):
        out += amp * np.array([noise.noise(Vector((ox + ti * freq, oy + 17.3 + octave, oz))) for ti in t])
        norm += amp
        amp *= 0.5
        freq *= 2.0
    out /= norm
    scale = np.percentile(np.abs(out), 98)
    return np.clip(out / scale, -1.0, 1.0) if scale > 1e-6 else out

def rank_uniform(x):
    return (np.argsort(np.argsort(x)) + 0.5) / len(x)

def enforce_min_dwell(states, min_len):
    out = states.copy()
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if i > 0 and j - i < min_len:
            out[i:j] = out[i - 1]
        i = j
    return out

def forward_fill(values, bad):
    if bad.all():
        return np.zeros_like(values)
        
    out = values.copy()
    last = out[np.argmax(~bad)]

    for i in range(len(out)):
        if bad[i]:
            out[i] = last
        else:
            last = out[i]
    return out

def sample_curve(obj):
    """World-space polyline of the evaluated curve (any spline type), in curve order."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    
    try:
        if len(mesh.polygons) > 0:
            raise RuntimeError("Curve has bevel/extrude geometry: set Extrude and Bevel Depth to 0")

        count = len(mesh.vertices)
        adjacency = [[] for _ in range(count)]
        for edge in mesh.edges:
            a, b = edge.vertices
            adjacency[a].append(b)
            adjacency[b].append(a)

        ends = [i for i in range(count) if len(adjacency[i]) == 1]
        if count < 2 or any(len(a) > 2 for a in adjacency) or len(ends) not in (0, 2):
            raise RuntimeError("Curve must contain exactly one spline")

        cyclic = not ends
        start = ends[0] if ends else 0
        order, prey, cur = [start], -1, start
        
        while True:
            nxt = [v for v in adjacency[cur] if v != prey]
            if not nxt or nxt[0] == start:
                break
            prey, cur = cur, nxt[0]
            order.append(cur)
        
        if len(order) != count:
            raise RuntimeError("Curve must contain exactly one spline")

        points = np.array([tuple(obj.matrix_world @ mesh.vertices[i].co) for i in order])
        
    finally:
        evaluated.to_mesh_clear()
    
    return points, cyclic

class ArcPath:
    """Polyline parameterised by arc length."""

    def __init__(self, points, cyclic):
        pts = np.asarray(points, dtype=np.float64)
        if cyclic:
            pts = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.points = pts[np.concatenate([[True], seg > 1e-9])]
        self.cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))])
        self.length = float(self.cum[-1])
        self.cyclic = cyclic
        if len(self.points) < 2 or self.length <= 0.0:
            raise RuntimeError("Curve has zero length")

    def at(self, s):
        s = np.mod(s, self.length) if self.cyclic else s
        return np.stack([np.interp(s, self.cum, self.points[:, k]) for k in range(3)], axis=-1)

def generate(path):
    n, dt, upm = FRAME_COUNT, 1.0 / FPS, UNITS_PER_METER
    t = np.arange(n) * dt
    rng = random.Random(SEED)

    def track(hz, octaves=3):
        return fbm_track(rng, t, hz, octaves)
    
    # --- speed: few gaits from quantised noise, smoothed by a spring
    weights = np.array([g[2] for g in GAITS], dtype=np.float64)
    edges = np.cumsum(weights) / weights.sum()
    gait = np.searchsorted(edges, rank_uniform(track(GAIT_HZ, 1)), side="right").clip(0, len(GAITS) - 1)
    gait = enforce_min_dwell(gait, int(MIN_GAIT_S * FPS))
    target = np.array([GAITS[g][1] for g in gait])
    jitter = track(1.5)

    speed = np.empty(n)
    v, dv = target[0], 0.0
    for i in range(n):
        v, dv = smooth_damp(v, target[i], dv, ACCEL_TIME if target[i] > v else DECEL_TIME, dt)
        v = max(v, 0.0)
        speed[i] = v * (1.0 + SPEED_JITTER * jitter[i])

    travel = np.concatenate([[0.0], np.cumsum(speed[:-1] * dt)]) * upm
    scale = 1.0
    if FIT_TO_SPLINE:
        if travel[-1] <= 1e-9:
            raise RuntimeError("Generated motion never moves (all idle?) - change SEED or GAITS")
        scale = path.length / travel[-1]
        travel *= scale
    elif not path.cyclic and travel[-1] > path.length:
        print(f"[camera_path] warning: path ends at frame {np.argmax(travel > path.length)}, camera stops there")

    max_speed = max(g[1] for g in GAITS) or 1.0
    run_t = np.clip(speed / max_speed, 0.0, 1.0)
    calm = 1.0 - RUN_LOOK_DAMPING * run_t

    # --- heading along the curve (Blender XY plane), lazily following corners
    la = LOOKAHEAD_M * upm
    d = path.at(travel + la) - path.at(travel - 0.25 * la)
    horiz = np.hypot(d[:, 0], d[:, 1])
    bad = horiz < 1e-6 * max(la, 1e-6)
    heading = forward_fill(np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0]))), bad)
    slope = forward_fill(np.degrees(np.arctan2(d[:, 2], horiz)), bad)
    
    heading_s = np.empty(n)
    h, dh = heading[0], 0.0
    for i in range(n):
        h, dh = smooth_damp(h, heading[i], dh, TURN_TIME, dt)
        heading_s[i] = h

    # --- slow pans, modulated by mood
    activity = 0.5 + 0.5 * track(ACTIVITY_HZ, 2)
    pan_gain = 0.3 + 0.7 * activity
    pan_yaw = PAN_YAW_DEG * track(PAN_HZ) * pan_gain * calm
    pan_pitch = PAN_PITCH_DEG * track(PAN_HZ * 1.3) * pan_gain
    
    # --- glances
    trigger = track(GLANCE_HZ, 2)
    g_yaw, g_pitch, g_hold, g_speed = (track(2.3, 1) for _ in range(4))
    threshold = 0.6 - 0.6 * activity
    look_yaw, look_pitch = np.empty(n), np.empty(n)
    ty = tp = y = p = vy = vp = 0.0
    tau, release = RETURN_TIME, -1
    
    for i in range(n):
        if i > 0 and trigger[i - 1] - threshold[i - 1] < 0.0 <= trigger[i] - threshold[i]:
            magnitude = 0.35 + 0.65 * abs(g_yaw[i])
            ty = math.copysign(magnitude, g_yaw[i]) * GLANCE_YAW_DEG * calm[i]
            tp = g_pitch[i] * GLANCE_PITCH_DEG
            release = i + int(lerp(*GLANCE_HOLD_S, 0.5 + 0.5 * g_hold[i]) * FPS)
            tau = GLANCE_TIME * (0.6 + 0.8 * (0.5 + 0.5 * g_speed[i]))
        elif i == release:
            ty = tp = 0.0
            tau = RETURN_TIME
        
        y, vy = smooth_damp(y, ty, vy, tau, dt)
        p, vp = smooth_damp(p, tp, vp, tau, dt)
        look_yaw[i], look_pitch[i] = y, p

    yaw_b = heading_s - pan_yaw + look_yaw + JITTER_DEG * track(4.0, 2)
    pitch = np.clip(BASE_PITCH_DEG + SLOPE_FOLLOW * slope + pan_pitch + look_pitch + JITTER_DEG * track(4.0, 2), -PITCH_LIMIT_DEG, PITCH_LIMIT_DEG)
    
    # --- position
    hr = np.radians(heading_s)
    right = np.stack([np.sin(hr), -np.cos(hr), np.zeros(n)], axis=-1)
    phase = 2.0 * math.pi * (travel / upm) / STRIDE_M
    bob = BOB_M * upm * run_t
    lateral = LATERAL_WANDER_M * upm * track(0.1, 2) + 0.5 * bob * np.sin(phase)
    pos = path.at(travel)
    pos[:, 2] += EYE_HEIGHT_M * upm + bob * np.sin(2.0 * phase)
    pos += right * lateral[:, None]
    
    shares = {g[0]: float(np.mean(gait == k)) for k, g in enumerate(GAITS)}
    rot_speed = np.abs(np.diff(yaw_b)).max() if n > 1 else 0.0
    print(f"[camera_path] curve length {path.length:.2f} units, {n} frames - {n / FPS:.1f}s at {FPS:g} fps")
    print(f"[camera_path] gait shares {', '.join(f'{k} {v:.0%}' for k, v in shares.items())}")
    print(f"[camera_path] max yaw change {rot_speed:.1f} deg/frame")
    if FIT_TO_SPLINE:
        print(f"[camera_path] speed scale x{scale:.2f} (natural speeds at FRAME_COUNT ~{int(n * scale)})")
    
    return pos, yaw_b, pitch

def resolve_output():
    if OUTPUT_PATH.startswith("//") and not bpy.data.filepath:
        raise RuntimeError("Save the .blend next to the scene .obj first, or use an absolute OUTPUT_PATH")
    return bpy.path.abspath(OUTPUT_PATH)

def write_json(file, pos, yaw_b, pitch, curve_name):
    frames = []
    for i in range(len(pos)):
        bx, by, bz = pos[i]
        frames.append({
            "position": [round(bx, 5), round(bz, 5), round(-by, 5)],
            "yaw": round(-float(yaw_b[i]), 4),
            "pitch": round(float(pitch[i]), 4),
        })
        
    header = {"generator": "camera_path.py", "curve": curve_name, "seed": SEED, "fps": FPS}
    os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
    
    with open(file, "w", encoding="utf-8") as f:
        f.write("{\n")
        for key, value in header.items():
            f.write(f" {json.dumps(key)}: {json.dumps(value)},\n")
        f.write(' "frames": [\n')
        f.write(",\n".join(" " + json.dumps(frame) for frame in frames))
        f.write("\n ]\n}\n")
        
    print(f"[camera_path] wrote {len(frames)} frames to {file}")

def build_preview(pos, yaw_b, pitch):
    scene = bpy.context.scene
    data = bpy.data.cameras.get(PREVIEW_CAMERA) or bpy.data.cameras.new(PREVIEW_CAMERA)
    cam = bpy.data.objects.get(PREVIEW_CAMERA) or bpy.data.objects.new(PREVIEW_CAMERA, data)
    if cam.name not in scene.collection.all_objects:
        scene.collection.objects.link(cam)
        
    data.sensor.fit = "VERTICAL"
    data.angle_y = math.radians(RENDER_FOV_Y_DEG)
    data.clip_start, data.clip_end = 0.1, 10000.0
    
    cam.animation_data_clear()
    cam.rotation_mode = "XYZ"
    for i in range(len(pos)):
        cam.location = Vector(pos[i])
        cam.rotation_euler = (math.radians(90.0 + pitch[i]), 0.0, math.radians(yaw_b[i] - 90.0))
        cam.keyframe_insert("location", frame=i)
        cam.keyframe_insert("rotation_euler", frame=i)
        
    scene.camera = cam
    scene.frame_start = 0
    scene.frame_end = len(pos) - 1
    scene.render.fps = int(round(FPS))
    scene.render.resolution_x, scene.render.resolution_y = RENDER_RESOLUTION

def main():
    curve = bpy.data.objects.get(CURVE_NAME) if CURVE_NAME else bpy.context.active_object
    if curve is None or curve.type != "CURVE":
        raise RuntimeError("Select the movement curve (or set CURVE_NAME)")
    
    out_file = resolve_output()
    points, cyclic = sample_curve(curve)
    if REVERSE:
        points = points[::-1]
    pos, yaw_b, pitch = generate(ArcPath(points, cyclic))
    write_json(out_file, pos, yaw_b, pitch, curve.name)
    
    if PREVIEW_CAMERA:
        build_preview(pos, yaw_b, pitch)

main()