#!/usr/bin/env python3
"""
visualise_pairs.py
------------------
Render OpenVLA contrastive-pair action deltas as coloured arrows on a tabletop
scene in Isaac Sim and save one PNG per pair.

Run with Isaac Sim's bundled interpreter:
    cd <isaac-sim install dir>
    ./python.sh /path/to/visualise_pairs.py --pairs ~/pairs.json --out ~/figs

Input: pairs.json produced by export_pairs.py. Per pair the translation parts
of action_a (blue), action_b (orange) and, when present, gt_vector (grey) are
drawn as shaft+head arrows from a common tabletop anchor.

Display scaling: predicted deltas are centimetre-scale and would be invisible
at true size, so within each pair both predicted arrows share one scale factor
chosen to make the longer of the two a fixed display length — relative
magnitude and direction are preserved, absolute length is not. The ground
truth is scaled independently (different convention; magnitudes are not
comparable). Scale factors are recorded in the output index for captioning.

Outputs: <out>/<pair_id>.png and <out>/index.csv (pair_id, instructions,
dx values, sign-flip flag, scale factors) for figure captions.
"""

import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", required=True, help="path to pairs.json")
parser.add_argument("--out", default="./figs", help="output directory")
parser.add_argument("--gui", action="store_true", help="render with the GUI (default headless)")
parser.add_argument("--limit", type=int, default=0, help="render only the first N pairs")
parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720])
args = parser.parse_args()

# ---- 1. The app must exist before any omni/isaacsim import ------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": not args.gui})

# ---- 2. Version-tolerant imports (5.x namespace, 4.x fallback) ---------------
import json
import numpy as np

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, VisualCylinder, VisualCone
    from isaacsim.sensors.camera import Camera
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid, VisualCylinder, VisualCone
    from omni.isaac.sensor import Camera

from PIL import Image

COLOR_A = np.array([0.10, 0.45, 1.00])   # instruction A - blue
COLOR_B = np.array([1.00, 0.55, 0.10])   # instruction B - orange
COLOR_GT = np.array([0.45, 0.45, 0.45])  # ground truth  - grey
DISPLAY_LEN = 0.25       # metres: display length of the longer arrow in a pair
SHAFT_RADIUS = 0.006
HEAD_RADIUS = 0.016
HEAD_LEN = 0.05
SETTLE_FRAMES = 60       # first pair also absorbs shader compilation


def quat_from_z(direction):
    """Quaternion (w, x, y, z) rotating +Z onto `direction`."""
    z = np.asarray(direction, dtype=float)
    z = z / (np.linalg.norm(z) + 1e-12)
    zaxis = np.array([0.0, 0.0, 1.0])
    v = np.cross(zaxis, z)
    c = float(np.dot(zaxis, z))
    if np.linalg.norm(v) < 1e-8:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1.0, 0, 0])
    v = v / np.linalg.norm(v)
    ang = np.arccos(np.clip(c, -1.0, 1.0))
    s = np.sin(ang / 2.0)
    return np.array([np.cos(ang / 2.0), v[0] * s, v[1] * s, v[2] * s])


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """World orientation for a USD camera (looks down -Z) at `eye` facing `target`."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-12)          # camera +Z points away from target
    x = np.cross(np.asarray(up, float), z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w < 1e-8:
        return np.array([1.0, 0, 0, 0])
    return np.array([w,
                     (m[2, 1] - m[1, 2]) / (4 * w),
                     (m[0, 2] - m[2, 0]) / (4 * w),
                     (m[1, 0] - m[0, 1]) / (4 * w)])


class ArrowSet:
    """Creates and re-poses a fixed set of arrow prims instead of re-adding
    prims per pair — prim creation mid-run is slow and leak-prone."""

    def __init__(self, world, names_colors):
        self.parts = {}
        far = np.array([0.0, 0.0, -10.0])       # parked out of view
        for name, color in names_colors:
            shaft = world.scene.add(VisualCylinder(
                prim_path=f"/World/{name}_shaft", name=f"{name}_shaft",
                position=far, radius=SHAFT_RADIUS, height=0.01, color=color))
            head = world.scene.add(VisualCone(
                prim_path=f"/World/{name}_head", name=f"{name}_head",
                position=far, radius=HEAD_RADIUS, height=HEAD_LEN, color=color))
            self.parts[name] = (shaft, head)

    def pose(self, name, start, delta):
        shaft, head = self.parts[name]
        d = np.asarray(delta, dtype=float)
        length = float(np.linalg.norm(d))
        if length < 1e-5:
            self.hide(name)
            return
        u = d / length
        shaft_len = max(length - HEAD_LEN, 0.01)
        q = quat_from_z(u)
        shaft.set_world_pose(np.asarray(start) + u * shaft_len / 2.0, q)
        shaft.set_local_scale(np.array([1.0, 1.0, shaft_len / 0.01]))
        head.set_world_pose(np.asarray(start) + u * (shaft_len + HEAD_LEN / 2.0), q)

    def hide(self, name):
        shaft, head = self.parts[name]
        far = np.array([0.0, 0.0, -10.0])
        shaft.set_world_pose(far, np.array([1.0, 0, 0, 0]))
        head.set_world_pose(far + np.array([0.2, 0, 0]), np.array([1.0, 0, 0, 0]))


def main():
    with open(os.path.expanduser(args.pairs)) as f:
        pairs = json.load(f)
    if args.limit:
        pairs = pairs[: args.limit]
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.scene.add(FixedCuboid(
        prim_path="/World/table", name="table",
        position=np.array([0.40, 0.0, 0.05]),
        scale=np.array([0.60, 1.00, 0.10]),
        color=np.array([0.85, 0.85, 0.85])))

    anchor_default = np.array([0.40, 0.0, 0.20])
    cam_eye = np.array([1.35, 1.35, 0.95])
    camera = Camera(
        prim_path="/World/cam",
        position=cam_eye,
        orientation=look_at_quat(cam_eye, anchor_default),
        resolution=tuple(args.resolution))

    arrows = ArrowSet(world, [("arrow_a", COLOR_A), ("arrow_b", COLOR_B),
                              ("arrow_gt", COLOR_GT)])
    world.reset()
    camera.initialize()

    index_rows = []
    for i, p in enumerate(pairs):
        start = np.asarray(p.get("start_pos", anchor_default), dtype=float)
        ta = np.asarray(p["action_a"][:3], dtype=float)
        tb = np.asarray(p["action_b"][:3], dtype=float)

        longest = max(np.linalg.norm(ta), np.linalg.norm(tb))
        scale_pred = (DISPLAY_LEN / longest) if longest > 1e-6 else 0.0
        arrows.pose("arrow_a", start, ta * scale_pred)
        arrows.pose("arrow_b", start, tb * scale_pred)

        scale_gt = 0.0
        gt = p.get("gt_vector")
        if gt is not None:
            tg = np.asarray(gt[:3], dtype=float)
            n = np.linalg.norm(tg)
            scale_gt = (DISPLAY_LEN / n) if n > 1e-6 else 0.0
            arrows.pose("arrow_gt", start, tg * scale_gt)
        else:
            arrows.hide("arrow_gt")

        for _ in range(SETTLE_FRAMES if i == 0 else 15):
            world.step(render=True)

        rgba = camera.get_rgba()
        png = os.path.join(out_dir, f"{p['pair_id']}.png")
        Image.fromarray(rgba[..., :3].astype(np.uint8)).save(png)

        dxa, dxb = p["action_a"][0], p["action_b"][0]
        flip = dxa * dxb < 0
        print(f"[{i + 1}/{len(pairs)}] {p['pair_id']}  dx_a={dxa:+.4f} "
              f"dx_b={dxb:+.4f}  x-flip={'YES' if flip else 'no'}  -> {png}")
        index_rows.append({
            "pair_id": p["pair_id"], "scene_id": p["scene_id"],
            "instr_a": p["instr_a"], "instr_b": p["instr_b"],
            "dx_a": dxa, "dx_b": dxb, "x_sign_flip": flip,
            "pred_display_scale": round(scale_pred, 2),
            "gt_display_scale": round(scale_gt, 2),
            "image": os.path.basename(png)})

    index_path = os.path.join(out_dir, "index.csv")
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)
    print(f"wrote {len(index_rows)} figures + {index_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
