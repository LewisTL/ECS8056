#!/usr/bin/env python3
"""
rollout_pairs.py
----------------
Franka executes OpenVLA's predicted deltas per contrastive pair (A, home, B),
recorded as one self-explanatory MP4 per pair:

  * LEFT PANEL:  the real BridgeData frame the model saw, plus the instruction
    currently executing (blue = A, orange = B).
  * RIGHT PANEL: the Isaac Sim rollout, with a persistent end-effector trail
    per role (blue/orange spheres) so both trajectories remain visible for
    direct comparison by the end of the clip.

Usage (terminal inside the DCV desktop for live view, or --headless):
    cd /opt/IsaacSim
    ./python.sh ~/rollout_pairs.py --pairs ~/pairs.json --out ~/rollouts \
        --frames-root ~/bridge_frames

List available pair_id / scene_id values (no Isaac launch):
    ./python.sh ~/rollout_pairs.py --pairs ~/pairs.json --list

Render one pair by scene_id or pair_id (prefix and epNN shorthand accepted):
    ./python.sh ~/rollout_pairs.py --pairs ~/pairs.json --out ~/rollouts \
        --frames-root ~/bridge_frames --pair-id ep000000

Grasp-frame placement pairs (post-grasp observation). Use `--frame` (singular);
`--frames` is not a flag and used to be misread as `--frames-root`:

```
./python.sh ~/rollout_pairs.py --pairs ~/pairs.json --out ~/rollouts \
    --frames-root ~/bridge_frames --frame grasp --pair-id ep000000
```

--frames-root points at the cached frames (copy once with e.g.
`rclone copy gdrive:openvla_cache/v2/bridge/frames ~/bridge_frames`, or
`gdrive:openvla_cache/v2/constructed/frames` for the constructed stimuli).
Pairs whose image can't be found still render, without the photo.

Scale caveat: OpenVLA returns one 7-DoF action per instruction, not a full
trajectory. Each pair is scaled so the longer role reaches `--display-len`
metres, then the arm moves from a fixed workspace anchor along that mapped
translation vector. Direction is faithful within the display cap; absolute
distance is not. Use `--scale` only to override with a fixed multiplier.
"""

import argparse
import json
import os
import re
import textwrap

try:
    from export_pairs import map_action, BRIDGE_TO_ISAAC
except ImportError:
    map_action = None
    BRIDGE_TO_ISAAC = None

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--pairs", required=True)
ap.add_argument("--out", default="./rollouts")
ap.add_argument("--frames-root", default=None,
                help="dir containing the cached BridgeData PNGs")
ap.add_argument("--scene-id", type=int, nargs="+", default=None,
                help="render only pairs whose scene_id is in this list")
ap.add_argument("--pair-id", nargs="+", default=None,
                help="render matching pair_id values: exact match, prefix, "
                     "or epNN / NN shorthand (e.g. ep70 -> ep000070_*)")
ap.add_argument("--frame", nargs="+", choices=("initial", "grasp"), default=None,
                help="render only pairs at these observation frames "
                     "(use --frame grasp, not --frames)")
ap.add_argument("--list", action="store_true",
                help="print available pair_id / scene_id / frame values and exit "
                     "(does not launch Isaac Sim)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--display-len", type=float, default=0.12,
                help="max displacement (m) per pair after per-pair scaling "
                     "(matches the arrow-figure idea)")
ap.add_argument("--scale", type=float, default=None,
                help="fixed delta multiplier; overrides --display-len when set")
ap.add_argument("--anchor", type=float, nargs=3, default=[0.45, 0.0, 0.35],
                help="workspace pose the arm approaches before each role")
ap.add_argument("--min-ee-z", type=float, default=0.20,
                help="minimum end-effector height (m) when clamping targets")
ap.add_argument("--approach-steps", type=int, default=40,
                help="sim steps to reach the workspace anchor")
ap.add_argument("--steps", type=int, default=80,
                help="sim steps for the interpolated delta motion")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--trail-every", type=int, default=4,
                help="drop a trail sphere every N sim steps")
ap.add_argument("--headless", action="store_true")
args = ap.parse_args()


def _pair_id_patterns(raw: str):
    """Expand a user selector into matchable pair_id prefixes / exact ids."""
    raw = raw.strip()
    out = {raw}
    m = re.fullmatch(r"(?:ep)?(\d+)", raw, flags=re.IGNORECASE)
    if m:
        out.add(f"ep{int(m.group(1)):06d}")
    return out


def _pair_id_matches(pair_id: str, selectors) -> bool:
    for sel in selectors:
        for pat in _pair_id_patterns(sel):
            if pair_id == pat or pair_id.startswith(pat + "_") or pair_id.startswith(pat):
                return True
    return False


def pair_frame(p):
    """Observation frame for a pair record; older exports default to initial."""
    return str(p.get("frame", "initial"))


def output_stem(p):
    """Unique video basename: pair_id plus frame so initial/grasp do not collide."""
    return f"{p['pair_id']}_{pair_frame(p)}"


def load_and_filter_pairs(path, scene_ids=None, pair_ids=None, frames=None,
                          limit=0):
    """Load pairs.json and apply optional scene / pair / frame filters."""
    with open(os.path.expanduser(path)) as f:
        all_pairs = json.load(f)
    pairs = list(all_pairs)
    if scene_ids is not None:
        wanted = set(scene_ids)
        pairs = [p for p in pairs if int(p["scene_id"]) in wanted]
    if pair_ids is not None:
        pairs = [p for p in pairs if _pair_id_matches(p["pair_id"], pair_ids)]
    if frames is not None:
        wanted_frames = set(frames)
        pairs = [p for p in pairs if pair_frame(p) in wanted_frames]
    if limit:
        pairs = pairs[:limit]
    return all_pairs, pairs


def format_pair_index(pairs):
    lines = [
        f"  {p['pair_id']}  frame={pair_frame(p)}  (scene_id={p['scene_id']})"
        for p in pairs
    ]
    return "\n".join(lines) if lines else "  (none)"


all_pairs, pairs = load_and_filter_pairs(
    args.pairs, args.scene_id, args.pair_id, args.frame, args.limit)

if args.list:
    print(f"[rollout] {len(all_pairs)} pairs in {args.pairs}")
    print(format_pair_index(all_pairs))
    raise SystemExit(0)

if not pairs:
    filtered = (args.scene_id is not None or args.pair_id is not None
                or args.frame is not None or args.limit)
    from collections import Counter
    frame_counts = Counter(pair_frame(p) for p in all_pairs)
    missing_field = sum(1 for p in all_pairs if "frame" not in p)
    msg = ["[rollout] no pairs matched the given filters "
           f"(scene-id={args.scene_id}, pair-id={args.pair_id}, "
           f"frame={args.frame}, limit={args.limit})",
           f"[rollout] frame counts in file: {dict(frame_counts)} "
           f"({missing_field} records lack a 'frame' field and default to "
           f"'initial')",
           f"[rollout] {len(all_pairs)} pairs available:"]
    msg.append(format_pair_index(all_pairs[:40]))
    if len(all_pairs) > 40:
        msg.append(f"  ... and {len(all_pairs) - 40} more")
    if args.frame and "grasp" in args.frame and frame_counts.get("grasp", 0) == 0:
        msg.append(
            "[rollout] this pairs.json has no grasp-frame records. Grasp frames "
            "belong to placement scenes, which the current harvest does not cache; "
            "re-export from Notebook 04 against a log that contains them. "
            "Also use --frame grasp (singular), not --frames.")
    if filtered:
        msg.append("[rollout] re-run with --list, or pass a pair_id / scene_id / "
                   "frame from the list above")
    raise SystemExit("\n".join(msg))

print(f"[rollout] rendering {len(pairs)}/{len(all_pairs)} pairs: "
      f"{', '.join(output_stem(p) for p in pairs)}")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualSphere
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController)
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.core import World
    from omni.isaac.core.objects import VisualSphere
    from omni.isaac.franka import Franka
    from omni.isaac.franka.controllers import RMPFlowController
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.viewports import set_camera_view

from PIL import Image, ImageDraw, ImageFont

if map_action is None:
    BRIDGE_TO_ISAAC = np.eye(3)

    def map_action(values):
        """Map Bridge action translation into Isaac world axes (fallback copy)."""
        a = np.asarray(values, dtype=float).copy()
        a[:3] = BRIDGE_TO_ISAAC @ a[:3]
        return a

CAM_EYE = [1.9, 1.9, 1.5]
CAM_TARGET = [0.45, 0.0, 0.35]
COLOR_A_RGB = (26, 115, 255)
COLOR_B_RGB = (255, 140, 26)
COLOR_A = np.array(COLOR_A_RGB) / 255.0
COLOR_B = np.array(COLOR_B_RGB) / 255.0
PANEL_W = 640            # left panel width; sim frame is 1280x720 -> 1920x720
SIM_W, SIM_H = 1280, 720
TRAIL_POOL = 80          # spheres per role
PARK = np.array([0.0, 0.0, -10.0])
MAX_REACH = 0.22          # clamp displacement from anchor (metres)


def pair_display_scale(mapped_a, mapped_b, display_len, fixed_scale=None):
    """Return the multiplier applied to mapped translation deltas for this pair."""
    if fixed_scale is not None:
        return fixed_scale
    ta = np.asarray(mapped_a[:3], dtype=float)
    tb = np.asarray(mapped_b[:3], dtype=float)
    longest = max(float(np.linalg.norm(ta)), float(np.linalg.norm(tb)))
    return (display_len / longest) if longest > 1e-6 else 0.0


def log_pair_deltas(p):
    """Print Bridge and Isaac deltas for both roles (model output, not injected)."""
    ma = map_action(p["action_a"])[:3]
    mb = map_action(p["action_b"])[:3]
    ba = np.asarray(p["action_a"][:3], dtype=float)
    bb = np.asarray(p["action_b"][:3], dtype=float)
    print(f"    bridge A {np.round(ba, 6)}  B {np.round(bb, 6)}  "
          f"diff {np.round(bb - ba, 6)}")
    print(f"    isaac  A {np.round(ma, 6)}  B {np.round(mb, 6)}  "
          f"diff {np.round(mb - ma, 6)}")
    for axis, name in enumerate("xyz"):
        flip = ma[axis] * mb[axis] < 0
        if abs(ma[axis]) > 1e-6 or abs(mb[axis]) > 1e-6:
            print(f"    {name}-sign flip in logged actions: "
                  f"{'yes' if flip else 'no'}")


def clamp_ee_target(pos, anchor, min_z, max_reach=MAX_REACH):
    """Keep the target above the floor and within reach of the anchor."""
    pos = np.asarray(pos, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    pos[2] = max(pos[2], min_z)
    offset = pos - anchor
    dist = float(np.linalg.norm(offset))
    if dist > max_reach:
        pos = anchor + offset * (max_reach / dist)
        pos[2] = max(pos[2], min_z)
    return pos


def move_ee(controller, franka, world, target_pos, target_quat, steps, grab=None):
    """Hold one Cartesian target for a fixed number of sim steps."""
    target_pos = np.asarray(target_pos, dtype=float)
    for _ in range(steps):
        action = controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=target_quat)
        franka.apply_action(action)
        world.step(render=True)
        if grab is not None:
            grab()


def move_ee_linear(controller, franka, world, start, end, target_quat, steps,
                   grab=None, trail=None, trail_every=4):
    """Interpolate the end effector along a straight segment (smoother than one
    distant RMPFlow target)."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    for s in range(steps):
        alpha = (s + 1) / steps
        waypoint = start + alpha * (end - start)
        action = controller.forward(
            target_end_effector_position=waypoint,
            target_end_effector_orientation=target_quat)
        franka.apply_action(action)
        world.step(render=True)
        if trail is not None and s % trail_every == 0:
            ee_pos, _ = franka.end_effector.get_world_pose()
            trail.drop(ee_pos)
        if grab is not None:
            grab()


class Trail:
    """Pre-created sphere pool per role; spheres are parked below ground and
    moved onto the end-effector path as it executes."""

    def __init__(self, world, name, color):
        self.spheres = [world.scene.add(VisualSphere(
            prim_path=f"/World/trail_{name}_{k}", name=f"trail_{name}_{k}",
            position=PARK + np.array([0.05 * k, 0.0, 0.0]),
            radius=0.008, color=color)) for k in range(TRAIL_POOL)]
        self.i = 0

    def drop(self, pos):
        if self.i < len(self.spheres):
            self.spheres[self.i].set_world_pose(np.asarray(pos, dtype=float),
                                                np.array([1.0, 0, 0, 0]))
            self.i += 1

    def clear(self):
        for k, s in enumerate(self.spheres):
            s.set_world_pose(PARK + np.array([0.05 * k, 0.0, 0.0]),
                             np.array([1.0, 0, 0, 0]))
        self.i = 0


def scene_image_candidates(p, frames_root):
    """Return filesystem paths to try for the pair's BridgeData photo."""
    root = os.path.expanduser(frames_root)
    names = []
    if p.get("image_path"):
        rel = str(p["image_path"])
        names.extend([rel, os.path.basename(rel)])
        # Common layout: frames-root is already the frames/ directory.
        if rel.startswith("frames" + os.sep) or rel.startswith("frames/"):
            names.append(rel.split("/", 1)[-1].split("\\", 1)[-1])
    # Fallback from scene_id + observation frame when image_path is missing
    # (older exports only attached image_path when GT columns were complete).
    try:
        ep = int(p["scene_id"])
        frame = pair_frame(p)
        stem = f"ep_{ep:06d}_grasp.png" if frame == "grasp" else f"ep_{ep:06d}.png"
        names.extend([stem, os.path.join("frames", stem)])
    except (KeyError, TypeError, ValueError):
        pass
    cands = []
    seen = set()
    for name in names:
        path = os.path.join(root, name)
        if path not in seen:
            seen.add(path)
            cands.append(path)
    return cands


def load_scene_image(p):
    """Locate and load the pair's BridgeData frame, or None."""
    if not args.frames_root:
        return None
    cands = scene_image_candidates(p, args.frames_root)
    for cand in cands:
        if os.path.exists(cand):
            return Image.open(cand).convert("RGB")
    if cands:
        print(f"    note: scene image not found; tried {cands[0]}"
              + (f" (+{len(cands) - 1} more)" if len(cands) > 1 else ""))
    return None


def make_panel(scene_img, instr, role, pair_id, frame="initial"):
    """Left panel: scene photo + instruction banner coloured by role."""
    panel = Image.new("RGB", (PANEL_W, SIM_H), (24, 24, 24))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    if scene_img is not None:
        img = scene_img.copy()
        img.thumbnail((PANEL_W - 20, 440))
        panel.paste(img, ((PANEL_W - img.width) // 2, 10))
        y = 10 + img.height + 14
        draw.text((10, y), f"model input ({frame} frame)", fill=(150, 150, 150),
                  font=font)
        y += 22
    else:
        draw.text((10, 10), "(scene image unavailable)", fill=(150, 150, 150),
                  font=font)
        y = 40

    color = COLOR_A_RGB if role == "a" else COLOR_B_RGB
    draw.rectangle([0, y, PANEL_W, y + 26], fill=color)
    draw.text((10, y + 6), f"executing instruction {role.upper()}",
              fill=(0, 0, 0), font=font)
    y += 34
    for line in textwrap.wrap(instr, width=70)[:4]:
        draw.text((10, y), line, fill=(235, 235, 235), font=font)
        y += 16
    draw.text((10, SIM_H - 22), f"{pair_id}  [{frame}]", fill=(120, 120, 120),
              font=font)
    return panel


def compose(sim_rgb, panel):
    canvas = Image.new("RGB", (PANEL_W + SIM_W, SIM_H))
    canvas.paste(panel, (0, 0))
    canvas.paste(Image.fromarray(sim_rgb), (PANEL_W, 0))
    return np.asarray(canvas)


def write_video(frames, path, fps):
    if not frames:
        return
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
        print(f"    saved {path} ({len(frames)} frames)")
    except Exception as e:
        seq_dir = path.rsplit(".", 1)[0] + "_frames"
        os.makedirs(seq_dir, exist_ok=True)
        for k, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(seq_dir, f"{k:04d}.png"))
        print(f"    imageio unavailable ({e}); wrote PNGs to {seq_dir}")
        print(f"    stitch: ffmpeg -framerate {fps} -i {seq_dir}/%04d.png "
              f"-pix_fmt yuv420p {path}")


def main():
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    # No tabletop cuboid: FixedCuboid is a physics collider and the scaled
    # OpenVLA deltas often drive the end-effector into it, producing jerky
    # RMPFlow motion. The ground plane alone is enough for the rollout view.
    franka = world.scene.add(Franka(prim_path="/World/franka", name="franka"))

    trails = {"a": Trail(world, "a", COLOR_A), "b": Trail(world, "b", COLOR_B)}

    set_camera_view(eye=CAM_EYE, target=CAM_TARGET)
    try:
        camera = Camera(prim_path="/OmniverseKit_Persp", resolution=(SIM_W, SIM_H))
    except Exception:
        camera = Camera(prim_path="/World/cap_cam", position=np.array(CAM_EYE),
                        resolution=(SIM_W, SIM_H))

    controller = RMPFlowController(name="rmpflow", robot_articulation=franka)
    world.reset()
    camera.initialize()
    controller.reset()
    set_camera_view(eye=CAM_EYE, target=CAM_TARGET)
    for _ in range(60):
        world.step(render=True)

    home_joints = franka.get_joint_positions()
    ee_home_pos, ee_home_quat = franka.end_effector.get_world_pose()
    anchor = np.asarray(args.anchor, dtype=float)
    print(f"[rollout] EE home pose: {np.round(ee_home_pos, 3)}")
    print(f"[rollout] workspace anchor: {np.round(anchor, 3)}  "
          f"display_len={args.display_len}  min_ee_z={args.min_ee_z}")
    print(f"[rollout] BRIDGE_TO_ISAAC =\n{BRIDGE_TO_ISAAC}")

    def go_home():
        franka.set_joint_positions(home_joints)
        controller.reset()
        for _ in range(20):
            world.step(render=True)

    for i, p in enumerate(pairs):
        frame = pair_frame(p)
        stem = output_stem(p)
        print(f"[{i + 1}/{len(pairs)}] {stem}")
        scene_img = load_scene_image(p)
        trails["a"].clear()
        trails["b"].clear()
        frames = []
        log_pair_deltas(p)
        mapped_a = map_action(p["action_a"])
        mapped_b = map_action(p["action_b"])
        disp_scale = pair_display_scale(
            mapped_a, mapped_b, args.display_len, args.scale)
        start = anchor.copy()

        for role in ("a", "b"):
            go_home()
            panel = make_panel(scene_img, p[f"instr_{role}"], role,
                               p["pair_id"], frame)

            def grab():
                frames.append(compose(
                    camera.get_rgba()[..., :3].astype(np.uint8), panel))

            for _ in range(int(args.fps * 0.5)):
                world.step(render=True)
                grab()

            mapped = map_action(p[f"action_{role}"])[:3]
            delta = np.asarray(mapped, dtype=float) * disp_scale
            move_ee(controller, franka, world, start, ee_home_quat,
                    args.approach_steps, grab=grab)
            target = clamp_ee_target(start + delta, start, args.min_ee_z)
            print(f"    {role}: scale={disp_scale:.2f}  "
                  f"isaac_delta_disp={np.round(delta, 4)}  "
                  f"target={np.round(target, 4)}")

            for _ in range(int(args.fps * 0.3)):
                world.step(render=True)
                grab()

            move_ee_linear(
                controller, franka, world, start, target, ee_home_quat,
                args.steps, grab=grab, trail=trails[role],
                trail_every=args.trail_every)

            for _ in range(int(args.fps * 0.7)):
                world.step(render=True)
                grab()

        write_video(frames, os.path.join(out_dir, f"{stem}.mp4"), args.fps)

    print(f"[rollout] done -> {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
