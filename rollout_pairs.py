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
        --frames-root ~/bridge_frames --pair-id ep000070

--frames-root points at the cached BridgeData frames (copy once with e.g.
`rclone copy gdrive:openvla_cache/bridge_multiobj/frames ~/bridge_frames`).
Pairs whose image can't be found still render, without the photo.

Scale caveat: OpenVLA returns one 7-DoF action per instruction, not a full
trajectory. By default the rollout uses a stylised pick-place choreography
(gripper open, approach, close, lift, transport, place, open, retract). The
predicted translation delta drives the transport leg only (same mapped/scaled
vector as the arrow figures). Use --motion delta for the legacy straight-line
displacement demo.
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

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--out", default="./rollouts")
ap.add_argument("--frames-root", default=None,
                help="dir containing the cached BridgeData PNGs")
ap.add_argument("--scene-id", type=int, nargs="+", default=None,
                help="render only pairs whose scene_id is in this list")
ap.add_argument("--pair-id", nargs="+", default=None,
                help="render matching pair_id values: exact match, prefix, "
                     "or epNN / NN shorthand (e.g. ep70 -> ep000070_*)")
ap.add_argument("--list", action="store_true",
                help="print available pair_id / scene_id values and exit "
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
                help="sim steps to reach the workspace anchor (delta mode)")
ap.add_argument("--steps", type=int, default=80,
                help="sim steps for delta-mode motion")
ap.add_argument("--motion", choices=("choreography", "delta"), default="choreography",
                help="choreography: stylised pick-place; delta: straight segment")
ap.add_argument("--no-prop", action="store_true",
                help="disable the visual grasp prop used in choreography mode")
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


def load_and_filter_pairs(path, scene_ids=None, pair_ids=None, limit=0):
    """Load pairs.json and apply optional scene / pair filters."""
    with open(os.path.expanduser(path)) as f:
        all_pairs = json.load(f)
    pairs = list(all_pairs)
    if scene_ids is not None:
        wanted = set(scene_ids)
        pairs = [p for p in pairs if int(p["scene_id"]) in wanted]
    if pair_ids is not None:
        pairs = [p for p in pairs if _pair_id_matches(p["pair_id"], pair_ids)]
    if limit:
        pairs = pairs[:limit]
    return all_pairs, pairs


def format_pair_index(pairs):
    lines = [f"  {p['pair_id']}  (scene_id={p['scene_id']})" for p in pairs]
    return "\n".join(lines) if lines else "  (none)"


all_pairs, pairs = load_and_filter_pairs(
    args.pairs, args.scene_id, args.pair_id, args.limit)

if args.list:
    print(f"[rollout] {len(all_pairs)} pairs in {args.pairs}")
    print(format_pair_index(all_pairs))
    raise SystemExit(0)

if not pairs:
    filtered = args.scene_id is not None or args.pair_id is not None or args.limit
    msg = ["[rollout] no pairs matched the given filters "
           f"(scene-id={args.scene_id}, pair-id={args.pair_id}, limit={args.limit})",
           f"[rollout] {len(all_pairs)} pairs available:"]
    msg.append(format_pair_index(all_pairs))
    if filtered:
        msg.append("[rollout] re-run with --list, or pass a pair_id / scene_id "
                   "from the list above")
    raise SystemExit("\n".join(msg))

print(f"[rollout] rendering {len(pairs)}/{len(all_pairs)} pairs: "
      f"{', '.join(p['pair_id'] for p in pairs)}")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController)
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.core import World
    from omni.isaac.core.objects import VisualCuboid, VisualSphere
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
APPROACH_DEPTH = 0.10     # descend from hover to grasp (metres)
LIFT_HEIGHT = 0.08        # lift after close (metres)
PLACE_DEPTH = 0.08        # lower before release (metres)
PROP_OFFSET = np.array([0.0, 0.0, -0.06])
CHOREO_STEPS = {
    "hover": 12, "approach": 28, "grasp": 18, "lift": 28,
    "transport": 45, "place_down": 22, "release": 18, "retract": 25,
}


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


def transport_delta(delta_display):
    """Use the predicted delta for horizontal transport at a fixed height."""
    d = np.asarray(delta_display, dtype=float).copy()
    d[2] = 0.0
    return d


def build_choreography_waypoints(start, delta_display, min_z):
    """Return (position, gripper_cmd, steps) legs for a placement-style demo."""
    start = np.asarray(start, dtype=float)
    hover = start.copy()
    grasp = clamp_ee_target(start + np.array([0.0, 0.0, -APPROACH_DEPTH]),
                            start, min_z)
    lifted = clamp_ee_target(grasp + np.array([0.0, 0.0, LIFT_HEIGHT]),
                             start, min_z)
    transport = lifted + transport_delta(delta_display)
    transport = clamp_ee_target(transport, start, min_z)
    transport[2] = lifted[2]
    place = clamp_ee_target(transport + np.array([0.0, 0.0, -PLACE_DEPTH]),
                            start, min_z)
    retract = transport.copy()
    return [
        (hover, None, CHOREO_STEPS["hover"]),
        (grasp, None, CHOREO_STEPS["approach"]),
        (grasp, "close", CHOREO_STEPS["grasp"]),
        (lifted, None, CHOREO_STEPS["lift"]),
        (transport, None, CHOREO_STEPS["transport"]),
        (place, None, CHOREO_STEPS["place_down"]),
        (place, "open", CHOREO_STEPS["release"]),
        (retract, None, CHOREO_STEPS["retract"]),
    ]


def gripper_set(franka, world, open_wide, settle=12):
    """Drive the Franka gripper open or closed over several sim steps."""
    try:
        if open_wide:
            franka.gripper.open()
        else:
            franka.gripper.close()
    except Exception:
        joints = np.asarray(franka.get_joint_positions(), dtype=float)
        joints[-2:] = [0.04, 0.04] if open_wide else [0.0, 0.0]
        franka.set_joint_positions(joints)
    for _ in range(settle):
        world.step(render=True)


class GraspProp:
    """Small visual cuboid parked at the grasp point and released at the place."""

    def __init__(self, world):
        self.cube = world.scene.add(VisualCuboid(
            prim_path="/World/grasp_prop", name="grasp_prop",
            position=PARK.copy(), scale=np.array([0.04, 0.04, 0.04]),
            color=np.array([0.85, 0.25, 0.20])))
        self.grasped = False

    def park(self):
        self.grasped = False
        self.cube.set_world_pose(PARK.copy(), np.array([1.0, 0, 0, 0]))

    def show_at(self, pos):
        self.grasped = False
        self.cube.set_world_pose(np.asarray(pos, dtype=float),
                                 np.array([1.0, 0, 0, 0]))

    def grasp(self, ee_pos):
        self.grasped = True
        self._sync(ee_pos)

    def release(self, place_pos):
        self.grasped = False
        self.cube.set_world_pose(np.asarray(place_pos, dtype=float),
                                 np.array([1.0, 0, 0, 0]))

    def _sync(self, ee_pos):
        if self.grasped:
            self.cube.set_world_pose(np.asarray(ee_pos, dtype=float) + PROP_OFFSET,
                                     np.array([1.0, 0, 0, 0]))


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
                   grab=None, trail=None, trail_every=4, prop=None):
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
        if prop is not None and prop.grasped:
            ee_pos, _ = franka.end_effector.get_world_pose()
            prop.grasp(ee_pos)
        if trail is not None and s % trail_every == 0:
            ee_pos, _ = franka.end_effector.get_world_pose()
            trail.drop(ee_pos)
        if grab is not None:
            grab()


def run_choreography(controller, franka, world, start, delta, quat, min_z,
                       grab, trail, prop, trail_every):
    """Execute the pick-place template; predicted delta drives the transport leg."""
    gripper_set(franka, world, True, settle=10)
    legs = build_choreography_waypoints(start, delta, min_z)
    grasp_pos = legs[1][0]
    if prop is not None:
        prop.show_at(np.asarray(grasp_pos, dtype=float) + PROP_OFFSET)
    ee_pos = np.asarray(start, dtype=float)
    for target, grip_cmd, n_steps in legs:
        target = np.asarray(target, dtype=float)
        if grip_cmd is None:
            move_ee_linear(controller, franka, world, ee_pos, target, quat, n_steps,
                           grab=grab, trail=trail, trail_every=trail_every,
                           prop=prop)
            ee_pos = target
            continue
        if grip_cmd == "close":
            gripper_set(franka, world, False)
            if prop is not None:
                ee_now, _ = franka.end_effector.get_world_pose()
                prop.grasp(ee_now)
        elif grip_cmd == "open":
            if prop is not None:
                prop.release(np.asarray(ee_pos, dtype=float) + PROP_OFFSET)
            gripper_set(franka, world, True)
        for _ in range(n_steps):
            world.step(render=True)
            if trail is not None:
                ee_now, _ = franka.end_effector.get_world_pose()
                trail.drop(ee_now)
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


def load_scene_image(p):
    """Locate and load the pair's BridgeData frame, or None."""
    if not args.frames_root or "image_path" not in p:
        return None
    root = os.path.expanduser(args.frames_root)
    for cand in (os.path.join(root, p["image_path"]),
                 os.path.join(root, os.path.basename(p["image_path"]))):
        if os.path.exists(cand):
            return Image.open(cand).convert("RGB")
    return None


def make_panel(scene_img, instr, role, pair_id):
    """Left panel: scene photo + instruction banner coloured by role."""
    panel = Image.new("RGB", (PANEL_W, SIM_H), (24, 24, 24))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    if scene_img is not None:
        img = scene_img.copy()
        img.thumbnail((PANEL_W - 20, 440))
        panel.paste(img, ((PANEL_W - img.width) // 2, 10))
        y = 10 + img.height + 14
        draw.text((10, y), "model input (BridgeData V2)", fill=(150, 150, 150),
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
    draw.text((10, SIM_H - 22), pair_id, fill=(120, 120, 120), font=font)
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
    prop = None if args.no_prop or args.motion != "choreography" else GraspProp(world)

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
    print(f"[rollout] motion={args.motion}  prop="
          f"{'off' if prop is None else 'on'}")
    print(f"[rollout] BRIDGE_TO_ISAAC =\n{BRIDGE_TO_ISAAC}")

    def go_home():
        franka.set_joint_positions(home_joints)
        controller.reset()
        for _ in range(20):
            world.step(render=True)

    for i, p in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}] {p['pair_id']}")
        scene_img = load_scene_image(p)
        if scene_img is None and args.frames_root:
            print("    note: scene image not found; rendering without photo")
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
            if prop is not None:
                prop.park()
            panel = make_panel(scene_img, p[f"instr_{role}"], role, p["pair_id"])

            def grab():
                frames.append(compose(
                    camera.get_rgba()[..., :3].astype(np.uint8), panel))

            for _ in range(int(args.fps * 0.5)):
                world.step(render=True)
                grab()

            mapped = map_action(p[f"action_{role}"])[:3]
            delta = np.asarray(mapped, dtype=float) * disp_scale
            print(f"    {role}: scale={disp_scale:.2f}  "
                  f"isaac_delta_disp={np.round(delta, 4)}")

            if args.motion == "choreography":
                run_choreography(
                    controller, franka, world, start, delta, ee_home_quat,
                    args.min_ee_z, grab, trails[role], prop, args.trail_every)
            else:
                move_ee(controller, franka, world, start, ee_home_quat,
                        args.approach_steps, grab=grab)
                target = clamp_ee_target(start + delta, start, args.min_ee_z)
                print(f"      target={np.round(target, 4)}")
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

        write_video(frames, os.path.join(out_dir, f"{p['pair_id']}.mp4"), args.fps)

    print(f"[rollout] done -> {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
