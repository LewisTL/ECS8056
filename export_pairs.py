"""
export_pairs.py — Contrastive-pair assembly and export for external visualisation.

Responsibilities:
  * Join the prediction log (written by model.append_prediction_log) with the
    cached BridgeData V2 manifest (written by data.cache_records).
  * Group predictions into contrastive pairs by pair_id and summarise repeated
    samples/paraphrases per role as mean and standard deviation.
  * Write the pairs.json consumed by the Isaac Sim visualiser
    (visualise_pairs.py) on the rendering instance.

Prerequisites: the prediction log must carry `pair_id`, `role` ('a' | 'b') and
`scene_id` columns, supplied via **extra at logging time. `scene_id` must match
`episode_index` in the manifest for the ground-truth join.

Axis-mapping caveat: BRIDGE_TO_ISAAC transforms translation components from the
Bridge action frame into the Isaac Sim world frame. It defaults to identity and
must be set from the pilot validation (sign agreement against gt_vector) before
rendered figures are treated as meaningful. The same matrix is applied to the
ground-truth vectors so predicted and reference arrows share a frame.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# Column names produced by model.append_prediction_log / data.cache_records.
ACTION_COLS = [f"a{i}" for i in range(7)]
GT_COLS = ["gt_dx", "gt_dy", "gt_dz", "gt_rx", "gt_ry", "gt_rz", "gt_gripper"]
REQUIRED_PRED_COLS = {"pair_id", "role", "scene_id", "instruction", *ACTION_COLS}

# 3x3 mapping from Bridge action axes to Isaac Sim world axes (translation
# only). Identity until validated in the pilot phase; a swap/flip would look
# like e.g. [[0, -1, 0], [1, 0, 0], [0, 0, 1]].
BRIDGE_TO_ISAAC = np.eye(3)

# Root position for arrows in the Isaac scene (metres, world frame). Bridge
# frames carry no reconstructable end-effector world pose, so a fixed tabletop
# anchor is used; the visualisation concerns direction, not absolute position.
DEFAULT_START_POS = [0.40, 0.00, 0.20]


def map_action(values) -> np.ndarray:
    """Return a 7-vector with the translation mapped into Isaac world axes."""
    a = np.asarray(values, dtype=float).copy()
    a[:3] = BRIDGE_TO_ISAAC @ a[:3]
    return a


def _summarise(group: pd.DataFrame):
    """Mean/std/count of the mapped action vectors within one (pair_id, role)."""
    acts = np.stack([map_action([r[c] for c in ACTION_COLS])
                     for _, r in group.iterrows()])
    return acts.mean(axis=0).tolist(), acts.std(axis=0).tolist(), len(acts)


def load_inputs(pred_path: str, manifest_path: str | None = None):
    """Read the prediction log and (optionally) the manifest as DataFrames."""
    preds = pd.read_csv(pred_path)
    missing = REQUIRED_PRED_COLS - set(preds.columns)
    if missing:
        raise ValueError(
            f"prediction log is missing columns {sorted(missing)}; "
            "supply them via **extra in append_prediction_log."
        )
    manifest = None
    if manifest_path:
        manifest = pd.read_csv(manifest_path)
        manifest["scene_id"] = manifest["episode_index"].astype(str)
    return preds, manifest


def build_pairs(preds: pd.DataFrame, manifest: pd.DataFrame | None = None):
    """Assemble pair records from the prediction log.

    Returns (pairs, skipped) where `skipped` counts pair_ids lacking both
    roles. Repeated predictions per role (samples or paraphrases) are averaged;
    the per-axis standard deviation and count are retained for effect-size and
    variance-cone use downstream.
    """
    pairs, skipped = [], 0
    for pair_id, grp in preds.groupby("pair_id"):
        roles = {r: g for r, g in grp.groupby("role")}
        if not {"a", "b"} <= roles.keys():
            skipped += 1
            continue

        act_a, std_a, n_a = _summarise(roles["a"])
        act_b, std_b, n_b = _summarise(roles["b"])
        scene_id = str(grp["scene_id"].iloc[0])

        entry = {
            "scene_id": scene_id,
            "pair_id": pair_id,
            "start_pos": DEFAULT_START_POS,
            "instr_a": roles["a"]["instruction"].iloc[0],
            "action_a": act_a, "action_a_std": std_a, "n_a": n_a,
            "instr_b": roles["b"]["instruction"].iloc[0],
            "action_b": act_b, "action_b_std": std_b, "n_b": n_b,
        }

        # Carry stratification fields through for downstream analysis. Prefer the
        # probe log (logged per prediction), falling back to the manifest.
        for col in ("category", "feasible_both", "spatial_term"):
            if col in grp.columns:
                entry[col] = str(grp[col].iloc[0])

        if manifest is not None:
            m = manifest[manifest["scene_id"] == scene_id]
            if len(m):
                m0 = m.iloc[0]
                gt = map_action([float(m0[c]) for c in GT_COLS])
                entry["gt_vector"] = gt.tolist()
                entry["image_path"] = m0["image_path"]
                for col in ("category", "feasible_both"):
                    if col not in entry and col in manifest.columns:
                        entry[col] = str(m0[col])

        pairs.append(entry)
    return pairs, skipped


def write_pairs(pairs: list, out_path: str) -> str:
    """Serialise pair records to JSON for transfer to the rendering instance."""
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[write_pairs] wrote {len(pairs)} pairs -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="pairs.json")
    args = ap.parse_args()

    preds, manifest = load_inputs(args.predictions, args.manifest)
    pairs, skipped = build_pairs(preds, manifest)
    if skipped:
        print(f"[build_pairs] skipped {skipped} pair_ids lacking both roles")
    write_pairs(pairs, args.out)
