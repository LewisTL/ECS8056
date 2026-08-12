"""
export_pairs.py: contrastive-pair assembly and export for external visualisation.

Responsibilities:
  * Join the prediction log (written by model.append_prediction_log) with the
    cached BridgeData V2 manifest (written by data.cache_records).
  * Group predictions into contrastive pairs by
    (pair_id, frame, condition, scene_source) and summarise repeated samples or
    paraphrases per role as mean and standard deviation. Predictions from
    different frames, conditions, or scene sources are never pooled: each names
    a different comparison, and averaging across them would combine measurements
    that answer different questions.
  * Write the pairs.json consumed by the Isaac Sim visualiser
    (visualise_pairs.py) on the rendering instance.

Prerequisites: the prediction log must carry `pair_id`, `role`, `scene_id`, and
`frame` columns, supplied via **extra at logging time. For the bridge source,
`scene_id` must correspond to `episode_index` in the manifest for the
ground-truth join. Older logs still load: a missing `frame` is treated as
`initial`, a missing `condition` as `baseline`, and a missing `scene_source` as
`bridge`, each with a one-time warning.

Two behaviours are worth stating because they were defects in the earlier
version. Stratification labels (`category`, `feasible_both`, `duplicate_target`)
are read from the manifest in preference to the prediction log: the log freezes
whatever the labels were when the probe ran, which hid every label added by
review afterwards and is why the manifest showed 15 pairable feasible referent
scenes while the analysis found 7. And single-role groups are expected rather
than malformed, since the term-stripped conditions produce one prediction with
no opposite; they are counted separately instead of being reported as
incomplete pairs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# Column names produced by model.append_prediction_log / data.cache_records.
ACTION_COLS = [f"a{i}" for i in range(7)]
CONTINUOUS_COLS = [f"c{i}" for i in range(7)]
GT_COLS = ["gt_dx", "gt_dy", "gt_dz", "gt_rx", "gt_ry", "gt_rz", "gt_gripper"]
# Grasp-frame ground truth is translation-only: rotation and gripper state were
# never persisted for the post-grasp window (see data.cache_records).
GRASP_GT_COLS = ["grasp_gt_dx", "grasp_gt_dy", "grasp_gt_dz"]
REQUIRED_PRED_COLS = {"pair_id", "role", "scene_id", "instruction", "frame", *ACTION_COLS}

# Factors that define a distinct comparison. Predictions are grouped by these
# together with pair_id, so no two of them are ever averaged together.
GROUP_COLS = ["pair_id", "frame", "condition", "scene_source"]

# Defaults applied to logs written before a factor existed, so earlier runs load
# without migration.
GROUP_DEFAULTS = {"frame": "initial", "condition": "baseline", "scene_source": "bridge"}

# Labels that describe the scene and can change after the probe has run. Always
# taken from the manifest when the scene is present there.
MANIFEST_LABELS = ("category", "feasible_both", "duplicate_target")

# Labels that describe the stimulus and are fixed at prediction time.
LOG_LABELS = ("spatial_term", "axis", "axis_index", "configuration",
              "expected_sign", "image_transform")

ROLE_NEUTRAL = "n"

# Roles assumed for a condition the controls table does not name, which covers
# every row in a log written before conditions existed.
DEFAULT_ROLES = frozenset({"a", "b"})


def expected_roles(condition: str) -> frozenset:
    """Roles a condition is supposed to produce.

    Read from the controls table so that a group holding fewer predictions than
    a pair is only reported as incomplete when the condition actually calls for
    more. The mirror and term-stripped conditions declare a single role by
    design, and counting those as broken pairs is what made the earlier export
    understate its own yield.
    """
    try:
        from controls import CONDITIONS
    except ImportError:
        return DEFAULT_ROLES
    entry = CONDITIONS.get(str(condition))
    return frozenset(entry.roles) if entry is not None else DEFAULT_ROLES

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


def _summarise(group: pd.DataFrame, cols=None):
    """Mean, standard deviation, and count of the mapped action vectors in a role."""
    cols = cols or ACTION_COLS
    acts = np.stack([map_action([r[c] for c in cols]) for _, r in group.iterrows()])
    return acts.mean(axis=0).tolist(), acts.std(axis=0).tolist(), len(acts)


def _continuous(group: pd.DataFrame):
    """Mean continuous readout for a role, or None when it was never logged."""
    if not set(CONTINUOUS_COLS) <= set(group.columns):
        return None
    values = group[CONTINUOUS_COLS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return map_action(values.mean(axis=0)).tolist()


def manifest_key(scene_id: str) -> str | None:
    """Manifest `episode_index` a scene id refers to, or None if it has none.

    Bridge scene ids are the episode index with a `b` prefix and zero padding;
    constructed scene ids name a composited stimulus that has no Bridge manifest
    row and therefore no ground truth.
    """
    text = str(scene_id).strip()
    body = text[1:] if text[:1].lower() == "b" else text
    return str(int(body)) if body.isdigit() else None


def load_inputs(pred_path: str, manifest_path: str | None = None):
    """Read the prediction log and (optionally) the manifest as DataFrames."""
    preds = pd.read_csv(pred_path)
    for col, default in GROUP_DEFAULTS.items():
        if col not in preds.columns:
            print(f"[load_inputs] prediction log has no {col!r} column; "
                  f"treating every row as {col}={default!r}.")
            preds[col] = default
        else:
            preds[col] = preds[col].fillna(default)
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

    Predictions are grouped by (pair_id, frame, condition, scene_source). Each
    factor names a distinct comparison: the initial frame is the primary
    observation for referent_selection scenes while the grasp frame is the
    decision point for placement_relation scenes, each condition transforms a
    different part of the stimulus, and the two scene sources carry the primary
    and secondary analyses. Pooling across any of them would average
    measurements that answer different questions.

    Returns (pairs, stats). `stats` counts groups by outcome: `paired` for
    groups holding both roles, `neutral` for the single-prediction term-stripped
    conditions, which are expected rather than malformed, and `incomplete` for
    groups genuinely missing a role, which usually means an interrupted run.

    Repeated predictions per role (samples or paraphrases) are averaged; the
    per-axis standard deviation and count are retained for effect-size and
    variance-cone use downstream. Where the continuous readout is present it is
    averaged the same way and carried as `cont_a` and `cont_b`.
    """
    pairs = []
    stats = {"paired": 0, "neutral": 0, "incomplete": 0}
    for keys, grp in preds.groupby(GROUP_COLS, dropna=False):
        pair_id, frame, condition, scene_source = keys
        roles = {r: g for r, g in grp.groupby("role")}
        scene_id = str(grp["scene_id"].iloc[0])

        entry = {
            "scene_id": scene_id,
            "pair_id": pair_id,
            "frame": frame,
            "condition": condition,
            "scene_source": scene_source,
            "start_pos": DEFAULT_START_POS,
        }

        expected = expected_roles(condition)
        if not expected <= roles.keys():
            stats["incomplete"] += 1
            continue

        for role in sorted(roles if len(expected) > 1 else expected):
            action, std, count = _summarise(roles[role])
            entry[f"instr_{role}"] = roles[role]["instruction"].iloc[0]
            entry[f"action_{role}"] = action
            entry[f"action_{role}_std"] = std
            entry[f"n_{role}"] = count
            continuous = _continuous(roles[role])
            if continuous is not None:
                entry[f"cont_{role}"] = continuous

        if {"a", "b"} <= roles.keys():
            entry["kind"] = "pair"
            stats["paired"] += 1
        else:
            # Conditions declaring one role produce one prediction with no
            # opposite: a complete record rather than half of one.
            entry["kind"] = "single"
            stats["neutral"] += 1

        # Stimulus properties are fixed when the prediction is made, so the log
        # is authoritative for them.
        for col in LOG_LABELS:
            if col in grp.columns and pd.notna(grp[col].iloc[0]):
                entry[col] = str(grp[col].iloc[0])

        # Scene labels can be added by review after the probe has run, so the
        # manifest is authoritative and the log is only a fallback.
        key = manifest_key(scene_id)
        row = None
        if manifest is not None and key is not None:
            match = manifest[manifest["scene_id"] == key]
            row = match.iloc[0] if len(match) else None
        for col in MANIFEST_LABELS:
            value = None
            if row is not None and col in manifest.columns and pd.notna(row[col]):
                value = str(row[col])
            elif col in grp.columns and pd.notna(grp[col].iloc[0]):
                value = str(grp[col].iloc[0])
            if value is not None:
                entry[col] = value

        if row is not None:
            # Grasp-frame ground truth and image come from the grasp_* manifest
            # columns; initial-frame ground truth is unchanged. Attach the image
            # whenever the path is present, even if the corresponding
            # ground-truth vector is incomplete, so rollouts can still show the
            # BridgeData photo.
            gt_cols, path_col = (
                (GRASP_GT_COLS, "grasp_image_path") if frame == "grasp"
                else (GT_COLS, "image_path")
            )
            path_val = row.get(path_col)
            if pd.notna(path_val) and str(path_val).strip():
                entry["image_path"] = str(path_val)
            if all(pd.notna(row.get(c)) for c in gt_cols):
                entry["gt_vector"] = map_action([float(row[c]) for c in gt_cols]).tolist()

        pairs.append(entry)
    return pairs, stats


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
    pairs, stats = build_pairs(preds, manifest)
    print(f"[build_pairs] {stats['paired']} pairs, {stats['neutral']} neutral records")
    if stats["incomplete"]:
        print(f"[build_pairs] skipped {stats['incomplete']} groups missing a role; "
              "the probe run was most likely interrupted")
    write_pairs(pairs, args.out)
