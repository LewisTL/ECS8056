"""
data.py — BridgeData V2 (Open X-Embodiment mirror) loading and scene filtering.

Responsibilities:
  * Stream episodes from the public GCS mirror without a full local download.
  * Extract the initial observation frame, its natural-language instruction, and
    an early-motion ground-truth vector for each episode.
  * Classify instructions as multi-object / spatially-relational via a transparent
    text heuristic (the pilot filter for objective O2 / risk R2).
  * Cache the surviving frames plus a manifest to Google Drive as a compact,
    reproducible evaluation set.

Schema notes (confirmed against gs://gresearch/robotics/bridge/0.1.0/):
  * observation['image']                       uint8  (480, 640, 3)
  * observation['natural_language_instruction'] string scalar
  * observation['state']                       float32 (7,)
  * action is a dict: world_vector (3,), rotation_delta (3,), open_gripper (bool),
    terminate_episode (float). The flat 7-vector layout used here is
    [world_vector(3), rotation_delta(3), gripper(1)], mirroring the OpenVLA output
    layout [dx, dy, dz, droll, dpitch, dyaw, gripper].

Convention caveat: ground-truth actions follow the OXE convention, whereas model
predictions are de-normalised with unnorm_key='bridge_orig'. Magnitudes are NOT
directly comparable across the two; the directional-consistency metric relies on
the SIGN of the translation components, which is convention-invariant.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

BRIDGE_GCS = "gs://gresearch/robotics/bridge/0.1.0/"

# Action layout indices for the flattened 7-vector.
IDX_TRANSLATION = slice(0, 3)   # world_vector
IDX_ROTATION = slice(3, 6)      # rotation_delta
IDX_GRIPPER = 6                 # float(open_gripper)


# --------------------------------------------------------------------------- #
# Instruction classification (text heuristic)
# --------------------------------------------------------------------------- #
# Single-token spatial-relation cues. Matched on word boundaries to avoid
# substring false positives (e.g. "left" inside "cleft").
SPATIAL_TOKENS = frozenset({
    "left", "right", "behind", "front", "near", "far", "beside",
    "above", "below", "top", "bottom", "back", "closer", "farther",
    "nearest", "farthest", "leftmost", "rightmost",
})

# Multi-word spatial phrases. Matched as lowercase substrings.
SPATIAL_PHRASES = (
    "next to", "in front of", "to the left", "to the right",
    "on top of", "close to", "far from", "between",
)

# Object-transfer structure: a transfer verb plus a relational preposition implies
# at least two distinct objects (source and target), e.g. "put the carrot on the
# plate". Captures multi-object scenes that carry no explicit spatial term.
TRANSFER_VERBS = frozenset({"put", "place", "move", "stack", "set"})
TRANSFER_PREPS = frozenset({"on", "onto", "in", "into", "from", "inside"})

_WORD_RE = re.compile(r"[a-z]+")


@dataclass
class InstructionTags:
    spatial_tokens: list[str] = field(default_factory=list)
    spatial_phrases: list[str] = field(default_factory=list)
    has_transfer: bool = False

    @property
    def has_spatial(self) -> bool:
        return bool(self.spatial_tokens or self.spatial_phrases)

    @property
    def is_multi_object(self) -> bool:
        # Treat either an explicit spatial relation or a transfer structure as a
        # multi-object candidate. Spatial scenes are the primary probe target;
        # transfer scenes are retained as a broader pool.
        return self.has_spatial or self.has_transfer

    @property
    def matched(self) -> str:
        parts = self.spatial_tokens + self.spatial_phrases
        if self.has_transfer:
            parts.append("transfer")
        return "|".join(parts)


def classify_instruction(text: str) -> InstructionTags:
    """Tag an instruction with spatial and transfer cues.

    The heuristic is approximate by design: scene composition is inferred from
    language alone, so terse multi-object descriptions may be missed and unusual
    phrasing may false-positive. Cached frames are intended for manual review
    before the evaluation set is finalised.
    """
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    word_set = set(words)

    tokens = sorted(SPATIAL_TOKENS & word_set)
    phrases = [p for p in SPATIAL_PHRASES if p in lowered]
    transfer = bool(TRANSFER_VERBS & word_set) and bool(TRANSFER_PREPS & word_set)
    return InstructionTags(spatial_tokens=tokens, spatial_phrases=phrases,
                           has_transfer=transfer)


# --------------------------------------------------------------------------- #
# Episode extraction
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeRecord:
    episode_index: int
    instruction: str
    image: np.ndarray            # uint8 (H, W, 3) — initial frame
    gt_vector: np.ndarray        # float32 (7,) — early-motion ground truth
    num_steps: int
    tags: InstructionTags


def _flatten_action(world_vector, rotation_delta, open_gripper) -> np.ndarray:
    """Combine the action dict components into the flat 7-vector layout."""
    vec = np.zeros(7, dtype=np.float32)
    vec[IDX_TRANSLATION] = np.asarray(world_vector, dtype=np.float32)
    vec[IDX_ROTATION] = np.asarray(rotation_delta, dtype=np.float32)
    vec[IDX_GRIPPER] = float(bool(open_gripper))
    return vec


def extract_episode(
    episode,
    episode_index: int,
    early_steps: int = 5,
    skip_first_noop: bool = True,
) -> EpisodeRecord:
    """Pull the initial frame, instruction, and early-motion ground truth.

    The initial frame is taken from the first step (the scene the policy observes).
    The ground-truth vector is the net action summed over the first `early_steps`
    non-initial steps, because the initial transition in BridgeData V2 is a
    recorded no-op and carries no directional signal.
    """
    steps = list(episode["steps"])
    first = steps[0]

    image = first["observation"]["image"].numpy()
    instruction = first["observation"]["natural_language_instruction"].numpy().decode("utf-8")

    motion_steps = steps[1:] if (skip_first_noop and len(steps) > 1) else steps
    motion_steps = motion_steps[:early_steps]

    gt = np.zeros(7, dtype=np.float32)
    for step in motion_steps:
        action = step["action"]
        gt += _flatten_action(
            action["world_vector"].numpy(),
            action["rotation_delta"].numpy(),
            action["open_gripper"].numpy(),
        )
    # Gripper is a state, not a delta; record the initial-step value rather than a
    # sum over the early window.
    gt[IDX_GRIPPER] = float(bool(first["action"]["open_gripper"].numpy()))

    return EpisodeRecord(
        episode_index=episode_index,
        instruction=instruction,
        image=image,
        gt_vector=gt,
        num_steps=len(steps),
        tags=classify_instruction(instruction),
    )


def stream_episodes(n: int, split_start: int = 0, early_steps: int = 5):
    """Yield EpisodeRecord objects from the GCS mirror, one episode at a time.

    Imported lazily so the TensorFlow/TFDS stack is only required when streaming,
    keeping this module importable in environments that hold only the model deps.
    """
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(BRIDGE_GCS)
    split = f"train[{split_start}:{split_start + n}]"
    dataset = builder.as_dataset(split=split)
    for offset, episode in enumerate(dataset):
        yield extract_episode(episode, episode_index=split_start + offset,
                              early_steps=early_steps)


# --------------------------------------------------------------------------- #
# Drive cache (filtered survivors only)
# --------------------------------------------------------------------------- #
MANIFEST_FIELDS = [
    "episode_index", "instruction", "is_multi_object", "has_spatial",
    "has_transfer", "matched_terms", "num_steps", "image_path",
    "gt_dx", "gt_dy", "gt_dz", "gt_rx", "gt_ry", "gt_rz", "gt_gripper",
]


def cache_records(records, out_dir: str, multi_object_only: bool = True) -> str:
    """Write surviving frames as PNGs plus a manifest CSV under `out_dir`.

    Only the filtered subset is persisted, keeping the cached evaluation set small
    (tens of MB) and reproducible across sessions without re-streaming. Returns the
    manifest path.
    """
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")

    written = 0
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for rec in records:
            if multi_object_only and not rec.tags.is_multi_object:
                continue
            rel = os.path.join("frames", f"ep_{rec.episode_index:06d}.png")
            Image.fromarray(rec.image).save(os.path.join(out_dir, rel))
            gt = rec.gt_vector
            writer.writerow({
                "episode_index": rec.episode_index,
                "instruction": rec.instruction,
                "is_multi_object": rec.tags.is_multi_object,
                "has_spatial": rec.tags.has_spatial,
                "has_transfer": rec.tags.has_transfer,
                "matched_terms": rec.tags.matched,
                "num_steps": rec.num_steps,
                "image_path": rel,
                "gt_dx": gt[0], "gt_dy": gt[1], "gt_dz": gt[2],
                "gt_rx": gt[3], "gt_ry": gt[4], "gt_rz": gt[5],
                "gt_gripper": gt[6],
            })
            written += 1
    print(f"[cache_records] wrote {written} frames + manifest to {out_dir}")
    return manifest_path


def load_manifest(out_dir: str):
    """Read a cached manifest back as a list of row dicts (no image decoding)."""
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))
