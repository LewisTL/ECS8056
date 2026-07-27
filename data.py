"""
data.py: BridgeData V2 (Open X-Embodiment mirror) loading and scene filtering.

Responsibilities:
  * Stream episodes from the public GCS mirror without a full local download.
  * Extract the initial observation frame, its natural-language instruction, and
    an early-motion ground-truth vector for each episode.
  * Detect the gripper-close (grasp) step per episode and extract a second
    observation plus post-grasp ground truth, the decision point for
    placement_relation scenes.
  * Classify instructions as multi-object / spatially-relational via a transparent
    text heuristic (the pilot filter for multi-object scene selection).
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


# Instruction categories (see `classify_instruction`).
CATEGORY_REFERENT = "referent_selection"   # spatial term selects WHICH object
CATEGORY_PLACEMENT = "placement_relation"  # spatial term is a DESTINATION
CATEGORY_OTHER = "other"                    # no usable spatial term


@dataclass
class InstructionTags:
    spatial_tokens: list[str] = field(default_factory=list)
    spatial_phrases: list[str] = field(default_factory=list)
    has_transfer: bool = False
    category: str = CATEGORY_OTHER

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


def _categorise(tokens: list[str], phrases: list[str]) -> str:
    """Assign an instruction category from the shape of the matched spatial cue.

    Heuristic (transparent and approximate):
      * "placement_relation": a destination phrase is present (`to the left`,
        `to the right`, `next to`, `in front of`, `on top of`, `close to`,
        `far from`, `between`). A destination phrase names a goal location,
        e.g. "move the cloth to the right of the colander".
      * "referent_selection": a spatial cue is present but no destination
        phrase matched, so the cue is a bare token or a possessive phrase that
        modifies the object noun phrase instead, e.g. "pick up the cup on the
        left".
      * "other": no spatial cue.

    Known limitation: an instruction that carries both a referent modifier and
    a destination phrase (e.g. "put the cup on the left on the shelf") is
    still misread as placement only, since any matched destination phrase
    decides the category outright. Categories are annotated, not deleted, so
    manual review can correct edge cases; `category_manual` exists precisely
    for this.
    """
    if phrases:
        return CATEGORY_PLACEMENT
    if tokens:
        return CATEGORY_REFERENT
    return CATEGORY_OTHER


def classify_instruction(text: str) -> InstructionTags:
    """Tag an instruction with spatial and transfer cues, plus a category.

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
    category = _categorise(tokens, phrases)
    return InstructionTags(spatial_tokens=tokens, spatial_phrases=phrases,
                           has_transfer=transfer, category=category)


# --------------------------------------------------------------------------- #
# Episode extraction
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeRecord:
    episode_index: int
    instruction: str
    image: np.ndarray            # uint8 (H, W, 3): initial frame
    gt_vector: np.ndarray        # float32 (7,): early-motion ground truth
    num_steps: int
    tags: InstructionTags
    grasp_frame_index: int | None = None
    grasp_image: np.ndarray | None = None        # uint8 (H, W, 3): grasp-step frame
    grasp_gt_vector: np.ndarray | None = None    # float32 (7,): post-grasp ground truth


def _flatten_action(world_vector, rotation_delta, open_gripper) -> np.ndarray:
    """Combine the action dict components into the flat 7-vector layout."""
    vec = np.zeros(7, dtype=np.float32)
    vec[IDX_TRANSLATION] = np.asarray(world_vector, dtype=np.float32)
    vec[IDX_ROTATION] = np.asarray(rotation_delta, dtype=np.float32)
    vec[IDX_GRIPPER] = float(bool(open_gripper))
    return vec


def _gripper_open(step) -> bool:
    """Coerce a step's open_gripper action field to a plain bool.

    Accepts a TensorFlow scalar (has `.numpy()`) or a plain Python/NumPy bool,
    so the same helper works on live BridgeData V2 episodes and on the
    synthetic steps used in tests.
    """
    value = step["action"]["open_gripper"]
    if hasattr(value, "numpy"):
        value = value.numpy()
    return bool(value)


def grasp_index(steps) -> int | None:
    """Index of the first step where the gripper transitions open to closed.

    In BridgeData V2, `action['open_gripper']` is True while the gripper is
    open; the grasp is the first True-to-False transition. Only the first
    transition is reported; later regrasps are ignored.

    Returns None when the gripper never closes, when it is already closed at
    the first step (there is no prior open state to transition from), or when
    the transition falls on the final step, since no step remains to measure
    post-grasp motion.
    """
    n = len(steps)
    if n < 2:
        return None
    prev_open = _gripper_open(steps[0])
    for i in range(1, n):
        cur_open = _gripper_open(steps[i])
        if prev_open and not cur_open:
            return i if i < n - 1 else None
        prev_open = cur_open
    return None


def extract_episode(
    episode,
    episode_index: int,
    early_steps: int = 5,
    skip_first_noop: bool = True,
    post_grasp_steps: int = 5,
) -> EpisodeRecord:
    """Pull the initial frame, instruction, early-motion ground truth, and the
    grasp-frame observation.

    The initial frame is taken from the first step (the scene the policy
    observes) and remains the primary observation for referent_selection
    scenes. The ground-truth vector is the net action summed over the first
    `early_steps` non-initial steps, because the initial transition in
    BridgeData V2 is a recorded no-op and carries no directional signal.

    The grasp frame is the step where the gripper first closes (see
    `grasp_index`): for placement_relation scenes, the two instruction
    variants demand identical motion until the object is in hand, so the
    grasp step is the first point where they can diverge. `grasp_gt_vector` is
    the net action summed over the `post_grasp_steps` steps that follow the
    grasp index, the transport direction the placement term is supposed to
    determine. As with the early-motion vector, the gripper component is a
    state (recorded at the grasp step) rather than a sum. All three grasp
    fields are None when `grasp_index` returns None.
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

    grasp_frame_index = grasp_index(steps)
    grasp_image = None
    grasp_gt_vector = None
    if grasp_frame_index is not None:
        grasp_step = steps[grasp_frame_index]
        grasp_image = grasp_step["observation"]["image"].numpy()
        post_steps = steps[grasp_frame_index + 1: grasp_frame_index + 1 + post_grasp_steps]
        ggt = np.zeros(7, dtype=np.float32)
        for step in post_steps:
            action = step["action"]
            ggt += _flatten_action(
                action["world_vector"].numpy(),
                action["rotation_delta"].numpy(),
                action["open_gripper"].numpy(),
            )
        ggt[IDX_GRIPPER] = float(_gripper_open(grasp_step))
        grasp_gt_vector = ggt

    return EpisodeRecord(
        episode_index=episode_index,
        instruction=instruction,
        image=image,
        gt_vector=gt,
        num_steps=len(steps),
        tags=classify_instruction(instruction),
        grasp_frame_index=grasp_frame_index,
        grasp_image=grasp_image,
        grasp_gt_vector=grasp_gt_vector,
    )


def stream_episodes(n: int, split_start: int = 0, early_steps: int = 5,
                     post_grasp_steps: int = 5):
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
                              early_steps=early_steps,
                              post_grasp_steps=post_grasp_steps)


# --------------------------------------------------------------------------- #
# Drive cache (filtered survivors only)
# --------------------------------------------------------------------------- #
# Default value for the manual feasibility review (see update_manifest_annotations).
FEASIBLE_DEFAULT = "unreviewed"

# category_source values: the heuristic default, or manual once category_manual
# is set (see update_manifest_annotations).
CATEGORY_SOURCE_HEURISTIC = "heuristic"
CATEGORY_SOURCE_MANUAL = "manual"

MANIFEST_FIELDS = [
    "episode_index", "instruction", "category", "is_multi_object", "has_spatial",
    "has_transfer", "matched_terms", "num_steps", "image_path",
    "feasible_both", "feasibility_note",
    "gt_dx", "gt_dy", "gt_dz", "gt_rx", "gt_ry", "gt_rz", "gt_gripper",
    "grasp_frame_index", "grasp_image_path",
    "grasp_gt_dx", "grasp_gt_dy", "grasp_gt_dz",
    "category_manual", "category_source",
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

            # Grasp fields stay empty when no grasp was detected, matching the
            # existing convention for unfilled manual-review columns.
            grasp_frame_index = ""
            grasp_rel = ""
            grasp_dx = grasp_dy = grasp_dz = ""
            if rec.grasp_frame_index is not None:
                grasp_frame_index = rec.grasp_frame_index
                grasp_rel = os.path.join("frames", f"ep_{rec.episode_index:06d}_grasp.png")
                Image.fromarray(rec.grasp_image).save(os.path.join(out_dir, grasp_rel))
                ggt = rec.grasp_gt_vector
                grasp_dx, grasp_dy, grasp_dz = ggt[0], ggt[1], ggt[2]

            writer.writerow({
                "episode_index": rec.episode_index,
                "instruction": rec.instruction,
                "category": rec.tags.category,
                "is_multi_object": rec.tags.is_multi_object,
                "has_spatial": rec.tags.has_spatial,
                "has_transfer": rec.tags.has_transfer,
                "matched_terms": rec.tags.matched,
                "num_steps": rec.num_steps,
                "image_path": rel,
                # Feasibility is filled in later by manual review, not deleted.
                "feasible_both": FEASIBLE_DEFAULT,
                "feasibility_note": "",
                "gt_dx": gt[0], "gt_dy": gt[1], "gt_dz": gt[2],
                "gt_rx": gt[3], "gt_ry": gt[4], "gt_rz": gt[5],
                "gt_gripper": gt[6],
                "grasp_frame_index": grasp_frame_index,
                "grasp_image_path": grasp_rel,
                "grasp_gt_dx": grasp_dx, "grasp_gt_dy": grasp_dy, "grasp_gt_dz": grasp_dz,
                "category_manual": "",
                "category_source": CATEGORY_SOURCE_HEURISTIC,
            })
            written += 1
    print(f"[cache_records] wrote {written} frames + manifest to {out_dir}")
    return manifest_path


def load_manifest(out_dir: str):
    """Read a cached manifest back as a list of row dicts (no image decoding).

    Tolerates manifests written before a column was added to MANIFEST_FIELDS:
    every row is normalised to the current field set, with missing columns
    read as an empty string rather than raising a KeyError downstream.
    """
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{col: row.get(col, "") for col in MANIFEST_FIELDS} for row in rows]


# Fields update_manifest_annotations may write. Setting category_manual always
# sets category_source to CATEGORY_SOURCE_MANUAL in the same call.
ANNOTATION_FIELDS = frozenset({
    "feasible_both", "feasibility_note", "category_manual", "category_source",
})


def update_manifest_annotations(out_dir: str, annotations: dict) -> str:
    """Rewrite the manifest with manual annotations filled in.

    Annotates rather than deletes: contrastive prompts built by antonym swap can
    imply physically infeasible placements, so each scene is reviewed for whether
    the implied placement is possible on BOTH sides of the referent. The same
    reviewed path also carries manual category correction, since a spatial term
    that both selects a referent and names a destination is misread by
    `_categorise`; the heuristic `category` column is never overwritten, and
    downstream code reads `category_manual` when populated, falling back to
    `category`.

    Args:
        out_dir: directory holding manifest.csv.
        annotations: {episode_index: {field: value, ...}}, where field is one
            of ANNOTATION_FIELDS (`feasible_both`, `feasibility_note`,
            `category_manual`, `category_source`). Keys may be int or str;
            only the provided fields are overwritten, other rows and other
            fields keep their existing values. Setting `category_manual` sets
            `category_source` to `manual` regardless of any explicit
            `category_source` value passed in the same note.

    Returns the manifest path.
    """
    manifest_path = os.path.join(out_dir, "manifest.csv")
    rows = load_manifest(out_dir)
    # Normalise keys to strings so int/str episode indices both match.
    ann = {str(k): v for k, v in annotations.items()}

    updated = 0
    for row in rows:
        note = ann.get(str(row["episode_index"]))
        if not note:
            continue
        touched = False
        for field_name in ANNOTATION_FIELDS:
            if field_name in note:
                row[field_name] = note[field_name]
                touched = True
        if "category_manual" in note:
            row["category_source"] = CATEGORY_SOURCE_MANUAL
        if touched:
            updated += 1

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    print(f"[update_manifest_annotations] updated {updated} rows in {manifest_path}")
    return manifest_path
