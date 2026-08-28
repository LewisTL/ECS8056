"""
data.py: BridgeData V2 loading and scene filtering.

Responsibilities:
  * Stream episodes from the public GCS mirror without a full local download.
  * Extract the initial observation frame, its natural-language instruction, and
    an early-motion ground-truth vector for each episode.
  * Detect the gripper-close (grasp) step per episode and extract a second
    observation plus post-grasp ground truth, the decision point for
    placement_relation scenes.
  * Classify instructions as multi-object / spatially-relational via a transparent
    text heuristic (the pilot filter for multi-object scene selection).
  * Harvest the frames into two disjoint roles, one supplying the base frames the
    constructed experiments are built from and one supplying unaltered frames to
    the first-action object-tracking gate, and cache them plus a manifest to
    Google Drive.
  * Build antonym-swapped minimal pairs.

Two roles, one corpus. The experiments run on constructed scenes (see
`compose_scenes.py`), which are composited from real Bridge frames. Unaltered
Bridge frames supply the first-action object-tracking gate in Notebook 04: the
check that the first predicted action tracks a unique named object, without which
a miss toward a named referent on constructed scenes is not interpretable as a
spatial-language failure. A frame that served as a construction base is not that
check, so the harvest assigns each cached frame exactly one role and records it
in `split`. The two roles have different admission rules: a validation trial needs
an instruction that already yields a lateral minimal pair in a referent-selection
reading, which is rare, while a construction base needs only a nameable object,
which is almost every frame.

Schema notes (confirmed against gs://gresearch/robotics/bridge/0.1.0/):
  * observation['image']                       uint8  (480, 640, 3)
  * observation['natural_language_instruction'] string scalar
  * observation['state']                       float32 (7,)
  * action is a dict: world_vector (3,), rotation_delta (3,), open_gripper (bool),
    terminate_episode (float). The flat 7-vector layout used here is
    [world_vector(3), rotation_delta(3), gripper(1)], mirroring the OpenVLA output
    layout [dx, dy, dz, droll, dpitch, dyaw, gripper].
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

BRIDGE_GCS = "gs://gresearch/robotics/bridge/0.1.0/"

# Supported protobuf runtime window for the TensorFlow Datasets stack, bounded from
# both sides. The lower bound is set by the `tensorflow-metadata` generated code,
# which refuses to run on an older runtime. The upper bound is set by released TFDS,
# which reads dataset metadata through `FieldDescriptor.label`; protobuf 7 removed
# that attribute, so a newer runtime imports cleanly and then fails deep inside
# `builder_from_directory`. Both edges are reported before streaming begins.
PROTOBUF_MIN = (6, 31, 1)
PROTOBUF_MAX_EXCLUSIVE = (7, 0, 0)
PROTOBUF_SPEC = 'protobuf>=6.31.1,<7'

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

# Longer prepositions first so the object-phrase split prefers `onto` over `on`.
# Used only to decide whether a prenominal `left`/`right` modifies the
# manipulated object (referent) or a landmark after the preposition (placement).
_OBJECT_PHRASE_SPLIT_RE = re.compile(
    r"\b(?:onto|into|inside|from|next|beside|behind|under|over|on|in|to)\b"
)
# `the left cup`, `leftmost`, not `the left side` or `the left of`.
_PRENOMINAL_REFERENT_RE = re.compile(
    r"\b(?:left|right)most\b|\bthe\s+(?:left|right)\s+(?!side\b|of\b)"
)

_WORD_RE = re.compile(r"[a-z]+")


# Instruction categories (see `classify_instruction`).
CATEGORY_REFERENT = "referent_selection"   # spatial term selects WHICH object
CATEGORY_PLACEMENT = "placement_relation"  # spatial term is a DESTINATION
CATEGORY_OTHER = "other"                    # no usable spatial term


# --------------------------------------------------------------------------- #
# Set roles
# --------------------------------------------------------------------------- #
# The role a cached frame plays. `construction` frames are the base frames the
# constructed experiments are composited from; `validation` frames are kept
# unaltered and supply the first-action object-tracking gate in Notebook 04.
#
# The two are disjoint by construction rather than by convention. A frame that
# was edited into an experimental stimulus cannot also be the instrument check
# that licenses reading the first action as object-seeking, because the two
# measurements would share the same scene, objects, and camera. Assignment
# happens once, at harvest, and is recorded in the manifest so nothing
# downstream has to re-derive it.
SPLIT_CONSTRUCTION = "construction"
SPLIT_VALIDATION = "validation"
SPLIT_VALUES = (SPLIT_CONSTRUCTION, SPLIT_VALIDATION)
# Frames cached before the roles existed carry no split and are claimed by
# neither, so an older manifest cannot silently leak into either set.
SPLIT_UNASSIGNED = ""


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


def _object_phrase(text: str) -> str:
    """Return the instruction text up to the first relational preposition.

    The chunk before that preposition is the manipulated object. A prenominal
    `left`/`right` there selects which object to act on. The same modifier after
    the preposition selects a landmark, which is a destination.
    """
    return _OBJECT_PHRASE_SPLIT_RE.split(text, maxsplit=1)[0]


def _prenominal_referent(text: str) -> bool:
    """True when left/right modifies the manipulated object, not a landmark."""
    return bool(_PRENOMINAL_REFERENT_RE.search(_object_phrase(text)))


def _categorise(tokens: list[str], phrases: list[str],
                *, words: list[str] | None = None,
                text: str = "") -> str:
    """Assign an instruction category from the shape of the matched spatial cue.

    Heuristic (transparent and approximate):
      * "placement_relation": a destination phrase is present (`to the left`,
        `to the right`, `next to`, `in front of`, `on top of`, `close to`,
        `far from`, `between`), or a transfer verb is present and the spatial
        term is not a prenominal modifier of the object (`put the cup on the
        left`, `put the sushi on the left plate`). A destination names a goal
        location, e.g. "move the cloth to the right of the colander".
      * "referent_selection": a spatial cue is present but is not a destination:
        a pickup instruction with a locative (`pick up the cup on the left`), or
        a prenominal modifier of the object even under a transfer verb (`put
        the left cup in the box`).
      * "other": no spatial cue.

    Treating `on the left` as a destination phrase would also tag pickup
    instructions as placement, so the transfer verb is what distinguishes
    "put the cup on the left" from "pick up the cup on the left". Prenominal
    position is checked only on the object phrase, so "put the sushi on the
    left plate" stays placement (the modifier is on the landmark) while "put
    the left cup in the box" stays referent.

    Known limitation: an instruction that carries both a referent modifier and
    a destination phrase (e.g. "put the cup on the left on the shelf") is
    still misread as placement only, since any matched destination phrase
    decides the category outright. Categories are annotated, not deleted, so
    manual review can correct edge cases; `category_manual` exists precisely
    for this.
    """
    if phrases:
        return CATEGORY_PLACEMENT
    if not tokens:
        return CATEGORY_OTHER
    word_set = set(words or [])
    if TRANSFER_VERBS & word_set:
        if text and _prenominal_referent(text):
            return CATEGORY_REFERENT
        return CATEGORY_PLACEMENT
    return CATEGORY_REFERENT


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
    category = _categorise(tokens, phrases, words=words, text=lowered)
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


def _as_array(value) -> np.ndarray:
    """Materialise an observation image as an HxWx3 uint8 array.

    Accepts an already-decoded tensor or array, or the encoded bytes returned
    when the dataset is built with image decoding skipped. Skipping the decode
    matters at harvest scale: an episode holds tens of frames and only one or two
    are ever read, so decoding every frame in every episode dominates the cost of
    streaming and buys nothing.
    """
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    return np.asarray(value)


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
    extract_grasp: bool = True,
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

    `extract_grasp=False` skips the grasp step entirely, so no second frame is
    decoded and the three grasp fields are None. Placement scenes are the only
    consumer of that frame, so a harvest that does not intend to analyse them
    pays nothing for it.
    """
    steps = list(episode["steps"])
    first = steps[0]

    image = _as_array(first["observation"]["image"])
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

    grasp_frame_index = grasp_index(steps) if extract_grasp else None
    grasp_image = None
    grasp_gt_vector = None
    if grasp_frame_index is not None:
        grasp_step = steps[grasp_frame_index]
        grasp_image = _as_array(grasp_step["observation"]["image"])
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


def release_components(version_text: str) -> tuple[int, ...]:
    """The leading numeric components of a version string, suffixes discarded."""
    return tuple(int(part) for part in re.findall(r"\d+", version_text)[:3])


def protobuf_runtime_problem(version_text: str) -> str | None:
    """An actionable message when a protobuf runtime cannot serve the TFDS stack.

    Returns None when the version falls inside the supported window, or when it
    cannot be parsed, in which case the stack is left to report its own failure
    rather than being blocked by a version string this function does not recognise.
    """
    version = release_components(version_text)
    if not version:
        return None
    if version < PROTOBUF_MIN:
        cause = (
            "older than the tensorflow-metadata generated code, which protobuf "
            "refuses to run on a lower runtime"
        )
    elif version >= PROTOBUF_MAX_EXCLUSIVE:
        cause = (
            "newer than released TensorFlow Datasets supports: protobuf 7 removed "
            "`FieldDescriptor.label`, which TFDS uses to read dataset metadata, so "
            "streaming fails with an AttributeError once a builder is opened"
        )
    else:
        return None
    return (
        f"The protobuf runtime ({version_text}) is {cause}. Install a runtime "
        f"inside the supported window and restart the session:\n"
        f'    pip install "{PROTOBUF_SPEC}"\n'
        f"The restart is required because protobuf caches generated modules for the "
        f"life of the interpreter."
    )


def installed_protobuf_version() -> str:
    """The installed protobuf runtime version, or an empty string when absent."""
    try:
        import google.protobuf as protobuf
    except ImportError:
        return ""
    return getattr(protobuf, "__version__", "")


def stream_episodes(n: int, split_start: int = 0, early_steps: int = 5,
                     post_grasp_steps: int = 5, extract_grasp: bool = True,
                     lazy_images: bool = True):
    """Yield EpisodeRecord objects from the GCS mirror, one episode at a time.

    Imported lazily so the TensorFlow/TFDS stack is only required when streaming,
    keeping this module importable in environments that hold only the model deps.

    `lazy_images` asks the dataset not to decode the step images, leaving them as
    encoded bytes that `_as_array` decodes only for the frames actually read. An
    episode holds tens of frames and at most two are used, so this removes most of
    the streaming cost, which matters once the harvest runs to thousands of
    episodes. The request is dropped with a warning if the installed TFDS rejects
    the decoder specification, since a slower stream is better than none.
    """
    # The TFDS stack is usable only against a bounded range of protobuf runtimes,
    # and only one edge announces itself at import: a runtime that is too old raises
    # a version error, while one that is too new imports cleanly and fails later on
    # a removed descriptor attribute. Checking the version first reports both edges
    # in the same actionable form, before any download begins.
    problem = protobuf_runtime_problem(installed_protobuf_version())
    if problem:
        raise RuntimeError(problem)

    builder_from_directory = None
    try:
        import tensorflow_datasets as tfds

        # `builder_from_directory` is a re-export whose public aliases
        # (`tfds.builder_from_directory`, `tfds.core.builder_from_directory`) are
        # absent in some TFDS builds. It is defined in
        # `tensorflow_datasets.core.read_only_builder`, so fall back to importing
        # it from there before giving up.
        builder_from_directory = getattr(tfds, "builder_from_directory", None)
        if builder_from_directory is None:
            builder_from_directory = getattr(
                getattr(tfds, "core", None), "builder_from_directory", None)
        if builder_from_directory is None:
            from tensorflow_datasets.core.read_only_builder import (
                builder_from_directory,
            )
    except Exception as exc:  # noqa: BLE001 - environment-dependent import chain
        message = str(exc)
        if "Protobuf" in message or "runtime_version" in message:
            raise RuntimeError(
                "TensorFlow Datasets failed to import against the installed "
                f"protobuf runtime ({installed_protobuf_version() or 'unknown'}). "
                "Install a runtime inside the supported window and restart the "
                f'session:\n    pip install "{PROTOBUF_SPEC}"\n'
                "then restart the runtime before re-running."
            ) from exc
        raise
    if builder_from_directory is None:
        raise AttributeError(
            "tensorflow_datasets exposes no builder_from_directory in this "
            "environment. Upgrade the package with `pip install -U "
            "tensorflow_datasets` and restart the runtime."
        )

    builder = builder_from_directory(BRIDGE_GCS)
    split = f"train[{split_start}:{split_start + n}]"
    dataset = None
    if lazy_images:
        try:
            decoders = {"steps": {"observation": {"image": tfds.decode.SkipDecoding()}}}
            dataset = builder.as_dataset(split=split, decoders=decoders)
        except Exception as exc:  # noqa: BLE001 - TFDS build-dependent
            print(f"[stream_episodes] image decoding could not be skipped "
                  f"({type(exc).__name__}); streaming with full decoding, which "
                  f"is slower per episode")
    if dataset is None:
        dataset = builder.as_dataset(split=split)
    for offset, episode in enumerate(dataset):
        yield extract_episode(episode, episode_index=split_start + offset,
                              early_steps=early_steps,
                              post_grasp_steps=post_grasp_steps,
                              extract_grasp=extract_grasp)


# --------------------------------------------------------------------------- #
# Antonym-swapped minimal pairs
# --------------------------------------------------------------------------- #
# Longer phrases first so the compiled regex prefers "in front of" over "front".
ANTONYM_PAIRS = [
    ("in front of", "behind"),
    ("closer to", "farther from"),
    ("nearer to", "farther from"),
    ("leftmost", "rightmost"),
    ("nearest", "farthest"),
    ("left", "right"),
    ("top", "bottom"),
    ("front", "back"),
]

SWAP: dict[str, str] = {}
for _a, _b in ANTONYM_PAIRS:
    SWAP.setdefault(_a, _b)
    SWAP.setdefault(_b, _a)

# Spatial axis each antonym pair contrasts, and the index of the translation
# component in the action 7-vector [dx, dy, dz, ...] on which a correct response
# to the swap should appear.
#
# The probe previously read dx for every pair, which is only right for the
# lateral terms: a front/back or top/bottom swap has no reason to change the
# lateral component, so pooling them onto dx adds noise to the measurement. The
# axis is also what decides which terms admit a mirror control, since a
# horizontal flip of the image reverses the lateral axis and leaves the other
# two unchanged.
AXIS_LATERAL = "lateral"     # dx, reversed by a horizontal image flip
AXIS_DEPTH = "depth"         # dy, unchanged by a horizontal image flip
AXIS_VERTICAL = "vertical"   # dz, unchanged by a horizontal image flip
# Terms whose contrasted axis depends on the scene rather than on the term.
# "closer to the plate" points wherever the plate happens to be, so no fixed
# component can be assigned in advance. These are excluded from axis-specific
# analysis rather than assigned a plausible-looking default.
AXIS_SCENE_DEPENDENT = "scene_dependent"

AXIS_INDEX = {AXIS_LATERAL: 0, AXIS_DEPTH: 1, AXIS_VERTICAL: 2}

TERM_AXIS: dict[str, str] = {
    "left": AXIS_LATERAL,
    "right": AXIS_LATERAL,
    "leftmost": AXIS_LATERAL,
    "rightmost": AXIS_LATERAL,
    # Depth relations. The camera looks across the workspace, so front/back and
    # the bare distance superlatives contrast the axis running away from the
    # robot base.
    "in front of": AXIS_DEPTH,
    "behind": AXIS_DEPTH,
    "front": AXIS_DEPTH,
    "back": AXIS_DEPTH,
    "nearest": AXIS_DEPTH,
    "farthest": AXIS_DEPTH,
    "top": AXIS_VERTICAL,
    "bottom": AXIS_VERTICAL,
    # Distance to a named landmark: direction is set by where the landmark sits.
    "closer to": AXIS_SCENE_DEPENDENT,
    "nearer to": AXIS_SCENE_DEPENDENT,
    "farther from": AXIS_SCENE_DEPENDENT,
}

# Every swappable term must carry an axis, or a pair could reach the analysis
# with no defined expectation. Guards against a term being added to
# ANTONYM_PAIRS without a matching entry here.
_missing_axis = sorted(set(SWAP) - set(TERM_AXIS))
if _missing_axis:
    raise RuntimeError(
        f"ANTONYM_PAIRS terms without a TERM_AXIS entry: {_missing_axis}"
    )


def term_axis(term: str) -> str | None:
    """Spatial axis a swappable term contrasts, or None if the term is unknown."""
    return TERM_AXIS.get(term.lower().strip())


def term_axis_index(term: str) -> int | None:
    """Translation component index a term's swap should act on.

    Returns None for unknown terms and for scene-dependent relations, which have
    no axis that can be fixed ahead of seeing the scene.
    """
    axis = term_axis(term)
    return AXIS_INDEX.get(axis) if axis is not None else None


def is_lateral_term(term: str) -> bool:
    """Whether a term contrasts the lateral axis, the axis a mirror flip reverses."""
    return term_axis(term) == AXIS_LATERAL

_SWAP_KEYS = sorted(SWAP, key=len, reverse=True)
_SWAP_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _SWAP_KEYS) + r")\b",
    re.IGNORECASE,
)


def make_pair(instruction: str):
    """Return (term, swapped) if exactly one swappable spatial phrase occurs.

    Instructions with zero or multiple swappable terms return None, so a
    multi-term swap cannot break the minimal-pair property.
    """
    hits = _SWAP_RE.findall(instruction)
    if len(hits) != 1:
        return None
    term = hits[0].lower()
    swapped = _SWAP_RE.sub(lambda m: SWAP[m.group(0).lower()], instruction)
    return term, swapped


# --------------------------------------------------------------------------- #
# Drive cache (filtered survivors only)
# --------------------------------------------------------------------------- #
# Allowed values for the `feasible_both` annotation field.
FEASIBLE_VALUES = frozenset({"yes", "no", "unclear", "unreviewed"})
# Default value written at harvest (see update_manifest_annotations).
FEASIBLE_DEFAULT = "unreviewed"

# Allowed values for the duplicate-target field. A scene is a clean referent
# probe only when it contains two or more instances of the same object, so the
# antonym-swapped instruction names a distinct, real target on both sides.
DUPLICATE_VALUES = frozenset({"yes", "no", "unclear", "unreviewed"})
# Default value for the duplicate-target field before any proposal or review.
DUPLICATE_DEFAULT = "unreviewed"

# category_source values: the heuristic default, or manual once category_manual
# is set (see update_manifest_annotations).
CATEGORY_SOURCE_HEURISTIC = "heuristic"
CATEGORY_SOURCE_MANUAL = "manual"

# duplicate_source values: empty before any label, `auto` when written by the
# open-vocabulary detector pass, or `manual` when confirmed in manual review.
DUPLICATE_SOURCE_AUTO = "auto"
DUPLICATE_SOURCE_MANUAL = "manual"

MANIFEST_FIELDS = [
    "episode_index", "instruction", "category", "is_multi_object", "has_spatial",
    "has_transfer", "matched_terms", "num_steps", "image_path", "split",
    "feasible_both", "feasibility_note",
    "duplicate_target", "duplicate_note", "duplicate_source", "duplicate_score",
    "gt_dx", "gt_dy", "gt_dz", "gt_rx", "gt_ry", "gt_rz", "gt_gripper",
    "grasp_frame_index", "grasp_image_path",
    "grasp_gt_dx", "grasp_gt_dy", "grasp_gt_dz",
    "category_manual", "category_source",
]


def validation_eligible(instruction: str) -> bool:
    """Whether an instruction can supply an unaltered object-tracking trial.

    Three requirements, all on the wording, so the decision can be taken at
    harvest before any frame is inspected. The instruction must yield exactly one
    antonym swap, that swap must contrast the lateral axis, and the term must
    select a referent rather than name a destination. Stripping the term then
    leaves a well-formed object-seeking prompt (`pick up the cup`), which is the
    instruction the first-action gate in Notebook 04 runs.

    Roughly one instruction in two hundred qualifies, which is why the validation
    set is harvested across many more episodes than it keeps.
    """
    made = make_pair(instruction)
    if made is None:
        return False
    term, _ = made
    if not is_lateral_term(term):
        return False
    return classify_instruction(instruction).category == CATEGORY_REFERENT


def manifest_row(record, *, split: str = SPLIT_UNASSIGNED,
                 image_path: str = "", grasp_image_path: str = "") -> dict:
    """Build one manifest row from an EpisodeRecord.

    Shared by every caching path so a row cannot be assembled two ways. Review
    and annotation fields are written at their defaults rather than left out,
    because the manifest annotates rather than deletes and later passes fill them
    in place.
    """
    gt = record.gt_vector
    grasp_frame_index = ""
    grasp_dx = grasp_dy = grasp_dz = ""
    if record.grasp_frame_index is not None:
        grasp_frame_index = record.grasp_frame_index
        ggt = record.grasp_gt_vector
        grasp_dx, grasp_dy, grasp_dz = ggt[0], ggt[1], ggt[2]
    return {
        "episode_index": record.episode_index,
        "instruction": record.instruction,
        "category": record.tags.category,
        "is_multi_object": record.tags.is_multi_object,
        "has_spatial": record.tags.has_spatial,
        "has_transfer": record.tags.has_transfer,
        "matched_terms": record.tags.matched,
        "num_steps": record.num_steps,
        "image_path": image_path,
        "split": split,
        # Feasibility is filled in later by manual review, not deleted.
        "feasible_both": FEASIBLE_DEFAULT,
        "feasibility_note": "",
        # Duplicate-target is filled in later by the automated proposal pass and
        # manual confirmation, not deleted.
        "duplicate_target": DUPLICATE_DEFAULT,
        "duplicate_note": "",
        "duplicate_source": "",
        "duplicate_score": "",
        "gt_dx": gt[0], "gt_dy": gt[1], "gt_dz": gt[2],
        "gt_rx": gt[3], "gt_ry": gt[4], "gt_rz": gt[5],
        "gt_gripper": gt[6],
        "grasp_frame_index": grasp_frame_index,
        "grasp_image_path": grasp_image_path,
        "grasp_gt_dx": grasp_dx, "grasp_gt_dy": grasp_dy, "grasp_gt_dz": grasp_dz,
        "category_manual": "",
        "category_source": CATEGORY_SOURCE_HEURISTIC,
    }


def append_manifest_rows(out_dir: str, rows) -> str:
    """Append rows to the manifest, creating or widening the header as needed.

    Appending under an assumed set of column names is the failure that corrupted
    the prediction log: a writer given more fields than the header declares emits
    the row without complaint, and the file is unreadable from that point on while
    nothing reports an error. The header is therefore read first, and a file whose
    header does not already cover every column being written is rewritten under the
    union before anything is appended. Columns the file carries and the current
    schema does not are preserved rather than dropped, since a manifest may hold
    annotations added by a later pass.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.csv")
    rows = list(rows)
    if not rows:
        return path

    fields = list(MANIFEST_FIELDS)
    existing_header = None
    if os.path.isfile(path):
        with open(path, newline="") as f:
            existing_header = next(csv.reader(f), None)
    if existing_header:
        fields = list(MANIFEST_FIELDS) + [c for c in existing_header
                                          if c not in MANIFEST_FIELDS]
        if existing_header != fields:
            with open(path, newline="") as f:
                current = [dict(r) for r in csv.DictReader(f)]
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in current:
                    writer.writerow({k: row.get(k, "") for k in fields})
            os.replace(tmp, path)
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path


# --------------------------------------------------------------------------- #
# Harvest
# --------------------------------------------------------------------------- #
# Streaming is sequential and the validation criteria are rare, so a harvest that
# reaches a useful validation count runs for thousands of episodes and across more
# than one session. The position reached is therefore recorded next to the frames:
# the manifest cannot supply it, because episodes that were streamed and cached by
# neither role leave no row, so the last cached index understates how far the
# stream got and a resumed run would re-stream what it already rejected.
HARVEST_STATE_NAME = "harvest_state.json"

# Rows are appended in batches, since the cache lives on a network-mounted Drive
# where a write per frame is slow. The batch is small enough that an interrupted
# session loses seconds of work.
HARVEST_FLUSH_EVERY = 25


def read_harvest_state(out_dir: str) -> dict:
    """Read the recorded harvest position, or an empty state."""
    path = os.path.join(out_dir, HARVEST_STATE_NAME)
    if not os.path.isfile(path):
        return {"episodes_streamed": 0, "next_split_start": 0}
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return {"episodes_streamed": 0, "next_split_start": 0}
    state.setdefault("episodes_streamed", 0)
    state.setdefault("next_split_start", 0)
    return state


def write_harvest_state(out_dir: str, state: dict) -> str:
    """Record the harvest position, replacing the file atomically."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, HARVEST_STATE_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def split_counts(rows) -> dict:
    """Count manifest rows by split."""
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("split", SPLIT_UNASSIGNED) or SPLIT_UNASSIGNED
        counts[key] = counts.get(key, 0) + 1
    return counts


def harvest_records(
    records,
    out_dir: str,
    *,
    validation_target: int | None,
    construction_target: int | None,
    cache_grasp: bool = False,
    flush_every: int = HARVEST_FLUSH_EVERY,
    verbose: bool = True,
) -> dict:
    """Cache streamed episodes into the two roles, appending and resumable.

    Every frame whose instruction satisfies `validation_eligible` is cached as
    validation and is never offered to construction, whatever the validation
    target. Holding the rule rather than the count fixed is what keeps the roles
    disjoint under a later decision to enlarge the object-tracking pool: if
    eligible frames spilled into construction once the target was met, growing
    that pool afterwards would mean either re-streaming or admitting frames that
    had already been edited into stimuli. Remaining frames become
    construction bases until `construction_target` is reached, after which
    streaming continues for validation alone.

    `validation_target` and `construction_target` decide when to stop, not what to
    admit. `None` for either means unbounded. No language filter is applied to the
    construction pool: a constructed trial writes its own instruction over the
    episode's manipulated object, so requiring the source wording to carry a
    spatial or transfer structure would discard usable frames for a property the
    stimulus does not inherit.

    Frames already in the manifest are skipped, so a re-run resumes rather than
    duplicating, and rows are appended in batches with the stream position
    recorded alongside so an interrupted session loses only the current batch.
    """
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    existing = load_manifest(out_dir) if os.path.isfile(
        os.path.join(out_dir, "manifest.csv")) else []
    seen = {str(row["episode_index"]) for row in existing}
    counts = split_counts(existing)
    validation = counts.get(SPLIT_VALIDATION, 0)
    construction = counts.get(SPLIT_CONSTRUCTION, 0)
    state = read_harvest_state(out_dir)

    summary = {
        "streamed": 0,
        "skipped_existing": 0,
        "cached": {SPLIT_VALIDATION: 0, SPLIT_CONSTRUCTION: 0},
        "totals": {SPLIT_VALIDATION: validation, SPLIT_CONSTRUCTION: construction},
        "manifest_path": os.path.join(out_dir, "manifest.csv"),
    }

    def targets_met() -> bool:
        val_done = validation_target is not None and validation >= validation_target
        con_done = (construction_target is not None
                    and construction >= construction_target)
        return val_done and con_done

    pending: list[dict] = []

    def flush() -> None:
        if pending:
            append_manifest_rows(out_dir, pending)
            pending.clear()
        write_harvest_state(out_dir, state)

    if targets_met():
        if verbose:
            print(f"[harvest_records] targets already met: "
                  f"{validation} validation, {construction} construction")
        return summary

    for record in records:
        summary["streamed"] += 1
        state["next_split_start"] = int(record.episode_index) + 1
        state["episodes_streamed"] = state.get("episodes_streamed", 0) + 1

        if str(record.episode_index) in seen:
            summary["skipped_existing"] += 1
            continue

        if validation_eligible(record.instruction):
            split = SPLIT_VALIDATION
        else:
            if construction_target is not None and construction >= construction_target:
                continue
            split = SPLIT_CONSTRUCTION

        rel = os.path.join("frames", f"ep_{int(record.episode_index):06d}.png")
        Image.fromarray(record.image).save(os.path.join(out_dir, rel))
        grasp_rel = ""
        if cache_grasp and record.grasp_image is not None:
            grasp_rel = os.path.join(
                "frames", f"ep_{int(record.episode_index):06d}_grasp.png")
            Image.fromarray(record.grasp_image).save(os.path.join(out_dir, grasp_rel))

        pending.append(manifest_row(record, split=split, image_path=rel,
                                    grasp_image_path=grasp_rel))
        seen.add(str(record.episode_index))
        if split == SPLIT_VALIDATION:
            validation += 1
        else:
            construction += 1
        summary["cached"][split] += 1

        if len(pending) >= flush_every:
            flush()
            if verbose:
                print(f"[harvest_records] {summary['streamed']} streamed, "
                      f"{validation} validation, {construction} construction")
        if targets_met():
            break

    flush()
    summary["totals"] = {SPLIT_VALIDATION: validation,
                         SPLIT_CONSTRUCTION: construction}
    summary["next_split_start"] = state["next_split_start"]
    summary["targets_met"] = targets_met()
    if verbose:
        print(f"[harvest_records] streamed {summary['streamed']} episodes, cached "
              f"{summary['cached'][SPLIT_VALIDATION]} validation and "
              f"{summary['cached'][SPLIT_CONSTRUCTION]} construction")
        print(f"[harvest_records] totals: {summary['totals']}, next split_start "
              f"{summary['next_split_start']}, targets met "
              f"{summary['targets_met']}")
    return summary


# Harvest sizing. The validation set is scarce by nature, so its target is a
# count and the episode budget follows from it rather than the reverse. The
# construction target is the pool the constructed experiments are drawn from: only
# a fraction of base frames survive detection, surface location, and placement, so
# the pool has to exceed the number of experimental scenes wanted by several times.
# The multiple is not assumed. `compose_scenes.project_yield` measures it from the
# set already built, and the harvest is sized from that projection, since the
# binding rate is the share of frames yielding a within-frame pair whose two scenes
# both pass the screen, and that compounds well below the per-scene rate.
DEFAULT_VALIDATION_TARGET = 50
DEFAULT_CONSTRUCTION_TARGET = 16000


def harvest(
    out_dir: str,
    *,
    episodes: int,
    split_start: int | None = None,
    validation_target: int = DEFAULT_VALIDATION_TARGET,
    construction_target: int | None = DEFAULT_CONSTRUCTION_TARGET,
    cache_grasp: bool = False,
    early_steps: int = 5,
    post_grasp_steps: int = 5,
    verbose: bool = True,
) -> dict:
    """Stream up to `episodes` episodes and cache them into the two roles.

    `split_start` defaults to the position recorded by the previous run, so
    repeated calls walk forward through the corpus instead of re-streaming what
    was already rejected. Returns the `harvest_records` summary.
    """
    if split_start is None:
        split_start = read_harvest_state(out_dir).get("next_split_start", 0)
    if verbose:
        print(f"[harvest] streaming train[{split_start}:{split_start + episodes}] "
              f"-> {out_dir}")
    records = stream_episodes(n=episodes, split_start=split_start,
                              early_steps=early_steps,
                              post_grasp_steps=post_grasp_steps,
                              extract_grasp=cache_grasp)
    return harvest_records(records, out_dir,
                           validation_target=validation_target,
                           construction_target=construction_target,
                           cache_grasp=cache_grasp, verbose=verbose)


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
# sets category_source to CATEGORY_SOURCE_MANUAL in the same call. The
# duplicate_source is written explicitly by the caller (the auto pass passes
# `auto`, the review UI passes `manual`).
ANNOTATION_FIELDS = frozenset({
    "feasible_both", "feasibility_note", "category_manual", "category_source",
    "duplicate_target", "duplicate_note", "duplicate_source", "duplicate_score",
})


def update_manifest_annotations(out_dir: str, annotations: dict) -> str:
    """Rewrite the manifest with manual annotations filled in.

    Annotates rather than deletes: contrastive prompts built by antonym swap can
    imply physically infeasible placements, so each scene is reviewed for whether
    the implied placement is possible on BOTH sides of the referent. The same
    reviewed path also carries manual category correction, since a spatial term
    that both selects a referent and names a destination is misread by
    `_categorise`; this function never overwrites the heuristic `category`
    column (`refresh_heuristic_categories` is the dedicated rewrite for that
    column), and downstream code reads `category_manual` when populated,
    falling back to `category`.

    Args:
        out_dir: directory holding manifest.csv.
        annotations: {episode_index: {field: value, ...}}, where field is one
            of ANNOTATION_FIELDS (`feasible_both`, `feasibility_note`,
            `category_manual`, `category_source`, `duplicate_target`,
            `duplicate_note`, `duplicate_source`, `duplicate_score`). Keys may
            be int or str; only the provided fields are overwritten, other rows
            and other fields keep their existing values. Setting
            `category_manual` sets `category_source` to `manual` regardless of
            any explicit `category_source` value passed in the same note. A
            `feasible_both` value outside FEASIBLE_VALUES or a
            `duplicate_target` value outside DUPLICATE_VALUES raises ValueError.
            `duplicate_source` is written as provided (the auto pass passes
            `auto`, the review UI passes `manual`).

    Returns the manifest path.
    """
    manifest_path = os.path.join(out_dir, "manifest.csv")
    rows = load_manifest(out_dir)
    # Normalise keys to strings so int/str episode indices both match.
    ann = {str(k): v for k, v in annotations.items()}

    for ep_key, note in ann.items():
        if "feasible_both" in note and note["feasible_both"] not in FEASIBLE_VALUES:
            raise ValueError(
                f"feasible_both for episode {ep_key} must be one of "
                f"{sorted(FEASIBLE_VALUES)}, got {note['feasible_both']!r}"
            )
        if "duplicate_target" in note and note["duplicate_target"] not in DUPLICATE_VALUES:
            raise ValueError(
                f"duplicate_target for episode {ep_key} must be one of "
                f"{sorted(DUPLICATE_VALUES)}, got {note['duplicate_target']!r}"
            )

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


def refresh_heuristic_categories(out_dir: str) -> dict:
    """Recompute the heuristic `category` column from the current classifier.

    Harvest wrote `category` under whatever rule was current then. A later
    tightening of `_categorise` therefore leaves the stored column stale, and
    anything that reads the CSV directly (the probe, pair export) would keep
    the old label. This rewrite updates that column in place. `category_manual`
    is not touched, so a reviewer's override remains the effective category.

    Returns a dict with `updated`, `total`, and `manifest_path`.
    """
    manifest_path = os.path.join(out_dir, "manifest.csv")
    rows = load_manifest(out_dir)
    updated = 0
    for row in rows:
        new = classify_instruction(row.get("instruction", "")).category
        if row.get("category") != new:
            row["category"] = new
            updated += 1
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    print(f"[refresh_heuristic_categories] updated {updated} of {len(rows)} "
          f"rows in {manifest_path}")
    return {
        "updated": updated,
        "total": len(rows),
        "manifest_path": manifest_path,
    }


def _effective_category(row: dict) -> str:
    """Return category_manual when set, otherwise the current heuristic.

    The current classifier is applied to the instruction rather than the
    stored `category` column, so a harvest written under an older rule does
    not keep destination-locative transfer instructions in the referent
    queue. An empty instruction falls back to the stored column.
    """
    if row.get("category_manual"):
        return row["category_manual"]
    instruction = row.get("instruction") or ""
    if instruction:
        return classify_instruction(instruction).category
    return row.get("category") or CATEGORY_OTHER


def _review_item(row: dict) -> dict:
    """Build a review-queue record from a manifest row."""
    instruction = row.get("instruction", "")
    made = make_pair(instruction)
    if made is None:
        term, instr_b, pairable = "", "", False
    else:
        term, instr_b = made
        pairable = True
    return {
        "episode_index": row["episode_index"],
        "split": row.get("split", SPLIT_UNASSIGNED) or SPLIT_UNASSIGNED,
        "category": _effective_category(row),
        "instruction": instruction,
        "instr_b": instr_b,
        "spatial_term": term,
        "pairable": pairable,
        "image_path": row.get("image_path", ""),
        "grasp_image_path": row.get("grasp_image_path", ""),
        "feasible_both": row.get("feasible_both") or FEASIBLE_DEFAULT,
        "feasibility_note": row.get("feasibility_note", ""),
        "duplicate_target": row.get("duplicate_target") or DUPLICATE_DEFAULT,
        "duplicate_note": row.get("duplicate_note", ""),
        "duplicate_source": row.get("duplicate_source", ""),
        "duplicate_score": row.get("duplicate_score", ""),
        "category_manual": row.get("category_manual", ""),
    }


def review_summary(out_dir: str, *, splits=(SPLIT_VALIDATION,)) -> dict:
    """Summarise annotation progress for a cached manifest.

    Counters cover the validation split by default, which is the object-tracking
    pool. Construction frames are not annotated for source-wording feasibility:
    the constructed scene writes its own instruction and its arrangement is
    chosen rather than found, and the constructed set has its own approval pass
    in `compose_scenes.py`. Pass `splits=None` to count every row.

    Returns a dict with:
      * `by_split`: {split: count} over the whole manifest, unfiltered
      * `reviewed_scope`: the splits the counters below cover
      * `by_category_feasibility`: {(category, feasible_both): count}
      * `pairable`: count of scenes with a valid antonym pair
      * `non_pairable`: count of scenes without a valid antonym pair
      * `referent_pairable_unreviewed`: pairable referent_selection still unreviewed
      * `referent_pairable_yes`: pairable referent_selection marked feasible
      * `referent_pairable_dup_unreviewed`: pairable referent_selection whose
        duplicate_target is still unreviewed
      * `referent_pairable_dup_yes`: pairable referent_selection with two or more
        identical objects (duplicate_target == 'yes')
      * `referent_pairable_dup_unclear`: pairable referent_selection whose
        duplicate_target is unclear and awaiting confirmation
      * `duplicate_source`: {source: count} across pairable referent_selection
        scenes, showing how many labels came from the auto pass versus review
      * `primary_eligible`: pairable referent_selection scenes that satisfy the
        headline stratum (feasible_both == 'yes' and duplicate_target == 'yes')
    """
    all_rows = load_manifest(out_dir)
    by_split = split_counts(all_rows)
    allowed = set(splits) if splits is not None else None
    rows = [r for r in all_rows
            if allowed is None
            or (r.get("split", SPLIT_UNASSIGNED) or SPLIT_UNASSIGNED) in allowed]
    by_cat_feas: dict[tuple[str, str], int] = {}
    pairable = 0
    non_pairable = 0
    referent_unreviewed = 0
    referent_yes = 0
    referent_dup_unreviewed = 0
    referent_dup_yes = 0
    referent_dup_unclear = 0
    dup_source: dict[str, int] = {}
    primary_eligible = 0
    for row in rows:
        item = _review_item(row)
        key = (item["category"], item["feasible_both"])
        by_cat_feas[key] = by_cat_feas.get(key, 0) + 1
        if item["pairable"]:
            pairable += 1
            if item["category"] == CATEGORY_REFERENT:
                if item["feasible_both"] == "unreviewed":
                    referent_unreviewed += 1
                elif item["feasible_both"] == "yes":
                    referent_yes += 1
                dup = item["duplicate_target"]
                if dup == "unreviewed":
                    referent_dup_unreviewed += 1
                elif dup == "yes":
                    referent_dup_yes += 1
                elif dup == "unclear":
                    referent_dup_unclear += 1
                src = item["duplicate_source"] or ""
                dup_source[src] = dup_source.get(src, 0) + 1
                if item["feasible_both"] == "yes" and dup == "yes":
                    primary_eligible += 1
        else:
            non_pairable += 1
    return {
        "by_split": by_split,
        "reviewed_scope": tuple(sorted(allowed)) if allowed is not None else "all",
        "by_category_feasibility": by_cat_feas,
        "pairable": pairable,
        "non_pairable": non_pairable,
        "referent_pairable_unreviewed": referent_unreviewed,
        "referent_pairable_yes": referent_yes,
        "referent_pairable_dup_unreviewed": referent_dup_unreviewed,
        "referent_pairable_dup_yes": referent_dup_yes,
        "referent_pairable_dup_unclear": referent_dup_unclear,
        "duplicate_source": dup_source,
        "primary_eligible": primary_eligible,
        "in_scope": len(rows),
        "total": len(all_rows),
    }


def review_queue(
    out_dir: str,
    *,
    status: str | None = "unreviewed",
    duplicate_status: str | None = None,
    categories: list[str] | None = None,
    only_pairable: bool = True,
    splits=(SPLIT_VALIDATION,),
    limit: int | None = None,
) -> list[dict]:
    """Return an ordered list of scenes for manual review.

    Scoped to the validation split by default, because that is the set this review
    qualifies: the constructed experiments carry their own approval pass, and a
    frame reserved as a construction base is never shown as a validation trial.
    Frames cached before the roles existed carry no split and are therefore in
    neither scope; pass `splits=None` to review them.

    Priority order:
      1. referent_selection, pairable, matching status
      2. placement_relation with a grasp frame, pairable, matching status
      3. remaining pairable spatial scenes matching status
      4. non-pairable scenes only when only_pairable is False

    Args:
        out_dir: directory holding manifest.csv and frames/.
        status: filter on feasible_both; None keeps every status. Must be a
            member of FEASIBLE_VALUES when set.
        duplicate_status: filter on duplicate_target; None keeps every value.
            Must be a member of DUPLICATE_VALUES when set. Setting this to
            `unclear` focuses a session on the scenes the automated proposal
            pass flagged as borderline, so human effort confirms only those.
        categories: optional allow-list of effective categories; None keeps all.
        only_pairable: when True, drop scenes that do not form a minimal pair.
        splits: allow-list of manifest splits; None keeps every split.
        limit: optional maximum number of items to return.
    """
    if status is not None and status not in FEASIBLE_VALUES:
        raise ValueError(
            f"status must be one of {sorted(FEASIBLE_VALUES)} or None, "
            f"got {status!r}"
        )
    if duplicate_status is not None and duplicate_status not in DUPLICATE_VALUES:
        raise ValueError(
            f"duplicate_status must be one of {sorted(DUPLICATE_VALUES)} or "
            f"None, got {duplicate_status!r}"
        )
    cat_allow = set(categories) if categories is not None else None

    items = [_review_item(row) for row in load_manifest(out_dir)]
    if splits is not None:
        allowed = set(splits)
        items = [it for it in items if it["split"] in allowed]
    if status is not None:
        items = [it for it in items if it["feasible_both"] == status]
    if duplicate_status is not None:
        items = [it for it in items if it["duplicate_target"] == duplicate_status]
    if cat_allow is not None:
        items = [it for it in items if it["category"] in cat_allow]
    if only_pairable:
        items = [it for it in items if it["pairable"]]

    def _priority(it: dict) -> tuple:
        cat = it["category"]
        has_grasp = bool(it["grasp_image_path"])
        pairable = it["pairable"]
        if pairable and cat == CATEGORY_REFERENT:
            bucket = 0
        elif pairable and cat == CATEGORY_PLACEMENT and has_grasp:
            bucket = 1
        elif pairable:
            bucket = 2
        else:
            bucket = 3
        # Stable secondary key by episode index for resume-friendly order.
        try:
            ep = int(it["episode_index"])
        except (TypeError, ValueError):
            ep = 0
        return (bucket, ep)

    items.sort(key=_priority)
    if limit is not None:
        items = items[:limit]
    return items
