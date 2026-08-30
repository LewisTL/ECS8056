"""
compose_scenes.py: constructed two-instance scenes for the spatial-language probe.

A spatial term is the sole disambiguator only when a scene holds two instances of
the same object, so that "the cup on the left" and "the cup on the right" each
name a distinct, real target. Any other arrangement lets a model reach the right
answer by finding the only cup, which is object detection rather than spatial
language. BridgeData V2 rarely contains such a scene: of 2239 cached frames, the
filtered set that satisfies the condition is in single figures, and filtering can
only discard scenes, never create the ones the measurement needs.

This module builds them. A base frame holding one confident instance of the
target noun is taken, the instance is cut from its own pixels, mirrored, and
pasted at a controlled position. The result keeps the real camera pose, lighting,
and background while placing two instances at positions that are known because
they were chosen, so the expected direction comes from recorded geometry rather
than from the instruction text.

Three configurations are produced:

  * `opposite`: one instance either side of the start position. The classic
    well-posed trial.
  * `same_side_left` and `same_side_right`: both instances on one side. A model
    that grounds the term keeps one sign for both instructions here and differs
    only in magnitude, while a fixed word-to-direction mapping produces opposite
    signs. This configuration is what separates the two accounts, and it cannot
    be sampled from found data.

These scenes carry the experiments. The unaltered Bridge frames supply the
first-action object-tracking gate, and the harvest keeps the two roles disjoint, so
`select_base_scenes` builds only from frames assigned the construction role.

Three stages stand between a built scene and an experimental one, and each is
recorded rather than assumed. Every scene is screened once by a human against
criteria fixed in advance, blind to the model and before any prediction exists
(`sync_review`, `approval_summary`). The screened set is then frozen to a fixed
list (`select_evaluation_set`, `write_evaluation_set`) that every later stage
reads, so the set cannot drift as review continues. And a pasted object is only
useful if the model perceives it, which `manipulation_rate` measures rather than
presumes.

Two caveats are handled explicitly rather than assumed away. The relationship
between image x and the sign of the lateral action is not established by the
dataset, so the manifest records the expectation in image coordinates and the
analysis applies the convention separately (`IMAGE_X_TO_LATERAL_SIGN`). And the
screen conditions the result on stimulus quality: what is estimated is grounding
given a well-posed and perceptible scene, so rejected scenes are kept rather than
deleted and remain available as a check on what the screen bought.

The detector is imported lazily through `detect_duplicates`, so the geometry and
compositing helpers stay importable without a model download.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter

from data import (CATEGORY_REFERENT, SPLIT_CONSTRUCTION, is_lateral_term,
                  make_pair)
# Only the threshold constant is taken at import time, so the detector's heavy
# dependencies stay behind the lazy imports inside the functions that call it.
from detect_duplicates import DEFAULT_SCORE_THRESH

# Sign relating image x to the lateral action component: +1 if a target further
# right in the image implies a larger value on that component. The axis
# identification scored every component under both signs against scene geometry,
# on the demonstration ground truth and on the model, and fixed the lateral
# channel at component 1 (dy in the OpenVLA layout naming) with this sign
# inverted: a target further right in the image implies a more negative value.
# The constructed manifest stores the expectation in image coordinates only, so
# revising this constant never requires regenerating scenes.
IMAGE_X_TO_LATERAL_SIGN = -1

# Index of the identified lateral channel in the action translation. The layout
# naming (dx, dy, dz) does not describe image geometry; only this component has
# an established image reading, so overlays draw it alone rather than dressing
# the other two components in a geometry nothing has verified.
LATERAL_ACTION_INDEX = 1


def action_image_delta(c0, c1, c2, *, scale: float = 1.0,
                       lateral_sign: int = IMAGE_X_TO_LATERAL_SIGN):
    """Map a predicted translation onto image-pixel offsets.

    Only the identified lateral channel (component `LATERAL_ACTION_INDEX`) has
    an established image reading, so it alone is projected, onto image x
    through `lateral_sign`. The other two components have no identified image
    direction and contribute nothing; an overlay drawn from this function is a
    horizontal arrow. Returns `(pixel_x, pixel_y)` with image y increasing
    downward.
    """
    lateral = (float(c0), float(c1), float(c2))[LATERAL_ACTION_INDEX]
    return lateral * lateral_sign * scale, 0.0

CONFIG_OPPOSITE = "opposite"
CONFIG_SAME_LEFT = "same_side_left"
CONFIG_SAME_RIGHT = "same_side_right"
CONFIGURATIONS = (CONFIG_OPPOSITE, CONFIG_SAME_LEFT, CONFIG_SAME_RIGHT)

# Queries used to locate the gripper. The workspace start position defines which
# side of the scene a target sits on, so the same-side configurations depend on
# it. Several phrasings are tried because open-vocabulary detection is sensitive
# to wording.
GRIPPER_QUERIES = ("robot gripper", "robot arm", "robotic arm")
# Raised from a permissive value: at a very low threshold the detector almost
# always returns something, so `gripper_source` reported `detected` even when the
# box was noise and the declared fallback effectively never fired. The same-side
# configurations are defined relative to this position, so a spurious box
# silently mislabels the arrangement, which is worse than a recorded fallback.
GRIPPER_SCORE_THRESH = 0.15
# The arm enters these frames from above, so a box confined to the lower part of
# the image is not the arm however confident the detector is.
GRIPPER_MAX_CENTER_Y_FRACTION = 0.75

GRIPPER_SOURCE_DETECTED = "detected"
GRIPPER_SOURCE_FALLBACK = "image_center"

# Manifests written before y_gripper was recorded have only the column x. The
# arm enters these frames from above, so the unrecorded row is taken in the
# upper portion of the frame rather than at mid-height.
GRIPPER_Y_UNRECORDED_FRACTION = 0.22

# How the duplicate was cut out. A bounding box carries the table and any
# neighbouring object that falls inside the rectangle, so the copy reads as a
# pasted rectangle; the instance mask carries the object's own outline. Which was
# used is recorded per scene rather than assumed, because a box cutout is a
# weaker stimulus and the validation sample should be readable against it.
CUTOUT_MASK = "mask"
CUTOUT_BOX = "box"

# Compositing defaults.
DEFAULT_PADDING = 4        # pixels of context kept around the detected box
DEFAULT_FEATHER = 6        # width of the alpha ramp at the patch border
# Minimum distance between instance centres, in patch widths. Above one, so the
# two instances never overlap and "the left one" versus "the right one" is
# visually unambiguous.
DEFAULT_MIN_GAP = 1.2
DEFAULT_MARGIN = 8         # keep pasted content this far inside the frame

# Where the trial's instruction comes from.
#
# `inherited` takes the episode's own instruction, which keeps the wording as it
# appears in the source data but restricts the pool to episodes that already
# contain a lateral term in a referent-selection frame. That pool is very small.
#
# `synthesised` writes the instruction over the episode's manipulated object, so
# any cached frame with a nameable object qualifies. The pool is then the whole
# cache rather than a thin slice of it, and the wording is held constant across
# every trial, which removes instruction form as a nuisance variable: under
# inherited wording a difference between two scenes could always be attributed to
# their differing phrasing rather than to their geometry. The cost is external
# validity, since the instructions are no longer verbatim from the source data.
# The unmodified Bridge scenes supply the first-action object-tracking gate
# instead, which is what they are for, and `instruction_source` records which
# mode produced each scene.
MODE_INHERITED = "inherited"
MODE_SYNTHESISED = "synthesised"
INSTRUCTION_MODES = (MODE_INHERITED, MODE_SYNTHESISED)

# The synthesised wording. Deliberately the plainest form that names the object
# and a side: an unusual construction would confound a null result with a failure
# to parse the instruction at all.
SYNTHESISED_TEMPLATE = "pick up the {noun} on the left"

# Nouns that may not be written into a synthesised instruction. These name
# fixtures and surfaces rather than movable objects, so "pick up the microwave on
# the left" is an unsatisfiable request whatever the arrangement, and a failure to
# comply would say nothing about spatial grounding. Excluding them is the same
# requirement that motivated the constructed set in the first place: an
# instruction the scene cannot support confounds the measurement. The inherited
# mode needs no such list, because a demonstrated episode instruction is
# satisfiable by construction.
IMMOVABLE_NOUNS = frozenset({
    "sink", "stove", "oven", "microwave", "fridge", "refrigerator", "freezer",
    "dishwasher", "drawer", "cabinet", "cupboard", "counter", "countertop",
    "table", "tabletop", "shelf", "wall", "floor", "door", "burner", "faucet",
    "tap", "hob", "worktop", "surface", "top", "side", "area", "space",
})

MANIFEST_NAME = "constructed_manifest.csv"
FRAMES_DIR = "frames"

CONSTRUCTED_FIELDS = [
    "construct_id", "base_scene_id", "noun", "spatial_term", "configuration",
    "instr_a", "instr_b", "instruction_source",
    "image_path", "image_width", "image_height",
    "x_source", "x_pasted", "y_source",
    "y_base_source", "y_base_pasted", "y_shift_px", "surface_source",
    "x_gripper", "y_gripper", "gripper_source", "gripper_score",
    "x_target_a", "x_target_b", "expected_sign_image",
    "offset_a_px", "offset_b_px", "separation_px",
    "target_sign_a_image", "target_sign_b_image",
    "source_box", "paste_box", "source_score",
    "cutout_mode", "padding", "feather", "seed",
    # Hand labels copied back from the blind geometry-agreement sample by
    # `apply_validation_labels`. Empty for scenes outside the sample, so an
    # empty value means unlabelled rather than agreeing. The recorded geometry
    # columns above are never rewritten from these.
    "human_configuration", "human_two_instances",
    "human_gripper_ok", "human_paste_plausible",
]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class Placement:
    """Where a duplicate instance goes, and what the arrangement implies.

    Attributes:
        configuration: one of CONFIGURATIONS.
        x_source: centre x of the instance already in the scene.
        x_pasted: centre x of the duplicate.
        x_gripper: centre x of the start position.
        x_target_a: centre x of the instance the original instruction names.
        x_target_b: centre x of the instance the swapped instruction names.
        expected_sign_image: sign of `x_target_a - x_target_b`, the direction the
            paired difference should take in image coordinates.
    """

    configuration: str
    x_source: float
    x_pasted: float
    x_gripper: float
    x_target_a: float
    x_target_b: float
    expected_sign_image: int

    @property
    def separation(self) -> float:
        return abs(self.x_pasted - self.x_source)

    @property
    def offset_a(self) -> float:
        """Signed distance from the start position to the first target."""
        return self.x_target_a - self.x_gripper

    @property
    def offset_b(self) -> float:
        return self.x_target_b - self.x_gripper

    @property
    def target_sign_a(self) -> int:
        """Side of the start position the first instruction's target sits on.

        The absolute direction each instruction should move, as distinct from
        the relative order of the two targets. On a same-side arrangement both
        signs are equal, which is the condition that separates scene grounding
        from a fixed word-to-direction mapping.
        """
        return int(np.sign(self.offset_a))

    @property
    def target_sign_b(self) -> int:
        return int(np.sign(self.offset_b))


def resolve_targets(term: str, x_source: float, x_pasted: float):
    """Which instance each instruction names, from the term and the two positions.

    The original instruction carries `term`; the swapped instruction carries its
    antonym. A leftward term names the instance with the smaller image x.
    """
    x_left, x_right = sorted((float(x_source), float(x_pasted)))
    leftward = term.lower().strip() in ("left", "leftmost")
    return (x_left, x_right) if leftward else (x_right, x_left)


def plan_placement(
    configuration: str,
    term: str,
    x_source: float,
    x_gripper: float,
    patch_width: float,
    image_width: int,
    *,
    min_gap: float = DEFAULT_MIN_GAP,
    margin: int = DEFAULT_MARGIN,
) -> Placement | None:
    """Choose where the duplicate goes, or None if the configuration is impossible.

    The source instance cannot be moved, so a same-side configuration is only
    available on the side the source already occupies. Several candidate
    positions are tried in preference order and the first that satisfies every
    constraint is taken; if none does, the configuration is reported as
    unachievable in this frame. Returning None rather than clamping keeps every
    constructed scene one where the intended arrangement was actually achieved,
    so the recorded geometry can be trusted without further inspection.

    Constraints: the duplicate stays inside the frame, is at least `min_gap`
    patch widths from the source so the two never overlap, clears the start
    position by half a patch, and lands on the side the configuration requires.
    """
    half = patch_width / 2.0
    low, high = margin + half, image_width - margin - half
    if low >= high:
        return None
    gap = max(min_gap * patch_width, 1.0)
    source_left = x_source < x_gripper

    if configuration == CONFIG_OPPOSITE:
        # Reflecting about the start position keeps the two instances
        # symmetrically placed; mirroring about the frame centre is the fallback
        # when the reflection would fall outside the image.
        candidates = [2.0 * x_gripper - x_source, float(image_width) - x_source]
        wanted_left = not source_left
    elif configuration in (CONFIG_SAME_LEFT, CONFIG_SAME_RIGHT):
        if source_left != (configuration == CONFIG_SAME_LEFT):
            return None
        # Away from the start position first, since that gives the clearer
        # separation; between the source and the start position otherwise.
        outward = x_source - gap if source_left else x_source + gap
        inward = x_source + gap if source_left else x_source - gap
        candidates = [outward, inward]
        wanted_left = source_left
    else:
        raise ValueError(f"unknown configuration {configuration!r}")

    target = None
    for candidate in candidates:
        if not (low <= candidate <= high):
            continue
        if abs(candidate - x_source) < gap:
            continue
        if abs(candidate - x_gripper) < half:
            continue
        if (candidate < x_gripper) != wanted_left:
            continue
        target = candidate
        break
    if target is None:
        return None

    x_a, x_b = resolve_targets(term, x_source, target)
    return Placement(
        configuration=configuration,
        x_source=float(x_source),
        x_pasted=float(target),
        x_gripper=float(x_gripper),
        x_target_a=float(x_a),
        x_target_b=float(x_b),
        expected_sign_image=int(np.sign(x_a - x_b)),
    )


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Support surface geometry
# --------------------------------------------------------------------------- #
# The camera views the table obliquely, so the surface is not a horizontal band
# in the image: its image row varies with the column. Holding the duplicate's
# image y at the source's therefore walks it off the table, which is visible as
# an object floating above the surface or standing in front of its near edge.
#
# The rule used instead is constant relative depth. The surface occupies a run of
# rows in each column, and the object's base is placed at the same fractional
# position within that run as the source's base occupied in its own column. This
# needs no camera calibration, follows a surface of any shape, and has a useful
# side effect: a point at constant relative depth is at roughly constant distance
# from the camera, so the apparent size of the object is unchanged and the patch
# does not need rescaling.

SURFACE_SOURCE_DETECTED = "detected"
SURFACE_SOURCE_NONE = "none"


SURFACE_PROBE_DROP = 6      # pixels below the object's base to prompt from


def locate_surface(image, box, *, probe_drop: int = SURFACE_PROBE_DROP):
    """Mask of the surface the boxed object rests on, or None.

    The prompt is a point just below the centre of the object's base. That point
    is on the supporting surface whenever the object is resting on it, which is
    the case for every object in these frames, so no separate detection of the
    table is needed and no query has to name it. A named query would be worse:
    "table" returns a box spanning most of the frame, which localises nothing.

    The drop below the base is small so the probe stays on the same surface
    rather than reaching past a near edge onto the floor behind or in front.
    """
    from detect_duplicates import segment_surface

    x0, _, x1, y1 = (float(v) for v in box)
    x_probe = 0.5 * (x0 + x1)
    height = image.height if isinstance(image, Image.Image) else image.shape[0]
    y_probe = min(y1 + probe_drop, height - 1)
    return segment_surface(image, (x_probe, y_probe))


def surface_span(mask, x: int):
    """The run of rows the surface occupies in column `x`, or None.

    Returns (y_top, y_bottom) inclusive. The longest contiguous run is taken
    rather than the full extent of the column, because a segmentation may include
    specks elsewhere in the column and the extent would then span the gap between
    them and report rows that are not surface at all.
    """
    array = np.asarray(mask, dtype=bool)
    if not 0 <= int(x) < array.shape[1]:
        return None
    column = array[:, int(x)]
    rows = np.flatnonzero(column)
    if rows.size == 0:
        return None

    # Split the true rows into contiguous runs and keep the longest.
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [rows.size - 1]))
    lengths = ends - starts
    longest = int(np.argmax(lengths))
    return int(rows[starts[longest]]), int(rows[ends[longest]])


def plan_vertical(mask, x_source: float, y_base: float, x_target: float):
    """Row the duplicate's base must occupy at `x_target` to stay on the surface.

    Returns the target base row, or None when the surface is absent from either
    column or the result would not land on it. Returning None is what keeps a
    misplaced duplicate out of the set: the configuration is then skipped rather
    than composited off the table.
    """
    source = surface_span(mask, round(x_source))
    target = surface_span(mask, round(x_target))
    if source is None or target is None:
        return None

    top_source, bottom_source = source
    depth = bottom_source - top_source
    # A degenerate source run carries no depth information, so the fraction is
    # taken as the midpoint rather than dividing by zero.
    fraction = 0.5 if depth <= 0 else (float(y_base) - top_source) / depth
    fraction = min(max(fraction, 0.0), 1.0)

    top_target, bottom_target = target
    y_target = top_target + fraction * (bottom_target - top_target)
    y_target = int(round(y_target))
    if not (top_target <= y_target <= bottom_target):
        return None
    return y_target


def _rectangle_alpha(size, feather: int) -> Image.Image:
    """Alpha that fades to zero at the patch border.

    The fallback used when no instance mask is available. It blends the patch
    background into the surrounding table, which works only because the patch
    background is itself table from the same frame, and it is visibly a
    rectangle wherever that assumption fails. `CUTOUT_BOX` records its use.
    """
    width, height = size
    ramp_x = np.ones(width)
    ramp_y = np.ones(height)
    span = max(int(feather), 0)
    if span > 0:
        edge = np.linspace(0.0, 1.0, span + 2)[1:-1]
        span_x = min(span, width // 2)
        span_y = min(span, height // 2)
        if span_x > 0:
            ramp_x[:span_x] = edge[:span_x]
            ramp_x[-span_x:] = edge[:span_x][::-1]
        if span_y > 0:
            ramp_y[:span_y] = edge[:span_y]
            ramp_y[-span_y:] = edge[:span_y][::-1]
    alpha = np.outer(ramp_y, ramp_x)
    return Image.fromarray((alpha * 255).astype(np.uint8), mode="L")


def _mask_alpha(mask, feather: int) -> Image.Image:
    """Alpha from an instance mask, softened at the outline.

    The mask is the object's own silhouette, so the paste carries the object and
    not the table around it. A blur of the boundary avoids the hard, aliased
    edge that reads as a cutout, while the interior stays fully opaque: blurring
    alone would make a small object semi-transparent, so the core is restored
    after the blur.
    """
    alpha = np.asarray(mask, dtype=bool)
    out = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    span = max(int(feather), 0)
    if span > 0:
        blurred = out.filter(ImageFilter.GaussianBlur(radius=span / 2.0))
        merged = np.maximum(np.asarray(blurred, dtype=np.uint8),
                            (alpha * 255).astype(np.uint8))
        out = Image.fromarray(merged, mode="L")
    return out


def cut_patch(image: Image.Image, box, padding: int = DEFAULT_PADDING, mask=None):
    """Crop the instance with a little surrounding context.

    Returns (patch, crop_box, patch_alpha). `patch_alpha` is the object's
    silhouette within the crop when `mask` is given, and None otherwise, in
    which case the caller falls back to a rectangular blend.

    The context margin is kept in both cases: with a mask it gives the softened
    outline somewhere to fall off, and without one it is all that blends the
    rectangle into the table.
    """
    x0, y0, x1, y1 = box
    crop = (
        max(int(x0) - padding, 0),
        max(int(y0) - padding, 0),
        min(int(x1) + padding, image.width),
        min(int(y1) + padding, image.height),
    )
    patch = image.crop(crop)
    if mask is None:
        return patch, crop, None
    window = np.asarray(mask, dtype=bool)[crop[1]:crop[3], crop[0]:crop[2]]
    return patch, crop, window


def paste_duplicate(
    image: Image.Image,
    box,
    x_target: float,
    *,
    padding: int = DEFAULT_PADDING,
    feather: int = DEFAULT_FEATHER,
    mirror: bool = True,
    mask=None,
    dy: float = 0.0,
):
    """Composite a copy of the instance at `x_target`, shifted down by `dy`.

    The copy is horizontally mirrored by default, so the duplicate reads as a
    second object of the same kind rather than as a repeated cutout, and its
    shading remains consistent with a scene lit from one side.

    `dy` moves the copy vertically, which an oblique camera requires: the table's
    image row varies with the column, so a copy pasted at the source's own row
    would leave the surface. `plan_vertical` computes the shift; zero reproduces
    the fronto-parallel case.

    When `mask` is given the copy is cut along the object's outline, so only the
    object is transferred. Without it the whole box is transferred, including
    whatever background and neighbouring objects fall inside the rectangle.

    Returns (composited image, paste_box, cutout_mode).
    """
    patch, crop, window = cut_patch(image, box, padding=padding, mask=mask)
    if mirror:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if window is not None:
            window = window[:, ::-1]

    if window is not None and window.any():
        alpha = _mask_alpha(window, feather)
        mode = CUTOUT_MASK
    else:
        alpha = _rectangle_alpha(patch.size, feather)
        mode = CUTOUT_BOX

    width, height = patch.size
    left = int(round(x_target - width / 2.0))
    top = crop[1] + int(round(dy))
    left = max(0, min(left, image.width - width))
    top = max(0, min(top, image.height - height))

    out = image.copy()
    out.paste(patch, (left, top), alpha)
    return out, (left, top, left + width, top + height), mode


# --------------------------------------------------------------------------- #
# Gripper localisation
# --------------------------------------------------------------------------- #
def locate_gripper(image, *, queries=GRIPPER_QUERIES,
                   score_thresh: float = GRIPPER_SCORE_THRESH,
                   max_center_y_fraction: float = GRIPPER_MAX_CENTER_Y_FRACTION):
    """Estimate the start position's image centre.

    Returns (x, y, source, score). Falls back to the image centre when no query
    detects the arm, since a wrong-but-declared fallback is safer than a
    confident guess: the `gripper_source` column records which was used, and the
    validation sample reports how often detection agreed with a human.

    Candidates whose centre sits below `max_center_y_fraction` of the frame are
    discarded. The arm enters from above, so a low box is a tabletop object the
    query happened to score, and accepting it would place the start position on
    the wrong side of the scene while still reporting a confident detection.
    """
    from detect_duplicates import detect_instances

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    cutoff = float(image.height) * max_center_y_fraction
    best = None
    for query in queries:
        for found in detect_instances(image, query, score_thresh=score_thresh):
            if found.center_y > cutoff:
                continue
            if best is None or found.score > best.score:
                best = found
            break
    if best is None:
        return (float(image.width) / 2.0, float(image.height) / 2.0,
                GRIPPER_SOURCE_FALLBACK, 0.0)
    return (float(best.center_x), float(best.center_y),
            GRIPPER_SOURCE_DETECTED, float(best.score))


def gripper_image_xy(scene, image_height: float) -> tuple[float, float]:
    """Start-position pixel coordinates used to overlay a predicted action.

    y is the recorded detection centre when present. Manifests written before
    that column existed have only x, and the arm enters from the top of these
    frames, so the unrecorded row is taken in the upper portion of the frame
    rather than at mid-height.
    """
    x = float(scene["x_gripper"])
    raw = scene.get("y_gripper", "")
    if raw not in ("", None):
        try:
            y = float(raw)
        except (TypeError, ValueError):
            y = float("nan")
        if y == y:
            return x, y
    return x, float(image_height) * GRIPPER_Y_UNRECORDED_FRACTION


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #
@dataclass
class ConstructedScene:
    """One constructed trial and the geometry that defines its expectation."""

    construct_id: str
    base_scene_id: str
    noun: str
    spatial_term: str
    configuration: str
    instr_a: str
    instr_b: str
    instruction_source: str
    image_path: str
    image_width: int
    image_height: int
    x_source: float
    x_pasted: float
    y_source: float
    y_base_source: float
    y_base_pasted: float
    y_shift_px: float
    surface_source: str
    x_gripper: float
    y_gripper: float
    gripper_source: str
    gripper_score: float
    x_target_a: float
    x_target_b: float
    expected_sign_image: int
    offset_a_px: float
    offset_b_px: float
    separation_px: float
    target_sign_a_image: int
    target_sign_b_image: int
    source_box: str
    paste_box: str
    source_score: float
    cutout_mode: str
    padding: int
    feather: int
    seed: int


def build_scene(
    base_image: Image.Image,
    base_scene_id: str,
    instruction: str,
    noun: str,
    source_box,
    source_score: float,
    configuration: str,
    *,
    x_gripper: float | None = None,
    y_gripper: float | None = None,
    gripper_source: str = GRIPPER_SOURCE_FALLBACK,
    gripper_score: float = 0.0,
    padding: int = DEFAULT_PADDING,
    feather: int = DEFAULT_FEATHER,
    min_gap: float = DEFAULT_MIN_GAP,
    margin: int = DEFAULT_MARGIN,
    seed: int = 0,
    mask=None,
    surface=None,
    instruction_source: str = MODE_INHERITED,
):
    """Construct one two-instance trial from a single-instance base frame.

    Returns (image, ConstructedScene) or None when the configuration cannot be
    realised in this frame, for example because the duplicate would leave the
    image, land on the wrong side of the start position, or come to rest off the
    supporting surface.

    `mask` is the source instance's silhouette, used to cut the duplicate along
    the object's outline instead of its bounding box. `surface` is the mask of the
    surface the object rests on, used to place the duplicate at the row the
    perspective requires rather than at the source's own row.
    """
    made = make_pair(instruction)
    if made is None:
        return None
    term, swapped = made
    if not is_lateral_term(term):
        return None

    if x_gripper is None:
        x_gripper = base_image.width / 2.0
    if y_gripper is None:
        y_gripper = base_image.height / 2.0

    x0, y0, x1, y1 = (float(v) for v in source_box)
    x_source = 0.5 * (x0 + x1)
    y_source = 0.5 * (y0 + y1)
    patch_width = (x1 - x0) + 2 * padding

    placement = plan_placement(
        configuration, term, x_source, x_gripper, patch_width, base_image.width,
        min_gap=min_gap, margin=margin,
    )
    if placement is None:
        return None

    # The base of the object, which is what has to rest on the surface.
    y_base = y1
    if surface is None:
        dy, y_pasted, surface_source = 0.0, y_base, SURFACE_SOURCE_NONE
    else:
        planned = plan_vertical(surface, x_source, y_base, placement.x_pasted)
        if planned is None:
            return None
        dy, y_pasted, surface_source = (planned - y_base, float(planned),
                                        SURFACE_SOURCE_DETECTED)

    image, paste_box, cutout_mode = paste_duplicate(
        base_image, source_box, placement.x_pasted, padding=padding,
        feather=feather, mask=mask, dy=dy,
    )
    construct_id = f"c{int(base_scene_id):06d}_{configuration}_{term}"
    scene = ConstructedScene(
        construct_id=construct_id,
        base_scene_id=str(base_scene_id),
        noun=noun,
        spatial_term=term,
        configuration=configuration,
        instr_a=instruction,
        instr_b=swapped,
        instruction_source=instruction_source,
        image_path=os.path.join(FRAMES_DIR, f"{construct_id}.png"),
        image_width=image.width,
        image_height=image.height,
        x_source=placement.x_source,
        x_pasted=placement.x_pasted,
        y_source=y_source,
        y_base_source=y_base,
        y_base_pasted=y_pasted,
        y_shift_px=float(dy),
        surface_source=surface_source,
        x_gripper=placement.x_gripper,
        y_gripper=float(y_gripper),
        gripper_source=gripper_source,
        gripper_score=gripper_score,
        x_target_a=placement.x_target_a,
        x_target_b=placement.x_target_b,
        expected_sign_image=placement.expected_sign_image,
        offset_a_px=placement.offset_a,
        offset_b_px=placement.offset_b,
        separation_px=placement.separation,
        target_sign_a_image=placement.target_sign_a,
        target_sign_b_image=placement.target_sign_b,
        source_box=",".join(f"{v:.1f}" for v in (x0, y0, x1, y1)),
        paste_box=",".join(f"{v:.1f}" for v in paste_box),
        source_score=float(source_score),
        cutout_mode=cutout_mode,
        padding=padding,
        feather=feather,
        seed=seed,
    )
    return image, scene


def _scene_row(scene) -> dict:
    """Normalise a ConstructedScene or row dict to a manifest row."""
    row = asdict(scene) if isinstance(scene, ConstructedScene) else dict(scene)
    return {k: row.get(k, "") for k in CONSTRUCTED_FIELDS}


def write_constructed_manifest(scenes, out_dir: str) -> str:
    """Write the constructed-scene manifest, replacing any existing one."""
    os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONSTRUCTED_FIELDS)
        writer.writeheader()
        for scene in scenes:
            writer.writerow(_scene_row(scene))
    print(f"[write_constructed_manifest] wrote {len(scenes)} scenes -> {path}")
    return path


def append_constructed_manifest(scenes, out_dir: str) -> str:
    """Append scenes to the constructed manifest, creating the header if needed.

    Construction runs for tens of minutes over thousands of detector and
    segmentation calls, so a run that only wrote its manifest at the end would lose
    everything to a disconnected session. Appending as scenes are built also makes
    the file the record of what exists, which is what lets a later run resume from
    it.

    A file whose header does not cover the current schema is rewritten under the
    union before anything is appended, rather than being appended to under assumed
    column names.
    """
    os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)
    path = os.path.join(out_dir, MANIFEST_NAME)
    rows = [_scene_row(scene) for scene in scenes]
    if not rows:
        return path

    fields = list(CONSTRUCTED_FIELDS)
    header = None
    if os.path.isfile(path):
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
    if header:
        fields = list(CONSTRUCTED_FIELDS) + [c for c in header
                                             if c not in CONSTRUCTED_FIELDS]
        if header != fields:
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


def load_constructed_manifest(out_dir: str):
    """Read the constructed-scene manifest back as row dicts."""
    with open(os.path.join(out_dir, MANIFEST_NAME), newline="") as f:
        rows = list(csv.DictReader(f))
    return [{col: row.get(col, "") for col in CONSTRUCTED_FIELDS} for row in rows]


# --------------------------------------------------------------------------- #
# Batch construction
# --------------------------------------------------------------------------- #
# A base frame qualifies when the detector finds exactly one confident instance
# of the target noun: a second instance means the scene is already a two-instance
# trial and needs no construction, while no instance means there is nothing to
# duplicate. `SINGLE_HIGH` is the confidence the one instance must clear, and
# `SINGLE_SECOND_MAX` is the level a second detection must stay below for the
# frame to count as single-instance.
#
# The values are calibrated against the score distribution the diagnostic
# reports on this data rather than carried over from the detector's defaults. The
# scores are per-query sigmoid confidences and are not comparable across nouns:
# a container scores around 0.5 where a small produce item scores around 0.2, so
# a threshold set near the median of the observed distribution rejects whole
# classes of object rather than uncertain detections.
SINGLE_HIGH = 0.15
SINGLE_SECOND_MAX = 0.15

# Only referent-selection instructions can seed a construction; see
# `select_base_scenes` for why placement instructions cannot.
DEFAULT_CATEGORIES = (CATEGORY_REFERENT,)

# Detection floor used by the diagnostic. Low enough that a frame returning
# nothing at all is informative about the query rather than about the threshold.
DIAGNOSTIC_FLOOR = 0.02

# Outcomes of the base-frame gate, in the order they are tested.
GATE_OK = "ok"
GATE_NO_INSTANCE = "no_instance"
GATE_WEAK = "weak_instance"
GATE_MULTI = "multi_instance"

# Reported by the diagnostic rather than by the gate: the query itself was not an
# object name, so no detection was attempted. Kept distinct from `weak_instance`
# because a threshold cannot repair it.
GATE_BAD_NOUN = "unqueryable_noun"


def gate_base_frame(instances, *, single_high: float = SINGLE_HIGH,
                    single_second_max: float = SINGLE_SECOND_MAX) -> str:
    """Whether a frame's detections permit a construction, and why not if they do not.

    A construction needs exactly one instance whose box can be trusted: the box
    fixes what is cut, and a second instance would mean the frame is already a
    two-instance trial with a geometry that was not chosen.

    One instance means one object, not one box. The detector emits several
    overlapping boxes for a single confidently detected object, so the second
    score here would usually belong to the first object were they not suppressed
    first; `detect_instances` does that suppression, and this function assumes it.
    Passing raw detector output would make almost every frame read as
    `multi_instance`.

    Pure so the thresholds can be reasoned about without a detector, and shared
    with the diagnostic so that what is reported and what is built cannot drift
    apart. The absolute score scale is a property of the detector rather than of
    this data, so the thresholds are arguments and are calibrated against the
    observed distribution rather than assumed.
    """
    if not instances:
        return GATE_NO_INSTANCE
    scores = sorted((float(getattr(i, "score", i)) for i in instances), reverse=True)
    if scores[0] < single_high:
        return GATE_WEAK
    if len(scores) > 1 and scores[1] >= single_second_max:
        return GATE_MULTI
    return GATE_OK


def select_base_scenes(manifest_rows, *, categories=DEFAULT_CATEGORIES, limit=None,
                       instruction_mode: str = MODE_INHERITED,
                       template: str = SYNTHESISED_TEMPLATE,
                       splits=(SPLIT_CONSTRUCTION,)):
    """Bridge scenes eligible to seed a construction.

    Every returned row carries the instruction the trial will use, the noun to
    detect, and `instruction_source`, so the caller neither re-derives them nor
    has to know which mode produced them.

    Only frames the harvest assigned to the construction role are eligible.
    Frames reserved for validation are excluded here rather than downstream,
    because a frame composited into an experimental stimulus cannot also stand as
    evidence that the finding holds on unedited data. `splits=None` disables the
    restriction and exists only for inspecting a manifest that predates the roles.

    In `inherited` mode the instruction is the episode's own, which must yield a
    minimal pair on a lateral term since the construction places instances left
    and right. The category filter then defaults to `referent_selection` and is
    not optional in practice: in a placement instruction the spatial term governs
    the landmark rather than the object being moved ("move the pot to the right of
    the spoon"), so duplicating the moved object produces two pots while the
    swapped term still describes a destination beside the spoon, and the term does
    not select between the two instances. Passing `categories=None` disables the
    filter and is supported only for inspecting the wider pool.

    In `synthesised` mode the instruction is written from `template` over the
    episode's manipulated object, so any cached frame qualifies and neither a
    pre-existing spatial term nor a category is required. See
    `SYNTHESISED_TEMPLATE` for why this is the stronger stimulus as well as the
    larger pool, and `IMMOVABLE_NOUNS` for the one restriction it does impose:
    the written instruction has to be satisfiable, since an unliftable target
    would reintroduce the confound the constructed set exists to remove.

    Feasibility and duplicate-target labels are deliberately not consulted in
    either mode: the construction supplies the property those labels were
    filtering for, so requiring them would reimpose the scarcity the constructed
    set exists to escape.
    """
    from detect_duplicates import extract_manipulated_noun, extract_target_noun

    if instruction_mode not in INSTRUCTION_MODES:
        raise ValueError(f"unknown instruction_mode: {instruction_mode!r}")

    allowed_splits = set(splits) if splits is not None else None
    selected = []
    for row in manifest_rows:
        if not row.get("image_path"):
            continue
        if allowed_splits is not None and row.get("split", "") not in allowed_splits:
            continue

        if instruction_mode == MODE_SYNTHESISED:
            noun = extract_manipulated_noun(row.get("instruction", ""))
            if not noun or noun in IMMOVABLE_NOUNS:
                continue
            instruction = template.format(noun=noun)
        else:
            made = make_pair(row.get("instruction", ""))
            if made is None:
                continue
            term, _ = made
            if not is_lateral_term(term):
                continue
            if categories is not None:
                category = row.get("category_manual") or row.get("category", "")
                if category not in categories:
                    continue
            instruction = row["instruction"]
            noun = extract_target_noun(instruction)
            if not noun:
                continue

        selected.append({**row, "instruction": instruction, "target_noun": noun,
                         "instruction_source": instruction_mode})
        if limit is not None and len(selected) >= limit:
            break
    return selected


def sample_candidates(candidates, *, limit=None, seed: int = 0,
                      shuffle: bool = True):
    """A subset of the candidate pool, shuffled before it is truncated.

    Bridge episodes are ordered by scene and task, so a prefix of the manifest
    holds a few domains rather than a sample of the corpus. The build and the
    diagnostic both draw through here, so the frames the thresholds are calibrated
    on are drawn the same way as the frames the build will process. Calibrating on
    a prefix would fit the thresholds to whichever domains happen to come first.
    """
    rows = list(candidates)
    if shuffle:
        rng = np.random.default_rng(seed)
        rows = [rows[int(i)] for i in rng.permutation(len(rows))]
    return rows[:limit] if limit is not None else rows


# Scenes are appended to the manifest in batches. A frame's PNG is named from its
# base index and configuration, so a batch lost to a disconnected session is
# rebuilt identically on the next run rather than duplicated.
CONSTRUCTION_FLUSH_EVERY = 20


def run_construction(
    cache_dir: str,
    out_dir: str,
    *,
    manifest_rows=None,
    categories=DEFAULT_CATEGORIES,
    configurations=CONFIGURATIONS,
    targets: dict | None = None,
    overbuild: float = 1.0,
    limit=None,
    shuffle: bool = True,
    resume: bool = True,
    splits=(SPLIT_CONSTRUCTION,),
    padding: int = DEFAULT_PADDING,
    feather: int = DEFAULT_FEATHER,
    min_gap: float = DEFAULT_MIN_GAP,
    margin: int = DEFAULT_MARGIN,
    detect_gripper: bool = True,
    segment: bool = True,
    require_surface: bool = True,
    score_thresh: float = DEFAULT_SCORE_THRESH,
    single_high: float = SINGLE_HIGH,
    single_second_max: float = SINGLE_SECOND_MAX,
    instruction_mode: str = MODE_INHERITED,
    flush_every: int = CONSTRUCTION_FLUSH_EVERY,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Build the constructed set from cached Bridge frames.

    For each eligible base frame the target noun is detected; frames holding
    anything other than exactly one confident instance are skipped, and the rest
    seed one trial per achievable configuration. Frames and manifest rows are
    written under `out_dir` as they are built, and a summary reports what was built
    and why scenes were rejected, so the yield is inspectable rather than silent.

    `targets` gives a per-configuration count to build, multiplied by `overbuild`
    to leave room for the scenes manual approval will reject, and the run stops once
    every configuration has reached its goal. Without targets the whole candidate
    pool is processed.

    Candidates are shuffled under `seed` before `limit` applies. Bridge episodes are
    ordered by scene and task, so taking a prefix of the manifest samples a few
    domains rather than the corpus, and a truncated run would not be representative
    of the pool it was drawn from.

    `resume` skips base frames that already have scenes in the manifest, so an
    interrupted run continues instead of rebuilding. Construction is deterministic
    given the seed, so a rebuilt frame reproduces the scene it would have produced.

    The detection thresholds are arguments rather than fixed constants because
    their scale belongs to the detector, not to this data. `diagnose_detection`
    reports the distribution they should be set against.
    """
    from data import load_manifest
    from detect_duplicates import detect_instances, segment_instance

    if require_surface and not segment:
        raise ValueError(
            "require_surface needs segment=True: the supporting surface is found "
            "by segmentation, so without it every frame would be rejected")

    rows = manifest_rows if manifest_rows is not None else load_manifest(cache_dir)
    candidates = select_base_scenes(rows, categories=categories,
                                    instruction_mode=instruction_mode,
                                    splits=splits)
    candidates = sample_candidates(candidates, limit=limit, seed=seed,
                                   shuffle=shuffle)

    os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)
    existing = (load_constructed_manifest(out_dir)
                if os.path.isfile(os.path.join(out_dir, MANIFEST_NAME)) else [])
    done_bases = {str(row["base_scene_id"]) for row in existing} if resume else set()
    counts: dict = {}
    for row in existing:
        counts[row["configuration"]] = counts.get(row["configuration"], 0) + 1

    goals = ({config: int(np.ceil(count * overbuild))
              for config, count in targets.items()} if targets else {})

    def goals_met() -> bool:
        return bool(goals) and all(counts.get(config, 0) >= goal
                                   for config, goal in goals.items())

    scenes: list[ConstructedScene] = []
    pending: list[ConstructedScene] = []
    rejects = {GATE_NO_INSTANCE: 0, GATE_WEAK: 0, GATE_MULTI: 0,
               "no_surface": 0, "no_placement": 0, "missing_frame": 0}
    skipped_existing = 0
    processed = 0

    def flush() -> None:
        if pending:
            append_constructed_manifest(pending, out_dir)
            pending.clear()

    if goals_met():
        if verbose:
            print(f"[run_construction] goals already met: {counts}")
        return {"candidates": len(candidates), "constructed": 0,
                "by_configuration": counts, "by_cutout": {}, "rejected": rejects,
                "skipped_existing": 0, "processed": 0, "goals": goals,
                "goals_met": True}

    for row in candidates:
        if str(row["episode_index"]) in done_bases:
            skipped_existing += 1
            continue
        processed += 1
        instruction = row["instruction"]
        noun = row["target_noun"]
        frame_path = os.path.join(cache_dir, row["image_path"])
        if not os.path.isfile(frame_path):
            rejects["missing_frame"] += 1
            continue
        base_image = Image.open(frame_path).convert("RGB")

        found = detect_instances(base_image, noun, score_thresh=score_thresh)
        verdict = gate_base_frame(found, single_high=single_high,
                                  single_second_max=single_second_max)
        if verdict != GATE_OK:
            rejects[verdict] += 1
            continue

        if detect_gripper:
            x_gripper, y_gripper, gripper_source, gripper_score = (
                locate_gripper(base_image))
        else:
            x_gripper, y_gripper, gripper_source, gripper_score = (
                base_image.width / 2.0, base_image.height / 2.0,
                GRIPPER_SOURCE_FALLBACK, 0.0)

        # Segment once per base frame: neither mask depends on where the
        # duplicate is placed, so all configurations share them.
        mask = segment_instance(base_image, found[0].box) if segment else None
        surface = locate_surface(base_image, found[0].box) if segment else None
        if surface is None and require_surface:
            rejects["no_surface"] += 1
            continue

        built_any = False
        for configuration in configurations:
            result = build_scene(
                base_image, row["episode_index"], instruction, noun,
                found[0].box, found[0].score, configuration,
                x_gripper=x_gripper, y_gripper=y_gripper,
                gripper_source=gripper_source,
                gripper_score=gripper_score, padding=padding, feather=feather,
                min_gap=min_gap, margin=margin, seed=seed, mask=mask,
                surface=surface, instruction_source=row["instruction_source"],
            )
            if result is None:
                continue
            image, scene = result
            image.save(os.path.join(out_dir, scene.image_path))
            scenes.append(scene)
            pending.append(scene)
            counts[configuration] = counts.get(configuration, 0) + 1
            built_any = True
        if not built_any:
            rejects["no_placement"] += 1

        # Every configuration a frame can support is built even once its own cell
        # is full, so the opposite and same-side scenes of a frame stay available
        # as a pair. The selection step needs both to make the same-side against
        # opposite contrast a within-frame comparison.
        if len(pending) >= flush_every:
            flush()
            if verbose:
                print(f"[run_construction] {processed} frames processed, "
                      f"built {counts}")
        if goals_met():
            break

    flush()

    by_cutout: dict = {}
    for scene in scenes:
        by_cutout[scene.cutout_mode] = by_cutout.get(scene.cutout_mode, 0) + 1
    summary = {
        "candidates": len(candidates),
        "processed": processed,
        "skipped_existing": skipped_existing,
        "constructed": len(scenes),
        "by_configuration": counts,
        "by_cutout": by_cutout,
        "rejected": rejects,
        "goals": goals,
        "goals_met": goals_met(),
    }
    if verbose:
        print(f"[run_construction] {processed} of {summary['candidates']} candidate "
              f"frames processed ({skipped_existing} already built) -> "
              f"{summary['constructed']} new scenes")
        print(f"[run_construction] set now holds: {counts}")
        print(f"[run_construction] cutouts: {by_cutout}")
        print(f"[run_construction] rejected: {rejects}")
        if goals:
            print(f"[run_construction] goals {goals}, met {summary['goals_met']}")
        if by_cutout.get(CUTOUT_BOX):
            print(f"[run_construction] {by_cutout[CUTOUT_BOX]} scenes fell back to "
                  "a box cutout; those duplicates carry background as well as the "
                  "object and read as pasted rectangles")
    return summary


# --------------------------------------------------------------------------- #
# Detection diagnostic
# --------------------------------------------------------------------------- #
def diagnose_detection(
    cache_dir: str,
    candidates,
    *,
    limit: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
    score_thresh: float = DIAGNOSTIC_FLOOR,
    working_thresh: float = DEFAULT_SCORE_THRESH,
    single_high: float = SINGLE_HIGH,
    single_second_max: float = SINGLE_SECOND_MAX,
    top_k: int = 3,
):
    """Per-frame record of what the detector returned and how the gate ruled.

    A zero yield has three causes that the rejection counts alone cannot
    separate: the query may not name an object at all, the query may name one the
    frame does not contain, or the detector may be finding the object and scoring
    it below a threshold that was set without reference to this data. The first
    two are extraction faults and the third is a calibration fault, and they call
    for opposite responses.

    The first is settled without a detector: a query that is not an object name is
    reported as `unqueryable_noun` and no detection is run, so it can neither be
    mistaken for a weak detection nor contribute a meaningless score to the
    quantiles that set the threshold.

    Detection is otherwise run at `DIAGNOSTIC_FLOOR`, well below the working
    threshold, and the scores are returned unfiltered, so a frame whose best score
    sits just under `single_high` is visible as evidence that the threshold is too
    strict. The verdict, however, is taken over the instances that clear
    `working_thresh`, the threshold the build itself uses, so the reported
    distribution of verdicts is the one construction will produce rather than one
    produced at a floor no build runs at.

    Returns one dict per frame, with the top boxes retained so the notebook can
    draw them and confirm the detection is on the object rather than merely
    confident.
    """
    from detect_duplicates import detect_instances, is_object_noun

    rows = sample_candidates(candidates, limit=limit, seed=seed, shuffle=shuffle)
    out = []
    for row in rows:
        frame_path = os.path.join(cache_dir, row["image_path"])
        noun = row.get("target_noun", "")
        record = {
            "base_scene_id": str(row.get("episode_index", "")),
            "instruction": row.get("instruction", ""),
            "noun": noun,
            "image_path": row["image_path"],
            "scores": [],
            "boxes": [],
            "verdict": "missing_frame",
        }
        if not noun or not is_object_noun(noun):
            record["verdict"] = GATE_BAD_NOUN
        elif os.path.isfile(frame_path):
            image = Image.open(frame_path).convert("RGB")
            found = detect_instances(image, noun, score_thresh=score_thresh)
            record["scores"] = [round(float(i.score), 4) for i in found[:top_k]]
            record["boxes"] = [i.box for i in found[:top_k]]
            record["verdict"] = gate_base_frame(
                [i for i in found if i.score >= working_thresh],
                single_high=single_high, single_second_max=single_second_max)
        out.append(record)
    return out


def summarise_diagnosis(records, *, single_high: float = SINGLE_HIGH) -> dict:
    """Aggregate `diagnose_detection` into the numbers that pick a threshold.

    `recoverable_by_lowering_single_high` counts frames the gate rejected as weak
    but where the detector did return a box: these are the frames a lower
    `single_high` would admit, and their score quantiles say where to put it.

    Frames whose query was not an object name contribute no scores, so they
    neither inflate that count nor shift the quantiles. Counting them there would
    overstate what lowering the threshold buys, since the only thing a lower
    threshold admits for them is a box on an arbitrary region.
    """
    verdicts: dict = {}
    best = []
    recoverable = 0
    for record in records:
        verdicts[record["verdict"]] = verdicts.get(record["verdict"], 0) + 1
        if record["scores"]:
            best.append(record["scores"][0])
            if record["verdict"] == GATE_WEAK:
                recoverable += 1
    summary = {
        "frames": len(records),
        "by_verdict": verdicts,
        "with_any_detection": len(best),
        "recoverable_by_lowering_single_high": recoverable,
        "best_score_quantiles": {},
    }
    if best:
        quantiles = np.quantile(best, [0.1, 0.25, 0.5, 0.75, 0.9])
        summary["best_score_quantiles"] = {
            label: round(float(value), 4) for label, value
            in zip(("p10", "p25", "p50", "p75", "p90"), quantiles)
        }
        summary["best_score_max"] = round(float(np.max(best)), 4)
    return summary


# --------------------------------------------------------------------------- #
# Validation sample
# --------------------------------------------------------------------------- #
# The measurement reads direction from the recorded geometry, so that geometry
# is the study's central assumption and it rests on an open-vocabulary detector
# that was never evaluated on this data. A hand-labelled sample gives it a
# reportable agreement number.
#
# The sample is drawn at random and never chosen, so the inclusion rule stays
# mechanical: a sample picked by eye would be selected on the same appearance
# that the labelling then judges, and the agreement would measure nothing.
#
# Which scenes it is drawn from is resolved by `agreement_pool` rather than by the
# caller, so no stage can decide for itself what the pool is.

VALIDATION_NAME = "validation_sample.csv"

# Where an agreement sample was drawn from, recorded with the sample so its
# provenance is stated rather than inferred from when it happened to be drawn.
POOL_FROZEN = "frozen"
POOL_APPROVED = "approved"

VALIDATION_FIELDS = [
    "construct_id", "configuration", "instr_a",
    # Which pool the scene was drawn from, one of `POOL_FROZEN` or
    # `POOL_APPROVED`. Recorded per row because an agreement number is only
    # interpretable against the set it was measured on, and a sample file that
    # does not say which one leaves that to be inferred from when it was drawn.
    "pool",
    # Recorded by the pipeline, shown to the annotator only after labelling.
    # `auto_cutout_mode` is carried so that a judgement of the paste can be read
    # against how the duplicate was cut: a box cutout is expected to look worse,
    # and pooling the two would hide that.
    "auto_configuration", "auto_gripper_source", "auto_cutout_mode",
    # Hand labels.
    "human_configuration",   # opposite | same_side_left | same_side_right | unclear
    "human_two_instances",   # yes | no: are two instances of the noun visible
    "human_gripper_ok",      # yes | no: does the marked start position sit on the arm
    "human_paste_plausible",  # yes | no: does the duplicate read as a real object
    "notes",
]

VALIDATION_UNLABELLED = ""

# The hand labels, which are the only columns a redraw carries over from an
# existing sample file.
VALIDATION_LABEL_FIELDS = ("human_configuration", "human_two_instances",
                           "human_gripper_ok", "human_paste_plausible", "notes")


def draw_validation_sample(scenes, n: int = 50, seed: int = 0, retain=(),
                           pool: str = ""):
    """Draw a random sample of constructed scenes for hand labelling.

    Stratified by configuration so the same-side arrangements, which are fewer
    and carry the decisive comparison, are represented rather than left to the
    luck of a simple random draw.

    `retain` names scenes already labelled, which are kept when they are in the
    pool and topped up around rather than redrawn. Labelling is the scarcest
    resource in the pipeline, and a pool that grows as screening continues would
    otherwise discard the work every time the draw was repeated. Retaining is not
    a departure from mechanical selection: the retained scenes were themselves
    drawn at random, and restricting a random sample to a subset defined without
    reference to the draw leaves a random sample of that subset. Ids absent from
    the pool are not retained, so labels collected on scenes the pool excludes are
    dropped rather than carried into the number.
    """
    rng = np.random.default_rng(seed)
    by_config: dict = {}
    for scene in scenes:
        row = asdict(scene) if isinstance(scene, ConstructedScene) else dict(scene)
        by_config.setdefault(row["configuration"], []).append(row)
    if not by_config:
        return []

    keep = set(retain)
    per_config = max(1, n // len(by_config))
    drawn = []
    for configuration in sorted(by_config):
        group = by_config[configuration]
        held = [row for row in group if row["construct_id"] in keep]
        drawn.extend(held)
        undrawn = [row for row in group if row["construct_id"] not in keep]
        take = min(max(per_config - len(held), 0), len(undrawn))
        for index in rng.choice(len(undrawn), size=take, replace=False):
            drawn.append(undrawn[int(index)])

    # Top up from whatever is left, so the requested size is met when the pool
    # allows it even if some configurations are scarce.
    chosen = {row["construct_id"] for row in drawn}
    remaining = [row for group in by_config.values() for row in group
                 if row["construct_id"] not in chosen]
    if len(drawn) < n and remaining:
        extra = rng.choice(len(remaining), size=min(n - len(drawn), len(remaining)),
                           replace=False)
        drawn.extend(remaining[int(i)] for i in extra)

    return [{
        "construct_id": row["construct_id"],
        "configuration": row["configuration"],
        "instr_a": row["instr_a"],
        "pool": pool,
        "auto_configuration": row["configuration"],
        "auto_gripper_source": row.get("gripper_source", ""),
        "auto_cutout_mode": row.get("cutout_mode", ""),
        "human_configuration": VALIDATION_UNLABELLED,
        "human_two_instances": VALIDATION_UNLABELLED,
        "human_gripper_ok": VALIDATION_UNLABELLED,
        "human_paste_plausible": VALIDATION_UNLABELLED,
        "notes": "",
    } for row in drawn]


def write_validation_sample(rows, out_dir: str) -> str:
    """Write or update the validation sample, preserving existing labels.

    Only the hand labels are carried over from an existing file. The pipeline's
    own columns are taken from the rows passed in, so a redraw that changes the
    pool a scene belongs to is recorded as such instead of keeping the stale
    value from the previous draw.
    """
    path = os.path.join(out_dir, VALIDATION_NAME)
    existing = {}
    if os.path.isfile(path):
        for row in load_validation_sample(out_dir):
            existing[row["construct_id"]] = row
    merged = []
    for row in rows:
        prior = existing.get(row["construct_id"])
        if prior is not None:
            row = {**row, **{k: v for k, v in prior.items()
                             if v and k in VALIDATION_LABEL_FIELDS}}
        merged.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALIDATION_FIELDS)
        writer.writeheader()
        for row in merged:
            writer.writerow({k: row.get(k, "") for k in VALIDATION_FIELDS})
    print(f"[write_validation_sample] {len(merged)} scenes -> {path}")
    return path


def load_validation_sample(out_dir: str):
    """Read the validation sample back as row dicts."""
    with open(os.path.join(out_dir, VALIDATION_NAME), newline="") as f:
        return [{col: row.get(col, "") for col in VALIDATION_FIELDS}
                for row in csv.DictReader(f)]


def validation_agreement(rows) -> dict:
    """Agreement between the recorded geometry and the hand labels.

    Reports, over the labelled rows: how often the annotator read the same
    arrangement the pipeline recorded, how often two instances were visible at
    all, how often the localised start position sat on the arm, and how often
    the pasted instance read as a real object.

    Configuration agreement is the number the geometry claim rests on. Gripper
    agreement bears mainly on the same-side configurations, which are defined
    relative to the start position, so a low value there weakens the decisive
    comparison specifically rather than the whole constructed set.

    The plausibility of the paste is also reported per cutout mode. An outline
    cutout and a bounding-box cutout are different stimuli, the second visibly
    worse, so a pooled rate would average away the distinction that decides
    whether any box fallbacks are usable.
    """
    labelled = [r for r in rows if r.get("human_configuration")]
    out = {
        "drawn": len(rows),
        "labelled": len(labelled),
        "configuration_agreement": float("nan"),
        "two_instances_rate": float("nan"),
        "gripper_agreement": float("nan"),
        "paste_plausible_rate": float("nan"),
        "by_configuration": {},
        "by_cutout": {},
    }
    if not labelled:
        return out

    def rate(key, value="yes"):
        answered = [r for r in labelled if r.get(key)]
        if not answered:
            return float("nan")
        return sum(1 for r in answered if r[key] == value) / len(answered)

    matches = [r["human_configuration"] == r["auto_configuration"] for r in labelled]
    out["configuration_agreement"] = sum(matches) / len(matches)
    out["two_instances_rate"] = rate("human_two_instances")
    out["gripper_agreement"] = rate("human_gripper_ok")
    out["paste_plausible_rate"] = rate("human_paste_plausible")

    for configuration in sorted({r["auto_configuration"] for r in labelled}):
        group = [r for r in labelled if r["auto_configuration"] == configuration]
        out["by_configuration"][configuration] = {
            "n": len(group),
            "agreement": sum(1 for r in group
                             if r["human_configuration"] == configuration) / len(group),
        }

    for mode in sorted({r.get("auto_cutout_mode", "") for r in labelled} - {""}):
        group = [r for r in labelled if r.get("auto_cutout_mode") == mode]
        answered = [r for r in group if r.get("human_paste_plausible")]
        out["by_cutout"][mode] = {
            "n": len(group),
            "paste_plausible_rate": (
                sum(1 for r in answered if r["human_paste_plausible"] == "yes")
                / len(answered) if answered else float("nan")),
        }
    return out


# The label columns carried from the validation sample onto the constructed
# manifest. `notes` stays in the sample file, which remains the record of the
# labelling session.
MANIFEST_LABEL_FIELDS = ("human_configuration", "human_two_instances",
                         "human_gripper_ok", "human_paste_plausible")


def apply_validation_labels(out_dir: str) -> dict:
    """Copy the hand labels from the validation sample onto the manifest rows.

    Each labelled scene's manifest row gains the annotator's reading in the
    `human_*` columns, so a frame carries both what the pipeline recorded and
    what the image was judged to show, and a disagreement is visible on the
    row itself rather than only in the sample file. The recorded geometry is
    never rewritten from the labels: `configuration` and the coordinate
    columns are derived from the placement, and overwriting them would erase
    the disagreement the blind sample exists to expose while leaving the
    coordinates asserting the original arrangement. Scenes outside the sample
    keep empty labels, so an empty value means unlabelled rather than
    agreeing.

    The manifest is rewritten atomically under the union of its existing
    header and the label columns, preserving any columns beyond the current
    schema. Returns the row count, how many rows carry a label, and how many
    of those disagree with the recorded arrangement.
    """
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    labels = {r["construct_id"]: r for r in load_validation_sample(out_dir)
              if r.get("human_configuration")}

    fields = header + [c for c in MANIFEST_LABEL_FIELDS if c not in header]
    labelled = 0
    disagreements = 0
    for row in rows:
        label = labels.get(row.get("construct_id", ""))
        if label is None:
            continue
        for col in MANIFEST_LABEL_FIELDS:
            row[col] = label.get(col, "")
        labelled += 1
        if label["human_configuration"] != row.get("configuration", ""):
            disagreements += 1

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    os.replace(tmp, path)
    return {"rows": len(rows), "labelled": labelled,
            "configuration_disagreements": disagreements}


# The annotator's answer when the arrangement cannot be read from the image.
# A scene carrying it has no trustworthy arrangement under either account, so
# the probe excludes it rather than testing it under a label nobody stands
# behind.
ARRANGEMENT_UNCLEAR = "unclear"


def arrangement_target_signs(configuration: str, expected_sign_image: int):
    """The side each instruction's target sits on, implied by an arrangement.

    The two same-side arrangements fix both signs outright. The opposite
    arrangement puts the two targets either side of the start position, so the
    signs follow from which named target is further right, which is
    `expected_sign_image`: pure instance geometry, exact by construction and
    independent of where the gripper was detected.
    """
    if configuration == CONFIG_SAME_LEFT:
        return -1, -1
    if configuration == CONFIG_SAME_RIGHT:
        return 1, 1
    if configuration == CONFIG_OPPOSITE:
        sign = int(expected_sign_image)
        return sign, -sign
    raise ValueError(f"unknown configuration {configuration!r}")


def resolve_scene_arrangement(scene) -> dict | None:
    """The arrangement a scene is probed under, once the hand labels are in.

    Returns `None` for a scene whose hand label is `unclear`, which the probe
    set drops. A scene without a hand label, or whose label agrees with the
    recorded arrangement, keeps its recorded configuration and target signs
    unchanged. A scene whose label disagrees is probed under the human
    reading: the arrangement is defined relative to the real arm, and the
    annotator saw the arm where the detector may not have, so the recorded
    configuration and the gripper-relative signs are the columns the
    disagreement discredits. The target signs are re-derived from the human
    arrangement and the instances' relative order (`expected_sign_image`),
    which is exact by construction and does not rest on the gripper
    detection. The recorded geometry columns in the manifest are not touched;
    the correction is applied where the scene is read.
    """
    row = asdict(scene) if isinstance(scene, ConstructedScene) else dict(scene)
    label = str(row.get("human_configuration", "") or "")
    recorded = str(row["configuration"])
    if label == ARRANGEMENT_UNCLEAR:
        return None
    if not label or label == recorded:
        return {
            "configuration": recorded,
            "target_sign_a_image": int(row["target_sign_a_image"]),
            "target_sign_b_image": int(row["target_sign_b_image"]),
            "relabelled": False,
        }
    sign_a, sign_b = arrangement_target_signs(
        label, int(row["expected_sign_image"]))
    return {
        "configuration": label,
        "target_sign_a_image": sign_a,
        "target_sign_b_image": sign_b,
        "relabelled": True,
    }


# --------------------------------------------------------------------------- #
# Approval review
# --------------------------------------------------------------------------- #
# Every scene entering the experiments is looked at once by a human, and a scene
# that fails is excluded with its reason recorded. This is a stimulus screen, not
# curation of the evaluation set: it is applied before any prediction is made, it
# is blind to the model, and its criteria are fixed in advance, whereas the
# hand-picking this project rejected chose scenes on the property being measured
# and left the inclusion rule unstated. What it removes are composites that failed
# as stimuli, where the duplicate does not read as a second object of the named
# kind, so the instruction has no second referent and the trial measures nothing.
#
# The consequence for interpretation is recorded rather than hidden: the estimate
# is of grounding given a well-posed and perceptible scene, and rejected scenes stay
# in the manifest so a sample of them can be probed as a check on what the screen
# bought.
#
# Labels live in their own file keyed by construct_id. Putting them in the
# constructed manifest would lose them the next time construction rewrote it, which
# is the same reason feasibility labels live in the Bridge manifest rather than in a
# notebook cell.
REVIEW_NAME = "constructed_review.csv"

DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"
DECISION_UNLABELLED = ""

# Why a scene was rejected. Recorded per scene so the screen reports a funnel
# rather than a single pass rate: the reasons point at different faults, and which
# one dominates decides what to fix. A reason that concentrates in one
# configuration would also mean the arms of the comparison were screened
# differently, which the summary surfaces.
#
# Every reason is decidable from the image and the instruction alone. Whether the
# arrangement matches the one recorded, and whether the start position is really the
# arm, are deliberately not screening criteria: judging them requires being shown
# the pipeline's own answer, and a scene kept or dropped on agreement with that
# answer would make the geometry unfalsifiable. Those two are measured instead, on a
# random subsample labelled blind (`draw_validation_sample`), where disagreement is
# reported rather than removed.
REJECT_NO_SECOND_INSTANCE = "no_second_instance"   # the copy does not read as a
                                                  # second object of the named kind
REJECT_PASTE_IMPLAUSIBLE = "paste_implausible"     # it reads as a patch: a seam, a
                                                  # floating object, a wrong scale
REJECT_OBJECT_WRONG = "object_wrong"               # neither object is the noun the
                                                  # instruction names
REJECT_OTHER = "other"
REJECT_REASONS = (REJECT_NO_SECOND_INSTANCE, REJECT_PASTE_IMPLAUSIBLE,
                  REJECT_OBJECT_WRONG, REJECT_OTHER)

REVIEW_FIELDS = [
    "construct_id", "base_scene_id", "configuration", "instr_a",
    # Carried for the funnel, never shown to the annotator: a scene judged
    # alongside the pipeline's own answer is not an independent judgement.
    "auto_gripper_source", "auto_cutout_mode",
    "decision", "reject_reason", "notes",
]


def build_review_rows(scenes, existing=None):
    """One review row per constructed scene, preserving any decision already made.

    Covers the whole set rather than a sample, because the purpose is to decide
    inclusion for each scene. The random sample in `draw_validation_sample` keeps
    its separate purpose of measuring how well the recorded geometry matches human
    reading, and it is drawn from the pool `agreement_pool` resolves rather than
    from every scene reviewed here. Approval can be conditioned on without
    circularity because the screen never sees the recorded arrangement, so a
    decision carries no information about the quantity the agreement number
    measures.
    """
    prior = {row["construct_id"]: row for row in (existing or [])}
    rows = []
    for scene in scenes:
        row = asdict(scene) if isinstance(scene, ConstructedScene) else dict(scene)
        current = {
            "construct_id": row["construct_id"],
            "base_scene_id": row.get("base_scene_id", ""),
            "configuration": row.get("configuration", ""),
            "instr_a": row.get("instr_a", ""),
            "auto_gripper_source": row.get("gripper_source", ""),
            "auto_cutout_mode": row.get("cutout_mode", ""),
            "decision": DECISION_UNLABELLED,
            "reject_reason": "",
            "notes": "",
        }
        seen = prior.get(row["construct_id"])
        if seen is not None:
            current.update({k: v for k, v in seen.items()
                            if k in ("decision", "reject_reason", "notes") and v})
        rows.append(current)
    return rows


def write_review(rows, out_dir: str) -> str:
    """Write the review file, replacing it in place."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, REVIEW_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REVIEW_FIELDS})
    os.replace(tmp, path)
    return path


def load_review(out_dir: str):
    """Read the review file back as row dicts, or an empty list when absent."""
    path = os.path.join(out_dir, REVIEW_NAME)
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return [{col: row.get(col, "") for col in REVIEW_FIELDS}
                for row in csv.DictReader(f)]


def sync_review(scenes, out_dir: str) -> list:
    """Refresh the review file against the current set and return its rows.

    Construction adds scenes across several sessions, so the review file has to
    grow with the set while keeping the decisions already recorded.
    """
    rows = build_review_rows(scenes, load_review(out_dir))
    write_review(rows, out_dir)
    return rows


def approve(rows, construct_id: str, *, decision: str, reason: str = "",
            notes: str = "") -> list:
    """Record one decision in a review row list, in place, and return it.

    Rejecting without a reason raises: the reason is what makes the screen
    reportable as a funnel, and a blank one would leave a scene excluded for no
    stated cause.
    """
    if decision not in (DECISION_APPROVED, DECISION_REJECTED):
        raise ValueError(f"decision must be {DECISION_APPROVED!r} or "
                         f"{DECISION_REJECTED!r}, got {decision!r}")
    if decision == DECISION_REJECTED and reason not in REJECT_REASONS:
        raise ValueError(f"a rejection needs a reason from {list(REJECT_REASONS)}, "
                         f"got {reason!r}")
    for row in rows:
        if row["construct_id"] == construct_id:
            row["decision"] = decision
            row["reject_reason"] = reason if decision == DECISION_REJECTED else ""
            row["notes"] = notes
            return rows
    raise KeyError(f"no review row for {construct_id!r}")


def approved_ids(rows) -> set:
    """Construct ids approved for the experiments."""
    return {row["construct_id"] for row in rows
            if row.get("decision") == DECISION_APPROVED}


def approval_summary(rows, targets=None) -> dict:
    """The screen's funnel: what was reviewed, approved, and rejected why.

    Reported per configuration as well as overall. An approval rate that differs
    between the arrangements would mean the arms of the decisive comparison were
    screened to different standards, which is a threat to the comparison rather
    than a detail, so it is not averaged away.
    """
    total = len(rows)
    approved = [r for r in rows if r.get("decision") == DECISION_APPROVED]
    rejected = [r for r in rows if r.get("decision") == DECISION_REJECTED]
    reviewed = len(approved) + len(rejected)

    by_configuration: dict = {}
    for configuration in sorted({r.get("configuration", "") for r in rows}):
        group = [r for r in rows if r.get("configuration") == configuration]
        group_approved = [r for r in group
                          if r.get("decision") == DECISION_APPROVED]
        group_reviewed = [r for r in group
                          if r.get("decision") in (DECISION_APPROVED,
                                                   DECISION_REJECTED)]
        entry = {
            "scenes": len(group),
            "reviewed": len(group_reviewed),
            "approved": len(group_approved),
            "unlabelled": len(group) - len(group_reviewed),
            "approval_rate": (len(group_approved) / len(group_reviewed)
                              if group_reviewed else float("nan")),
        }
        if targets:
            goal = targets.get(configuration)
            if goal is not None:
                entry["target"] = goal
                entry["still_needed"] = max(goal - len(group_approved), 0)
        by_configuration[configuration] = entry

    by_reason: dict = {}
    for row in rejected:
        reason = row.get("reject_reason") or REJECT_OTHER
        by_reason[reason] = by_reason.get(reason, 0) + 1

    by_cutout: dict = {}
    for mode in sorted({r.get("auto_cutout_mode", "") for r in rows} - {""}):
        group = [r for r in rows if r.get("auto_cutout_mode") == mode]
        group_reviewed = [r for r in group
                          if r.get("decision") in (DECISION_APPROVED,
                                                   DECISION_REJECTED)]
        by_cutout[mode] = {
            "scenes": len(group),
            "approval_rate": (
                sum(1 for r in group_reviewed
                    if r["decision"] == DECISION_APPROVED) / len(group_reviewed)
                if group_reviewed else float("nan")),
        }

    return {
        "scenes": total,
        "reviewed": reviewed,
        "approved": len(approved),
        "rejected": len(rejected),
        "unlabelled": total - reviewed,
        "approval_rate": len(approved) / reviewed if reviewed else float("nan"),
        "by_configuration": by_configuration,
        "by_reject_reason": by_reason,
        "by_cutout": by_cutout,
    }


# --------------------------------------------------------------------------- #
# The frozen experimental set
# --------------------------------------------------------------------------- #
# How many approved scenes each arrangement contributes. The same-side cells carry
# the decisive comparison and are split evenly between the two sides, so a lateral
# asymmetry in the model, or a systematic error in locating the start position,
# appears as a difference between the two same-side cells instead of biasing their
# pooled result. The opposite cell is the sum of the two, which is also what a base
# frame yields: a frame supports one opposite arrangement and exactly one same-side
# arrangement, on whichever side its object already sits.
DEFAULT_TARGETS = {
    CONFIG_OPPOSITE: 300,
    CONFIG_SAME_LEFT: 150,
    CONFIG_SAME_RIGHT: 150,
}

EVALUATION_NAME = "evaluation_set.csv"
EVALUATION_FIELDS = ["construct_id", "base_scene_id", "configuration", "paired",
                     "seed"]


def pair_eligible_ids(scenes) -> set:
    """Construct ids whose base frame carries both arrangements of the contrast.

    A scene enters the frozen set only as half of a within-frame pair, so one
    whose frame never yielded the other arrangement cannot contribute however it
    is judged. Most of a built set is in that position, because a frame supports
    the opposite arrangement and a same-side one only when both placements fit, so
    screening the eligible scenes first is the difference between a feasible manual
    pass and one that spends most of its effort on scenes that cannot be used.
    """
    rows = [asdict(s) if isinstance(s, ConstructedScene) else dict(s)
            for s in scenes]
    built: dict = {}
    for row in rows:
        built.setdefault(str(row["base_scene_id"]), set()).add(row["configuration"])
    same_side = {CONFIG_SAME_LEFT, CONFIG_SAME_RIGHT}
    paired = {base for base, configurations in built.items()
              if CONFIG_OPPOSITE in configurations and configurations & same_side}
    return {row["construct_id"] for row in rows
            if str(row["base_scene_id"]) in paired}


def project_yield(scenes, review_rows, *, targets=DEFAULT_TARGETS,
                  candidates_processed: int | None = None) -> dict:
    """What the pool as built will support, and what filling the targets requires.

    Every rate is measured from the set rather than assumed: the share of base
    frames that yielded both arrangements, and the share of those pairs whose two
    scenes both passed the screen. Base frames are the unit throughout, because a
    same-side cell is filled from frames that also carry the opposite arrangement,
    so a frame rather than a scene is what a target consumes.

    `candidates_processed` is how many base frames construction has attempted,
    which the manifest cannot report: it holds the frames that produced a scene and
    not those rejected before one. Supplying it turns the projection from frames
    that yield scenes into frames that have to be harvested, which is the number a
    harvest is sized by.

    Projections are `None` where the measured rate needed for them is zero or
    undefined, so an unfilled cell cannot be reported as a finite requirement.
    """
    rows = [asdict(s) if isinstance(s, ConstructedScene) else dict(s)
            for s in scenes]
    decisions = {row["construct_id"]: row.get("decision", DECISION_UNLABELLED)
                 for row in review_rows}
    same_side = {CONFIG_SAME_LEFT, CONFIG_SAME_RIGHT}

    by_base: dict = {}
    for row in rows:
        by_base.setdefault(str(row["base_scene_id"]), []).append(row)

    paired_frames = 0
    decided_pairs = 0
    approved_pairs = 0
    approved_by_side: dict = {CONFIG_SAME_LEFT: 0, CONFIG_SAME_RIGHT: 0}
    for members in by_base.values():
        configurations = {row["configuration"] for row in members}
        if CONFIG_OPPOSITE not in configurations or not configurations & same_side:
            continue
        paired_frames += 1
        verdicts = {row["construct_id"]: decisions.get(row["construct_id"],
                                                       DECISION_UNLABELLED)
                    for row in members}
        # A frame still holding an unseen scene is not yet decided either way, so
        # it counts toward neither the approvals nor the rate they are taken over.
        if any(v == DECISION_UNLABELLED for v in verdicts.values()):
            continue
        decided_pairs += 1
        opposite_ok = any(verdicts[row["construct_id"]] == DECISION_APPROVED
                          for row in members
                          if row["configuration"] == CONFIG_OPPOSITE)
        sides = [row["configuration"] for row in members
                 if row["configuration"] in same_side
                 and verdicts[row["construct_id"]] == DECISION_APPROVED]
        if opposite_ok and sides:
            approved_pairs += 1
            approved_by_side[sides[0]] += 1

    frames = len(by_base)
    pair_rate = paired_frames / frames if frames else float("nan")
    joint_rate = approved_pairs / decided_pairs if decided_pairs else float("nan")
    per_candidate = (paired_frames / candidates_processed
                     if candidates_processed else float("nan"))

    needed = (int(targets.get(CONFIG_SAME_LEFT, 0))
              + int(targets.get(CONFIG_SAME_RIGHT, 0)))
    deficit = max(needed - approved_pairs, 0)

    def scaled(rate) -> int | None:
        if deficit == 0:
            return 0
        if rate != rate or rate <= 0:
            return None
        return int(np.ceil(deficit / rate))

    pairs_to_build = scaled(joint_rate)
    return {
        "base_frames": frames,
        "paired_frames": paired_frames,
        "pair_rate": pair_rate,
        "decided_pairs": decided_pairs,
        "approved_pairs": approved_pairs,
        "approved_by_side": approved_by_side,
        "joint_approval_rate": joint_rate,
        "paired_frames_needed": needed,
        "paired_frames_short": deficit,
        "pairs_to_build": pairs_to_build,
        "scenes_to_screen": None if pairs_to_build is None else 2 * pairs_to_build,
        "frames_to_yield_scenes": scaled(joint_rate * pair_rate),
        "candidates_to_process": scaled(joint_rate * per_candidate),
        "candidates_processed": candidates_processed,
    }


def select_evaluation_set(scenes, approved, *, targets=DEFAULT_TARGETS, seed: int = 0):
    """Choose the experimental set from the approved scenes.

    The same-side cells are filled first, because they are the binding constraint:
    a base frame can only supply the same-side arrangement on the side its object
    already occupies, so the scarcer side decides how many frames are needed. The
    opposite cell is then taken from those same frames, which makes the same-side
    against opposite contrast a within-frame comparison holding the background,
    the object, the cutout, and the instruction fixed, and leaves scene identity
    unable to explain a difference between the two.

    Frames are drawn at random under `seed` from those whose scenes were approved,
    so the rule stays mechanical and the set is reproducible. A cell that cannot be
    filled is reported as a shortfall rather than quietly padded from unpaired
    scenes, since replacing a paired scene with an unpaired one changes what the
    comparison controls for.

    Returns the chosen rows, the shortfall per cell, and the counts.
    """
    rng = np.random.default_rng(seed)
    approved = set(approved)
    rows = [asdict(s) if isinstance(s, ConstructedScene) else dict(s)
            for s in scenes]
    usable = [r for r in rows if r["construct_id"] in approved]

    by_base: dict = {}
    for row in usable:
        by_base.setdefault(str(row["base_scene_id"]), {})[row["configuration"]] = row

    same_side_configs = [c for c in (CONFIG_SAME_LEFT, CONFIG_SAME_RIGHT)
                         if c in targets]
    chosen: list[dict] = []
    shortfall: dict = {}
    selected_bases: list[str] = []

    for configuration in same_side_configs:
        # Sorted before the draw so the shuffle, and therefore the set, depends on
        # the seed alone rather than on dictionary order.
        eligible = sorted(base for base, built in by_base.items()
                          if configuration in built and CONFIG_OPPOSITE in built)
        goal = int(targets[configuration])
        take = min(goal, len(eligible))
        picked = [eligible[int(i)] for i in
                  rng.choice(len(eligible), size=take, replace=False)] if take else []
        shortfall[configuration] = goal - take
        for base in picked:
            chosen.append({**by_base[base][configuration], "paired": "yes"})
            selected_bases.append(base)

    opposite_goal = int(targets.get(CONFIG_OPPOSITE, 0))
    paired_opposite = selected_bases[:opposite_goal]
    for base in paired_opposite:
        chosen.append({**by_base[base][CONFIG_OPPOSITE], "paired": "yes"})
    shortfall[CONFIG_OPPOSITE] = opposite_goal - len(paired_opposite)

    out_rows = [{
        "construct_id": row["construct_id"],
        "base_scene_id": row["base_scene_id"],
        "configuration": row["configuration"],
        "paired": row.get("paired", "yes"),
        "seed": seed,
    } for row in chosen]

    counts: dict = {}
    for row in out_rows:
        counts[row["configuration"]] = counts.get(row["configuration"], 0) + 1
    return {
        "rows": out_rows,
        "counts": counts,
        "shortfall": {k: v for k, v in shortfall.items() if v > 0},
        "approved_scenes": len(usable),
        "paired_base_frames": len(set(selected_bases)),
        "complete": not any(v > 0 for v in shortfall.values()),
        "seed": seed,
    }


def write_evaluation_set(rows, out_dir: str) -> str:
    """Freeze the experimental set to disk.

    Written once and read by every later stage. Re-deriving the selection wherever
    it is needed would let the set drift as approval continues, so the probe, the
    analysis, and the figures would not be describing the same scenes.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, EVALUATION_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in EVALUATION_FIELDS})
    os.replace(tmp, path)
    print(f"[write_evaluation_set] froze {len(rows)} scenes -> {path}")
    return path


def load_evaluation_set(out_dir: str):
    """Read the frozen experimental set, or an empty list when it does not exist."""
    path = os.path.join(out_dir, EVALUATION_NAME)
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return [{col: row.get(col, "") for col in EVALUATION_FIELDS}
                for row in csv.DictReader(f)]


def evaluation_scenes(out_dir: str, scenes=None):
    """The constructed manifest rows named by the frozen set, in its order.

    The single entry point every later stage uses, so no stage decides for itself
    what the experimental set is.
    """
    frozen = load_evaluation_set(out_dir)
    if not frozen:
        return []
    rows = scenes if scenes is not None else load_constructed_manifest(out_dir)
    by_id = {row["construct_id"]: row for row in rows}
    paired = {row["construct_id"]: row.get("paired", "") for row in frozen}
    out = []
    for entry in frozen:
        row = by_id.get(entry["construct_id"])
        if row is not None:
            out.append({**row, "paired": paired.get(entry["construct_id"], "")})
    return out


def agreement_pool(out_dir: str, scenes=None, review_rows=None):
    """The scenes a geometry agreement sample may be drawn from, and which pool.

    Resolved here rather than at the call site. An earlier version let the
    notebook fall back to the whole constructed manifest when no frozen set
    existed, which drew the sample from rejected and unscreened scenes and made
    the agreement number describe a set the probe never runs on.

    The frozen set is the pool once it exists, since agreement is a claim about
    the stimuli the experiments use. Before the freeze the pool is the approved
    scenes: a rejected scene is excluded for faults such as an implausible paste
    that also make the arrangement harder to read, so admitting rejections would
    understate agreement on the set that is actually used, and an unscreened
    scene is of unknown quality. Neither substitution is silent, because the pool
    is returned alongside the rows and recorded with the sample.

    Restricting to approved scenes does not compromise the blind labelling. The
    screen is decided without the recorded arrangement in view, so approval
    carries no information about whether the arrangement matches, which is the
    quantity being measured.
    """
    rows = []
    source = scenes if scenes is not None else load_constructed_manifest(out_dir)
    for scene in source:
        rows.append(asdict(scene) if isinstance(scene, ConstructedScene)
                    else dict(scene))

    frozen = evaluation_scenes(out_dir, rows)
    if frozen:
        return frozen, POOL_FROZEN

    approved = approved_ids(review_rows if review_rows is not None
                            else load_review(out_dir))
    return [row for row in rows if row["construct_id"] in approved], POOL_APPROVED


# --------------------------------------------------------------------------- #
# Manipulation check
# --------------------------------------------------------------------------- #
def manipulation_rate(predictions, scenes,
                      value_col: str = f"c{LATERAL_ACTION_INDEX}") -> dict:
    """How often the model acts toward the pasted instance rather than the source.

    A composited object is only useful if the model perceives it. Run with the
    term-stripped instruction, which names the object without locating it, the
    action should sometimes point at the duplicate: a rate at or near zero means
    the paste is being ignored and the constructed scenes cannot support the
    measurement, whichever way the language results come out.

    The expectation is one-sided by design. A model free to choose between two
    instances need not split evenly, so the check is that the duplicate is
    chosen a non-trivial fraction of the time, not that the split is balanced.

    Args:
        predictions: rows carrying `base_scene_id`, `configuration`, and the
            lateral value column, from the neutral condition only.
        scenes: constructed manifest rows, supplying the two positions.
        value_col: column holding the lateral prediction. The default reads
            the identified lateral channel (`LATERAL_ACTION_INDEX`).

    Returns the rate, the counts, and the per-configuration breakdown.
    """
    import pandas as pd

    preds = pd.DataFrame(predictions)
    meta = pd.DataFrame(scenes)
    if preds.empty or meta.empty:
        return {"n": 0, "toward_pasted": 0, "rate": float("nan"), "by_configuration": {}}

    for frame in (preds, meta):
        for col in ("construct_id",):
            if col in frame.columns:
                frame[col] = frame[col].astype(str)
    merged = preds.merge(
        meta[["construct_id", "x_source", "x_pasted", "x_gripper", "configuration"]],
        on="construct_id", how="inner", suffixes=("", "_meta"),
    )
    if merged.empty:
        return {"n": 0, "toward_pasted": 0, "rate": float("nan"), "by_configuration": {}}

    value = merged[value_col].astype(float).to_numpy()
    x_source = merged["x_source"].astype(float).to_numpy()
    x_pasted = merged["x_pasted"].astype(float).to_numpy()
    x_gripper = merged["x_gripper"].astype(float).to_numpy()

    # Direction each instance lies in, expressed in action-sign terms.
    dir_source = np.sign(x_source - x_gripper) * IMAGE_X_TO_LATERAL_SIGN
    dir_pasted = np.sign(x_pasted - x_gripper) * IMAGE_X_TO_LATERAL_SIGN
    acted = np.sign(value)

    # Only informative where the two instances lie in opposite directions;
    # otherwise a single action is consistent with reaching for either.
    separable = (dir_source != dir_pasted) & (acted != 0)
    toward_pasted = separable & (acted == dir_pasted)

    by_config: dict = {}
    for name, group in merged.assign(
        separable=separable, toward=toward_pasted
    ).groupby("configuration"):
        usable = int(group["separable"].sum())
        by_config[str(name)] = {
            "n": usable,
            "toward_pasted": int(group["toward"].sum()),
            "rate": float(group["toward"].sum() / usable) if usable else float("nan"),
        }

    usable_total = int(separable.sum())
    return {
        "n": usable_total,
        "toward_pasted": int(toward_pasted.sum()),
        "rate": float(toward_pasted.sum() / usable_total) if usable_total else float("nan"),
        "by_configuration": by_config,
    }
