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

Two caveats are handled explicitly rather than assumed away. The relationship
between image x and the sign of the lateral action is not established by the
dataset, so the manifest records the expectation in image coordinates and the
analysis applies the convention separately (`IMAGE_X_TO_LATERAL_SIGN`). And a
pasted object is only useful if the model perceives it, which `manipulation_rate`
measures rather than presumes.

The detector is imported lazily through `detect_duplicates`, so the geometry and
compositing helpers stay importable without a model download.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter

from data import CATEGORY_REFERENT, is_lateral_term, make_pair
# Only the threshold constant is taken at import time, so the detector's heavy
# dependencies stay behind the lazy imports inside the functions that call it.
from detect_duplicates import DEFAULT_SCORE_THRESH

# Sign relating image x to the lateral action component: +1 if a target further
# right in the image implies a larger dx. Identity until the mirror control
# establishes it, following the convention already used for `BRIDGE_TO_ISAAC`.
# The constructed manifest stores the expectation in image coordinates only, so
# flipping this constant never requires regenerating scenes.
IMAGE_X_TO_LATERAL_SIGN = 1

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
# The unmodified Bridge scenes carry that burden instead, which is what they are
# for, and `instruction_source` records which mode produced each scene.
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
    "x_gripper", "gripper_source", "gripper_score",
    "x_target_a", "x_target_b", "expected_sign_image",
    "offset_a_px", "offset_b_px", "separation_px",
    "target_sign_a_image", "target_sign_b_image",
    "source_box", "paste_box", "source_score",
    "cutout_mode", "padding", "feather", "seed",
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
    """Estimate the start position's image x.

    Returns (x, source, score). Falls back to the image centre when no query
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
        return float(image.width) / 2.0, GRIPPER_SOURCE_FALLBACK, 0.0
    return float(best.center_x), GRIPPER_SOURCE_DETECTED, float(best.score)


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


def write_constructed_manifest(scenes, out_dir: str) -> str:
    """Write the constructed-scene manifest, creating the directory if needed."""
    os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONSTRUCTED_FIELDS)
        writer.writeheader()
        for scene in scenes:
            row = asdict(scene) if isinstance(scene, ConstructedScene) else dict(scene)
            writer.writerow({k: row.get(k, "") for k in CONSTRUCTED_FIELDS})
    print(f"[write_constructed_manifest] wrote {len(scenes)} scenes -> {path}")
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
SINGLE_HIGH = 0.25
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


def gate_base_frame(instances, *, single_high: float = SINGLE_HIGH,
                    single_second_max: float = SINGLE_SECOND_MAX) -> str:
    """Whether a frame's detections permit a construction, and why not if they do not.

    A construction needs exactly one instance whose box can be trusted: the box
    fixes what is cut, and a second instance would mean the frame is already a
    two-instance trial with a geometry that was not chosen.

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
                       template: str = SYNTHESISED_TEMPLATE):
    """Bridge scenes eligible to seed a construction.

    Every returned row carries the instruction the trial will use, the noun to
    detect, and `instruction_source`, so the caller neither re-derives them nor
    has to know which mode produced them.

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

    selected = []
    for row in manifest_rows:
        if not row.get("image_path"):
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


def run_construction(
    cache_dir: str,
    out_dir: str,
    *,
    manifest_rows=None,
    categories=DEFAULT_CATEGORIES,
    configurations=CONFIGURATIONS,
    limit=None,
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
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Build the constructed set from cached Bridge frames.

    For each eligible base frame the target noun is detected; frames holding
    anything other than exactly one confident instance are skipped, and the rest
    seed one trial per achievable configuration. Writes the frames and the
    manifest under `out_dir` and returns a summary of what was built and why
    scenes were rejected, so the yield is inspectable rather than silent.

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
    candidates = select_base_scenes(rows, categories=categories, limit=limit,
                                    instruction_mode=instruction_mode)

    scenes: list[ConstructedScene] = []
    rejects = {GATE_NO_INSTANCE: 0, GATE_WEAK: 0, GATE_MULTI: 0,
               "no_surface": 0, "no_placement": 0, "missing_frame": 0}

    os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)

    for row in candidates:
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
            x_gripper, gripper_source, gripper_score = locate_gripper(base_image)
        else:
            x_gripper, gripper_source, gripper_score = (
                base_image.width / 2.0, GRIPPER_SOURCE_FALLBACK, 0.0)

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
                x_gripper=x_gripper, gripper_source=gripper_source,
                gripper_score=gripper_score, padding=padding, feather=feather,
                min_gap=min_gap, margin=margin, seed=seed, mask=mask,
                surface=surface, instruction_source=row["instruction_source"],
            )
            if result is None:
                continue
            image, scene = result
            image.save(os.path.join(out_dir, scene.image_path))
            scenes.append(scene)
            built_any = True
        if not built_any:
            rejects["no_placement"] += 1

    write_constructed_manifest(scenes, out_dir)

    by_config: dict = {}
    by_cutout: dict = {}
    for scene in scenes:
        by_config[scene.configuration] = by_config.get(scene.configuration, 0) + 1
        by_cutout[scene.cutout_mode] = by_cutout.get(scene.cutout_mode, 0) + 1
    summary = {
        "candidates": len(candidates),
        "constructed": len(scenes),
        "by_configuration": by_config,
        "by_cutout": by_cutout,
        "rejected": rejects,
    }
    if verbose:
        print(f"[run_construction] {summary['candidates']} candidate frames -> "
              f"{summary['constructed']} constructed scenes {by_config}")
        print(f"[run_construction] cutouts: {by_cutout}")
        print(f"[run_construction] rejected: {rejects}")
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
    score_thresh: float = DIAGNOSTIC_FLOOR,
    single_high: float = SINGLE_HIGH,
    single_second_max: float = SINGLE_SECOND_MAX,
    top_k: int = 3,
):
    """Per-frame record of what the detector returned and how the gate ruled.

    A zero yield has two very different causes that the rejection counts alone
    cannot separate: the queried noun may not name anything in the frame, or the
    detector may be finding the object and scoring it below a threshold that was
    set without reference to this data. The first is a extraction fault and the
    second is a calibration fault, and they call for opposite responses.

    Detection is therefore run at `DIAGNOSTIC_FLOOR`, well below the working
    threshold, and the scores are returned unfiltered. A frame whose best score
    sits just under `single_high` is evidence the threshold is too strict; a frame
    with nothing at all above the floor is evidence the query is wrong.

    Returns one dict per frame, with the top boxes retained so the notebook can
    draw them and confirm the detection is on the object rather than merely
    confident.
    """
    from detect_duplicates import detect_instances

    rows = list(candidates)[:limit] if limit is not None else list(candidates)
    out = []
    for row in rows:
        frame_path = os.path.join(cache_dir, row["image_path"])
        record = {
            "base_scene_id": str(row.get("episode_index", "")),
            "instruction": row.get("instruction", ""),
            "noun": row.get("target_noun", ""),
            "image_path": row["image_path"],
            "scores": [],
            "boxes": [],
            "verdict": "missing_frame",
        }
        if os.path.isfile(frame_path):
            image = Image.open(frame_path).convert("RGB")
            found = detect_instances(image, record["noun"], score_thresh=score_thresh)
            record["scores"] = [round(float(i.score), 4) for i in found[:top_k]]
            record["boxes"] = [i.box for i in found[:top_k]]
            record["verdict"] = gate_base_frame(
                found, single_high=single_high, single_second_max=single_second_max)
        out.append(record)
    return out


def summarise_diagnosis(records, *, single_high: float = SINGLE_HIGH) -> dict:
    """Aggregate `diagnose_detection` into the numbers that pick a threshold.

    `recoverable` counts frames the gate rejected as weak but where the detector
    did return a box: these are the frames a lower `single_high` would admit, and
    their score quantiles say where to put it.
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

VALIDATION_NAME = "validation_sample.csv"

VALIDATION_FIELDS = [
    "construct_id", "configuration", "instr_a",
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


def draw_validation_sample(scenes, n: int = 50, seed: int = 0):
    """Draw a random sample of constructed scenes for hand labelling.

    Stratified by configuration so the same-side arrangements, which are fewer
    and carry the decisive comparison, are represented rather than left to the
    luck of a simple random draw.
    """
    rng = np.random.default_rng(seed)
    by_config: dict = {}
    for scene in scenes:
        row = asdict(scene) if isinstance(scene, ConstructedScene) else dict(scene)
        by_config.setdefault(row["configuration"], []).append(row)
    if not by_config:
        return []

    per_config = max(1, n // len(by_config))
    drawn = []
    for configuration in sorted(by_config):
        group = by_config[configuration]
        take = min(per_config, len(group))
        for index in rng.choice(len(group), size=take, replace=False):
            drawn.append(group[int(index)])

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
    """Write or update the validation sample, preserving existing labels."""
    path = os.path.join(out_dir, VALIDATION_NAME)
    existing = {}
    if os.path.isfile(path):
        for row in load_validation_sample(out_dir):
            existing[row["construct_id"]] = row
    merged = []
    for row in rows:
        prior = existing.get(row["construct_id"])
        if prior is not None:
            row = {**row, **{k: v for k, v in prior.items() if v}}
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


# --------------------------------------------------------------------------- #
# Manipulation check
# --------------------------------------------------------------------------- #
def manipulation_rate(predictions, scenes, value_col: str = "c0") -> dict:
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
        value_col: column holding the lateral prediction.

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
