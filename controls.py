"""
controls.py: stimulus transforms and the condition factorial for the probe.

The baseline contrast (one image, an instruction and its antonym-swapped
variant) cannot on its own support a claim about spatial grounding. Two
alternative explanations survive it. A model that maps the token `left` to a
leftward action without consulting the image reproduces the expected sign flip
exactly, so a positive result is not decisive. A model whose lateral output is
insensitive to the image altogether produces no difference for reasons that have
nothing to do with language, so a null result is not decisive either.

The conditions defined here separate those explanations by transforming one
factor at a time:

  * Mirroring the image reverses the lateral axis while holding the instruction
    fixed. A model that reads lateral position must reverse its lateral output.
    Paired with the term-stripped instruction this measures object grounding,
    which is the necessary condition the language claim rests on rather than the
    claim itself.
  * Pairing an instruction with a different scene leaves the language intact and
    removes its referent. A difference that survives is lexical.
  * Removing the spatial term gives a within-scene reference, so the term's
    contribution can be measured as a deviation from the model's own behaviour
    on that exact image rather than against the opposite instruction alone.

Transforms are deterministic: the same inputs always produce the same stimulus,
so a probe run is reproducible and restart-safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from data import SWAP, is_lateral_term

# --------------------------------------------------------------------------- #
# Condition factorial
# --------------------------------------------------------------------------- #
# Role of the instruction within a condition. Roles `a` and `b` are the original
# and antonym-swapped instructions, matching the baseline probe. Role `n` is the
# term-stripped instruction, which has no opposite and is therefore logged as a
# single prediction rather than a pair.
ROLE_A = "a"
ROLE_B = "b"
ROLE_NEUTRAL = "n"

# Image transform applied by a condition.
IMAGE_ORIGINAL = "original"
IMAGE_MIRROR = "mirror"
IMAGE_SWAPPED_SCENE = "swapped_scene"
IMAGE_GREY = "grey"
IMAGE_NOISE = "noise"

CONDITION_BASELINE = "baseline"
CONDITION_MIRROR = "mirror"
CONDITION_MIRROR_NEUTRAL = "mirror_neutral"
CONDITION_SWAPPED_SCENE = "swapped_scene"
CONDITION_NEUTRAL = "neutral"
CONDITION_GREY = "grey"
CONDITION_NOISE = "noise"


class Condition:
    """One cell of the factorial: an image transform and the roles it is run with.

    Attributes:
        name: value written to the `condition` column of the prediction log.
        image: image transform applied before inference.
        roles: instruction roles predicted under this condition.
        lateral_only: restrict to scenes whose spatial term contrasts the lateral
            axis. A horizontal flip reverses only that axis, so the mirror
            conditions carry no expectation for depth or vertical terms.
        purpose: what the condition establishes, for documentation and plots.
    """

    def __init__(self, name, image, roles, lateral_only=False, purpose=""):
        self.name = name
        self.image = image
        self.roles = tuple(roles)
        self.lateral_only = lateral_only
        self.purpose = purpose

    def __repr__(self) -> str:
        return f"Condition({self.name!r}, image={self.image!r}, roles={self.roles})"


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in (
        Condition(
            CONDITION_BASELINE, IMAGE_ORIGINAL, (ROLE_A, ROLE_B),
            purpose="Original contrast: same scene, spatial term swapped.",
        ),
        Condition(
            CONDITION_NEUTRAL, IMAGE_ORIGINAL, (ROLE_NEUTRAL,),
            purpose="Within-scene reference with the spatial term removed, "
                    "against which the term's marginal effect is measured.",
        ),
        Condition(
            CONDITION_MIRROR, IMAGE_MIRROR, (ROLE_A,), lateral_only=True,
            purpose="Lateral antisymmetry: with the instruction fixed, a model "
                    "that reads lateral position must reverse its lateral output.",
        ),
        Condition(
            CONDITION_MIRROR_NEUTRAL, IMAGE_MIRROR, (ROLE_NEUTRAL,), lateral_only=True,
            purpose="Object grounding without a spatial term, the nuisance "
                    "baseline the mirror condition is compared against.",
        ),
        Condition(
            CONDITION_SWAPPED_SCENE, IMAGE_SWAPPED_SCENE, (ROLE_A, ROLE_B),
            purpose="Lexical prior: the contrast run against a different real "
                    "scene, where the instruction has no referent.",
        ),
        Condition(
            CONDITION_GREY, IMAGE_GREY, (ROLE_A, ROLE_B),
            purpose="Secondary sanity check with no scene content at all. "
                    "Far out of distribution, so it is not a substitute for the "
                    "swapped-scene control.",
        ),
        Condition(
            CONDITION_NOISE, IMAGE_NOISE, (ROLE_A, ROLE_B),
            purpose="Secondary sanity check with unstructured scene content.",
        ),
    )
}

# Conditions run by default. The grey and noise ablations are excluded: an image
# far outside the training distribution can drive the model to a constant
# action, which would look like an absent lexical prior whether or not one
# exists. The swapped-scene control keeps the input in distribution and is the
# one the analysis relies on.
DEFAULT_CONDITIONS = (
    CONDITION_BASELINE,
    CONDITION_NEUTRAL,
    CONDITION_MIRROR,
    CONDITION_MIRROR_NEUTRAL,
    CONDITION_SWAPPED_SCENE,
)


def conditions_for(spatial_term: str, names=DEFAULT_CONDITIONS) -> list[Condition]:
    """Conditions applicable to a scene, given the term its instructions swap.

    Drops the mirror conditions for terms that do not contrast the lateral axis,
    since a horizontal flip leaves depth and vertical relations unchanged and
    the condition would carry no expectation.
    """
    lateral = is_lateral_term(spatial_term)
    return [CONDITIONS[n] for n in names
            if lateral or not CONDITIONS[n].lateral_only]


# --------------------------------------------------------------------------- #
# Image transforms
# --------------------------------------------------------------------------- #
def mirror_image(image: Image.Image) -> Image.Image:
    """Return the horizontally flipped image.

    The transform is its own inverse, so a mirrored stimulus can be checked
    against the original without tracking orientation separately.

    Flipping reverses the lateral axis of the scene. It also reflects the robot
    arm, which is a view the model was not trained on, so the mirror conditions
    are always interpreted against the term-stripped mirror baseline rather than
    in absolute terms.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def grey_image(image: Image.Image, level: int = 128) -> Image.Image:
    """Return a uniform grey image of the same size."""
    return Image.new("RGB", image.size, (level, level, level))


def noise_image(image: Image.Image, seed: int = 0) -> Image.Image:
    """Return uniform random noise of the same size, deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(image.size[1], image.size[0], 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def swapped_scene_id(scene_id, scene_ids, seed: int = 0):
    """Pick the scene whose image replaces this scene's, under a derangement.

    A derangement is a permutation with no fixed point, so no scene is ever
    paired with its own image and the control cannot silently degrade into the
    baseline. The permutation is a deterministic function of `scene_ids` and
    `seed`, so a resumed probe run reproduces the same assignment.

    Args:
        scene_id: the scene whose replacement is wanted.
        scene_ids: the full ordered pool of scene identifiers.
        seed: permutation seed.

    Raises ValueError when the pool holds fewer than two scenes, where no
    derangement exists.
    """
    return build_scene_swap(scene_ids, seed=seed)[scene_id]


def build_scene_swap(scene_ids, seed: int = 0) -> dict:
    """Map every scene id to a different scene id, deterministically.

    Uses a random rotation of a shuffled ordering. A rotation by a non-zero
    offset has no fixed point by construction, which avoids the rejection loop a
    naive shuffle-and-retry would need and keeps the result reproducible.
    """
    ids = list(scene_ids)
    if len(ids) < 2:
        raise ValueError(
            "a scene swap needs at least two scenes; with fewer, every "
            "assignment would pair a scene with its own image"
        )
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(ids)))
    offset = int(rng.integers(1, len(ids)))  # non-zero, so no element maps to itself
    return {ids[order[i]]: ids[order[(i + offset) % len(ids)]] for i in range(len(ids))}


def apply_image_transform(kind: str, image: Image.Image, *, seed: int = 0) -> Image.Image:
    """Apply a named image transform.

    `swapped_scene` is resolved by the caller, which must load the replacement
    scene's image and pass it here as `image`; the transform itself is then the
    identity. Keeping the resolution outside this function avoids threading a
    manifest and a cache directory through the transform layer.
    """
    if kind in (IMAGE_ORIGINAL, IMAGE_SWAPPED_SCENE):
        return image
    if kind == IMAGE_MIRROR:
        return mirror_image(image)
    if kind == IMAGE_GREY:
        return grey_image(image)
    if kind == IMAGE_NOISE:
        return noise_image(image, seed=seed)
    raise ValueError(f"unknown image transform {kind!r}")


# --------------------------------------------------------------------------- #
# Instruction transforms
# --------------------------------------------------------------------------- #
# Words that may introduce the spatial term and should be removed with it, so
# stripping "on the left" does not leave "pick up the cup on the".
_CARRIER_PREPS = ("on", "to", "in", "at", "from", "toward", "towards", "near", "by")
_CARRIER_DETS = ("the", "a", "an", "its", "your")

# A spatial term is used one of two ways, and each needs different treatment.
# In "pick up the cup on the left" it ends the phrase, so the carrier words
# leading into it must come out with it. In "place it at the right edge" it
# modifies a following noun, so only the term itself comes out and the carrier
# stays: "place it at the edge".
_PHRASE_END = r"(?=\s*$|[.,;!?]|\s+(?:of|and|then)\b)"

# Terms that are themselves relational prepositions: they take a landmark noun
# phrase directly, with no linking "of". Removing the term alone would strand
# the landmark, turning "put the spoon behind the bowl" into "put the spoon the
# bowl", so the landmark must come out with the term.
_RELATIONAL_TERMS = frozenset({
    "behind", "in front of", "closer to", "nearer to", "farther from",
})

# Landmark phrase stranded at the removal seam, for example the "of the pot"
# left behind by taking "to the left" out of "to the left of the pot". Bounded
# to a short noun phrase so it cannot swallow the rest of a compound
# instruction.
_STRANDED_OF = re.compile(r"^\s*of\s+(?:\w+\s+){0,2}\w+", re.IGNORECASE)


def _phrase_final_pattern(term: str) -> re.Pattern:
    """Match a phrase-final spatial term together with its carrier words."""
    preps = "|".join(_CARRIER_PREPS)
    dets = "|".join(_CARRIER_DETS)
    return re.compile(
        rf"(?:\s+(?:{preps}))?(?:\s+(?:{dets}))?\s+{re.escape(term)}\b{_PHRASE_END}",
        re.IGNORECASE,
    )


def _relational_pattern(term: str) -> re.Pattern:
    """Match a relational preposition together with the landmark it governs."""
    dets = "|".join(_CARRIER_DETS)
    return re.compile(
        rf"\s*\b{re.escape(term)}\b\s+(?:(?:{dets})\s+)?(?:\w+\s+){{0,1}}\w+",
        re.IGNORECASE,
    )


def _adjectival_pattern(term: str) -> re.Pattern:
    """Match a spatial term used as a modifier of a following noun."""
    return re.compile(rf"\s*\b{re.escape(term)}\b", re.IGNORECASE)


def _tidy(text: str) -> str:
    """Collapse the whitespace and stray punctuation a removal can leave behind."""
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    text = re.sub(r"[,;]\s*$", "", text).strip()
    return text


# Words that must not end an instruction: their presence means the removal cut
# a phrase in half rather than taking it whole.
_DANGLING = frozenset(_CARRIER_PREPS) | frozenset(_CARRIER_DETS) | {"of", "into", "onto"}


def strip_spatial_term(instruction: str, term: str) -> str | None:
    """Remove a spatial term and its carrier phrase from an instruction.

    "pick up the cup on the left" becomes "pick up the cup", giving a
    within-scene reference that names the object without locating it. Where the
    term modifies a following noun, only the term is removed, so "place it at
    the right edge" becomes "place it at the edge".

    Returns None when a clean removal is not possible, rather than emitting a
    malformed instruction: a truncated prompt would change the model's output
    for reasons unrelated to the spatial term and quietly corrupt the reference.
    Callers should count the refusals and report them, since the neutral
    condition then covers a subset of the scenes.

    This is a baseline, not a length-matched control. Removing words shortens
    the instruction, and length alone can move the prediction; where that
    matters, `substitute_spatial_term` provides a matched alternative.
    """
    if not instruction or not term:
        return None

    key = term.lower().strip()
    match = None
    if key in _RELATIONAL_TERMS:
        match = _relational_pattern(term).search(instruction)
    if match is None:
        match = _phrase_final_pattern(term).search(instruction)
    if match is None:
        # Modifier of a following noun: take the term alone and leave the rest,
        # so "at the right edge" becomes "at the edge".
        match = _adjectival_pattern(term).search(instruction)
    if match is None:
        return None

    head, tail = instruction[: match.start()], instruction[match.end() :]
    # A landmark phrase at the seam belongs to the relation just removed.
    tail = _STRANDED_OF.sub("", tail, count=1)
    result = _tidy(head + tail)

    words = re.findall(r"[\w']+", result.lower())
    if len(words) < 2 or words[-1] in _DANGLING:
        return None
    # A residual swappable term means the instruction located the object twice;
    # the reference would still carry spatial content.
    if any(re.search(rf"\b{re.escape(other)}\b", result, re.IGNORECASE) for other in SWAP):
        return None
    return result


# A substitution is only coherent where the term closes the instruction after a
# determiner, as in "on the left". Anywhere else, swapping in a noun produces
# text the model was never plausibly trained on ("put the spoon table the bowl",
# "in the table drawer"), which would confound the comparison it is meant to
# clean up.
def _substitution_pattern(term: str) -> re.Pattern:
    dets = "|".join(_CARRIER_DETS)
    return re.compile(
        rf"(?<=\s)(?:(?:{dets})\s+)?{re.escape(term)}\b(?=\s*$|[.!?]\s*$)",
        re.IGNORECASE,
    )


def substitute_spatial_term(
    instruction: str, term: str, replacement: str = "table"
) -> str | None:
    """Replace a phrase-final spatial term with a non-spatial noun.

    "the cup on the left" becomes "the cup on the table", holding instruction
    length and syntactic shape roughly constant while removing the spatial
    content. This is the fluency-matched counterpart to `strip_spatial_term`,
    for confirming that a difference against the neutral reference is not an
    artefact of the shorter prompt.

    Returns None unless the term closes the instruction, which is the only
    position where a noun substitution reads naturally. The condition therefore
    covers a subset of scenes and is reported as such.
    """
    if not instruction or not term:
        return None
    match = _substitution_pattern(term).search(instruction)
    if match is None:
        return None
    head, tail = instruction[: match.start()], instruction[match.end() :]
    determiner = "the " if re.match(rf"(?:{'|'.join(_CARRIER_DETS)})\s", match.group(0),
                                    re.IGNORECASE) else ""
    return _tidy(f"{head}{determiner}{replacement}{tail}")


# Default non-spatial replacement. A surface on which tabletop objects plausibly
# rest, so the substituted instruction stays coherent.
DEFAULT_SUBSTITUTE = "table"


# --------------------------------------------------------------------------- #
# Stimulus planning
# --------------------------------------------------------------------------- #
@dataclass
class Stimulus:
    """One prediction to run: which image to load, transformed how, and with
    which instruction.

    Attributes:
        condition: cell of the factorial, written to the prediction log.
        role: `a`, `b`, or `n` (term stripped).
        instruction: the text passed to the model.
        image_transform: transform applied after loading.
        image_scene_id: scene whose cached frame supplies the image. Equal to
            the scene under test except under the swapped-scene control.
    """

    condition: str
    role: str
    instruction: str
    image_transform: str
    image_scene_id: str


def plan_stimuli(
    scene: dict,
    *,
    swap_map: dict | None = None,
    conditions=DEFAULT_CONDITIONS,
) -> list[Stimulus]:
    """Expand one scene into the predictions its applicable conditions require.

    Args:
        scene: needs `scene_id`, `instr_a`, `instr_b`, and `spatial_term`.
        swap_map: scene-to-scene assignment from `build_scene_swap`. Required
            for the swapped-scene control, which is skipped when it is absent.
        conditions: condition names to expand.

    Conditions are skipped rather than approximated when their precondition
    fails: mirror conditions need a lateral term, the neutral conditions need a
    clean strip, and the swapped-scene control needs an assignment. Skipping
    keeps every logged prediction interpretable, at the cost of unequal counts
    per condition, which the analysis reports.
    """
    scene_id = scene["scene_id"]
    term = scene["spatial_term"]
    neutral = strip_spatial_term(scene["instr_a"], term)

    instructions = {ROLE_A: scene["instr_a"], ROLE_B: scene["instr_b"], ROLE_NEUTRAL: neutral}

    stimuli: list[Stimulus] = []
    for condition in conditions_for(term, conditions):
        image_scene_id = scene_id
        if condition.image == IMAGE_SWAPPED_SCENE:
            if not swap_map or scene_id not in swap_map:
                continue
            image_scene_id = swap_map[scene_id]
        for role in condition.roles:
            instruction = instructions.get(role)
            if not instruction:
                continue
            stimuli.append(Stimulus(
                condition=condition.name,
                role=role,
                instruction=instruction,
                image_transform=condition.image,
                image_scene_id=image_scene_id,
            ))
    return stimuli
