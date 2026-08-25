"""
detect_duplicates.py: automated proposal of the duplicate-target label.

The contrastive probe is a clean referent test only when a scene contains two or
more instances of the same object, so that an antonym-swapped instruction (for
example `left` to `right`) names a distinct, real target on both sides. A single
target among different-type distractors makes the swapped instruction
unsatisfiable and confounds the measurement.

This module proposes the `duplicate_target` manifest label without manual
inspection of every scene:

  1. Extract the target noun the spatial term selects (for example `cup`).
  2. Count instances of that noun in the cached frame with an open-vocabulary
     detector (OWLv2).
  3. Map the detection scores to a proposal (`yes` / `no` / `unclear`) with a
     transparent decision function. High-confidence proposals are accepted
     automatically; borderline cases are marked `unclear` and left for manual
     confirmation in the validation set review notebook.

Heavy dependencies (torch, transformers, and optionally spaCy) are imported
lazily inside the functions that need them, so the module stays importable in
environments that hold only the lightweight dependencies (the decision function
and the manifest plumbing remain usable without a model download).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from data import (
    ANTONYM_PAIRS,
    CATEGORY_REFERENT,
    DUPLICATE_SOURCE_AUTO,
    SPATIAL_PHRASES,
    SPATIAL_TOKENS,
    review_queue,
    update_manifest_annotations,
)

# Open-vocabulary detector checkpoint. OWLv2 ships with transformers, so no new
# hard dependency is introduced beyond the pinned transformers install.
OWLV2_CHECKPOINT = "google/owlv2-base-patch16-ensemble"

# Box-prompted segmentation checkpoint, used to cut an instance along its own
# outline rather than along its bounding box. Also ships with transformers.
SAM_CHECKPOINT = "facebook/sam-vit-base"

# Local hub cache for OWLv2 and SAM when the session-wide HF_HOME sits on a
# Google Drive FUSE mount. Drive cannot create the hub's snapshot symlink, so
# from_pretrained copies the blob into the snapshot directory, and that copy of
# a large safetensors file fails with Errno 5. These two checkpoints are small
# enough to live on the runtime disk and be re-downloaded each session. OpenVLA
# stays on Drive, which is why HF_HOME is pointed there. Override with
# HF_LOCAL_CACHE when a different local path is required.
DEFAULT_LOCAL_HF_CACHE = "/tmp/hf_local"

# Detection and decision thresholds. A box is
# only counted when its score clears DEFAULT_SCORE_THRESH. The proposal is then
# decided from the second-highest surviving score: two confident instances imply
# a duplicate target, a weak or absent second instance implies a single target,
# and the band between the two is left for manual confirmation. That reading is
# only valid once overlapping boxes have been suppressed, since the second score
# would otherwise often belong to the first object.
DEFAULT_SCORE_THRESH = 0.10
DEFAULT_HIGH = 0.30
DEFAULT_LOW = 0.15

# Overlap above which two boxes are treated as the same instance. OWLv2 predicts
# one box per image patch and its post-processing applies no suppression, so a
# single object routinely returns several surviving boxes: the object itself plus
# its parts and near-copies from neighbouring patches. Any decision that counts
# instances by counting boxes would read those as a second object.
#
# The overlap is measured against the smaller of the two boxes rather than
# against their union, because a box on a part sits inside the whole-object box
# and scores a low intersection over union while still describing the same
# instance. The threshold is deliberately permissive: merging two genuine
# instances costs one base frame and is caught by the manual screen, whereas
# splitting one instance in two silently discards a usable frame.
DEFAULT_OVERLAP = 0.5

# Words that never name the target object, stripped by the regex fallback.
_DETERMINERS = frozenset({"the", "a", "an", "this", "that", "these", "those",
                          "my", "your", "its", "one", "some", "any"})
_ACTION_VERBS = frozenset({"pick", "up", "grab", "take", "get", "put", "place",
                           "move", "stack", "set", "push", "pull", "lift",
                           "grasp", "hold"})
_PREPOSITIONS = frozenset({"on", "onto", "in", "into", "from", "inside", "of",
                           "to", "at", "with", "and", "then", "please"})

# Multi-word spatial cues and the swap terms, used to strip location words from
# the object phrase in the regex fallback.
_SPATIAL_PHRASE_WORDS = frozenset(
    w for phrase in (SPATIAL_PHRASES + tuple(a for a, _ in ANTONYM_PAIRS)
                     + tuple(b for _, b in ANTONYM_PAIRS))
    for w in phrase.split()
)
_STOP_WORDS = (_DETERMINERS | _ACTION_VERBS | _PREPOSITIONS
               | SPATIAL_TOKENS | _SPATIAL_PHRASE_WORDS)

_WORD_RE = re.compile(r"[a-z]+")

# Words that close the object phrase in `extract_manipulated_noun`. A superset of
# `_PREPOSITIONS`, kept separate so widening it cannot alter the stop-word set
# that `extract_target_noun` depends on. Particles that follow a verb rather than
# introduce a landmark (`up` in "pick up") are absent, since they are already
# treated as part of the verb.
_PHRASE_BOUNDARIES = frozenset({
    "on", "onto", "in", "into", "from", "inside", "outside", "of", "to", "at",
    "with", "and", "then", "off", "over", "under", "beside", "next", "against",
    "around", "through", "toward", "towards", "for", "near", "beneath",
    "behind", "between", "above", "below", "down", "out", "away", "atop",
    "upon", "across", "along", "underneath", "within", "past",
})

# Words that end an instruction by describing the resulting state rather than
# naming an object ("flip the pot upright"). Without these the last word of the
# phrase is an adverb, and in synthesised mode that would be queried as an object
# and written into the instruction.
_TRAILING_MODIFIERS = frozenset({
    "upright", "straight", "flat", "sideways", "upside", "closed", "shut",
    "open", "together", "apart", "over", "back", "aside", "level", "steady",
})

# Words that describe how or where a motion goes rather than what is moved
# ("move the pot diagonally"). They trail the object phrase in the same position
# a head noun would occupy, so without them the direction word becomes the query.
_MANNER_WORDS = frozenset({
    "diagonally", "directly", "forward", "forwards", "backward", "backwards",
    "upward", "upwards", "downward", "downwards", "inward", "inwards",
    "outward", "outwards", "clockwise", "counterclockwise", "further",
    "farther", "closer", "nearer", "again", "halfway", "slightly", "gently",
    "carefully", "slowly", "quickly", "firmly", "fully", "completely",
})

# Words that name a region, a quantity, or an abstraction rather than a physical
# object ("move the pot all the way to the left"). An open-vocabulary query for
# one of them returns whatever the detector finds most photo-like, which is an
# arbitrary box, so the frame is better dropped than queried.
_NON_OBJECT_NOUNS = frozenset({
    "way", "side", "bit", "thing", "place", "position", "spot", "area",
    "direction", "top", "bottom", "middle", "centre", "center", "edge",
    "corner", "front", "end", "half", "part", "all", "it", "them", "they",
    "there", "here",
})

# Nouns ending in the adverbial suffix, exempt from the suffix rule below.
_LY_NOUNS = frozenset({"jelly", "belly", "lily", "holly", "dolly", "doily",
                       "fly"})

# Shortest word accepted as an object name. Below this length the candidates in
# these instructions are pronouns and particles rather than objects.
_MIN_NOUN_LENGTH = 3

# Module-level model cache so the detector is loaded once per process.
_OWL_CACHE: dict = {}


def is_object_noun(word: str) -> bool:
    """Whether a word can stand as the name of an object to detect.

    A wrong noun is not a neutral failure. It is sent to an open-vocabulary
    detector, which always returns its best match for whatever it was asked for,
    so a query for `diagonally` produces a low-scoring box on an arbitrary region
    instead of nothing at all. Such a frame is then indistinguishable from one
    where the detector merely scored a real object weakly, and the two call for
    opposite responses: one is an extraction fault and the other a threshold
    fault. Refusing the word up front keeps that distinction available.

    In synthesised construction the noun is also written into the instruction, so
    an unfiltered adverb would yield a nonsensical trial rather than a skipped
    frame.

    The suffix rule catches adverbs the explicit lists do not enumerate, since
    manner adverbs are open-class in a way that objects in a kitchen are not. The
    few object names sharing the suffix are exempted rather than the rule being
    abandoned.
    """
    word = word.strip().lower()
    if len(word) < _MIN_NOUN_LENGTH:
        return False
    if (word in _STOP_WORDS or word in _TRAILING_MODIFIERS
            or word in _MANNER_WORDS or word in _NON_OBJECT_NOUNS
            or word in _PHRASE_BOUNDARIES):
        return False
    return not (word.endswith("ly") and word not in _LY_NOUNS)


def extract_target_noun(instruction: str, tags=None) -> str | None:
    """Return the head noun the spatial term selects, or None if undetermined.

    A regex heuristic is used first because the instructions are short
    imperatives (for example "grab the red block on the right"), a form that
    general-purpose parsers routinely mis-tag by reading the leading verb as a
    noun or the object noun as a verb. The     heuristic keeps only words that
    `is_object_noun` accepts, which removes action verbs, determiners,
    prepositions, spatial cues, and manner adverbs, and returns the last such word
    before the first spatial cue, which for these instructions is the selected
    object. A spaCy parse is used only as a fallback when the
    heuristic finds nothing, and it takes the root of the first object noun
    chunk. `tags` is accepted for interface symmetry with
    `data.classify_instruction` and is currently unused.
    """
    text = instruction.strip()
    if not text:
        return None

    noun = _extract_with_regex(text)
    if noun:
        return noun
    return _extract_with_spacy(text)


def extract_manipulated_noun(instruction: str) -> str | None:
    """Return the head noun of the object being manipulated, or None.

    Distinct from `extract_target_noun`, which returns the noun a spatial term
    selects. The two differ whenever the instruction contains a landmark: in
    "put the corn in the pot on the left" the spatial term selects the pot, while
    the object being manipulated is the corn. Scene construction in synthesised
    mode needs the manipulated object, because that object is the one the episode
    demonstrates is graspable and present in the frame.

    The head is taken as the last content word of the first noun phrase that can
    stand as an object name, where the phrase ends at the first preposition.
    Ending at the preposition is what separates the object from any landmark, and
    taking the last word rather than the first is what skips modifiers: "grab the
    red block" gives `block`, not `red`.

    Taking the last word is also what exposes a trailing adverb, which occupies
    the head's position without naming anything ("move the pot diagonally"). The
    phrase is therefore scanned from the end and the first word `is_object_noun`
    accepts is returned, so a modifier at the end is passed over in the same way
    one at the start is.

    A wrong noun here is not merely a wasted frame. In synthesised mode it is
    written into the instruction as well as queried, so an adverb read as an object
    would produce a nonsensical trial rather than a rejected one.
    """
    text = instruction.strip().lower()
    if not text:
        return None
    words = _WORD_RE.findall(text)

    phrase = []
    for word in words:
        if (word in _ACTION_VERBS or word in _DETERMINERS
                or word in _TRAILING_MODIFIERS):
            continue
        # A preposition closes the object phrase and opens a landmark phrase.
        if word in _PHRASE_BOUNDARIES:
            if phrase:
                break
            continue
        if word in SPATIAL_TOKENS or word in _SPATIAL_PHRASE_WORDS:
            if phrase:
                break
            continue
        phrase.append(word)

    for word in reversed(phrase):
        if is_object_noun(word):
            return word
    return _extract_with_spacy(text)


def _extract_with_spacy(text: str) -> str | None:
    """Head noun of the first object noun chunk via spaCy, or None if spaCy is
    unavailable or finds no chunk headed by a word that names an object."""
    try:
        import spacy
    except ImportError:
        return None
    nlp = _OWL_CACHE.get("spacy")
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return None
        _OWL_CACHE["spacy"] = nlp

    doc = nlp(text)
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ not in {"NOUN", "PROPN"}:
            continue
        candidate = (root.lemma_.lower() or root.text.lower())
        if is_object_noun(candidate):
            return candidate
    return None


def _extract_with_regex(text: str) -> str | None:
    """Heuristic head-noun extraction without a parser."""
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)

    # Position of the first single-token spatial cue; the object phrase
    # precedes it. Only SPATIAL_TOKENS mark the cue: multi-word phrases such as
    # "to the left" contribute generic words ("the", "to") that would otherwise
    # cut the object phrase short.
    cue_index = len(words)
    for i, w in enumerate(words):
        if w in SPATIAL_TOKENS:
            cue_index = i
            break

    before = [w for w in words[:cue_index] if is_object_noun(w)]
    if before:
        return before[-1]
    remaining = [w for w in words if is_object_noun(w)]
    return remaining[0] if remaining else None


def classify_counts(
    scores,
    *,
    high: float = DEFAULT_HIGH,
    low: float = DEFAULT_LOW,
) -> tuple[str, float]:
    """Map detection scores to a duplicate-target proposal and a confidence.

    The decision is driven by the second-highest detection score, which stands
    for the confidence that a second instance of the object exists. That reading
    requires one score per object, which is what `count_instances` returns once
    overlapping boxes have been suppressed:

      * `yes` when the second-highest score is at least `high` (two confident
        instances of the same object).
      * `no` when the second-highest score is below `low` (no credible second
        instance).
      * `unclear` otherwise, leaving the scene for manual confirmation.

    The returned confidence is the second-highest score. Pure and model-free so
    the banding can be unit-tested in isolation.
    """
    ordered = sorted((float(s) for s in scores), reverse=True)
    second = ordered[1] if len(ordered) >= 2 else 0.0
    if second >= high:
        return "yes", second
    if second < low:
        return "no", second
    return "unclear", second


def _path_is_drive(path: str) -> bool:
    """Whether a cache path sits on Colab's Google Drive FUSE mount."""
    return "/content/drive" in path.replace("\\", "/")


def hf_cache_dir() -> str | None:
    """Hub cache for OWLv2 and SAM, or None to use the session default.

    `HF_LOCAL_CACHE` wins when set. Otherwise a Drive-backed HF_HOME (or the
    older TRANSFORMERS_CACHE / HF_HUB_CACHE variables) is replaced with
    `DEFAULT_LOCAL_HF_CACHE`, so the detector checkpoints never share the 7B
    OpenVLA cache on FUSE. A non-Drive default is left unchanged.
    """
    override = os.environ.get("HF_LOCAL_CACHE", "").strip()
    if override:
        return override
    candidates = (
        os.environ.get("HF_HOME", ""),
        os.environ.get("HF_HUB_CACHE", ""),
        os.environ.get("TRANSFORMERS_CACHE", ""),
    )
    if any(_path_is_drive(path) for path in candidates):
        return DEFAULT_LOCAL_HF_CACHE
    return None


def _pretrained_kwargs() -> dict:
    """`from_pretrained` kwargs that keep OWLv2 and SAM off a Drive cache."""
    cache_dir = hf_cache_dir()
    if cache_dir is None:
        return {}
    os.makedirs(cache_dir, exist_ok=True)
    return {"cache_dir": cache_dir}


def _load_owl():
    """Load and cache the OWLv2 processor and model on the available device."""
    if "owl" in _OWL_CACHE:
        return _OWL_CACHE["owl"]
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    kwargs = _pretrained_kwargs()
    if kwargs:
        print(f"[detect_duplicates] loading {OWLV2_CHECKPOINT} into "
              f"{kwargs['cache_dir']}")
    processor = Owlv2Processor.from_pretrained(OWLV2_CHECKPOINT, **kwargs)
    model = Owlv2ForObjectDetection.from_pretrained(OWLV2_CHECKPOINT, **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _OWL_CACHE["owl"] = (processor, model, device)
    return _OWL_CACHE["owl"]


def _load_sam():
    """Load and cache the segmentation model, or None when it is unavailable.

    Returning None rather than raising keeps scene construction runnable when the
    checkpoint cannot be fetched: the caller falls back to a box cutout and
    records which was used, so a degraded stimulus is visible in the manifest
    instead of stopping the run.
    """
    if "sam" in _OWL_CACHE:
        return _OWL_CACHE["sam"]
    try:
        import torch
        from transformers import SamModel, SamProcessor

        kwargs = _pretrained_kwargs()
        if kwargs:
            print(f"[detect_duplicates] loading {SAM_CHECKPOINT} into "
                  f"{kwargs['cache_dir']}")
        processor = SamProcessor.from_pretrained(SAM_CHECKPOINT, **kwargs)
        model = SamModel.from_pretrained(SAM_CHECKPOINT, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        loaded = (processor, model, device)
    except Exception as exc:  # noqa: BLE001 - reported, not silently swallowed
        print(f"[detect_duplicates] segmentation unavailable ({exc}); "
              "scene construction will fall back to box cutouts")
        loaded = None
    _OWL_CACHE["sam"] = loaded
    return loaded


def padded_target_size(width: int, height: int) -> tuple[int, int]:
    """The (height, width) OWLv2 box coordinates are normalised against.

    `Owlv2ImageProcessor` pads the image to a square of side `max(height,
    width)` with grey on the bottom and right before resizing, so the predicted
    boxes are fractions of that padded square rather than of the original frame.
    Passing the original size to `post_process_object_detection` scales the
    shorter axis by too small a factor: on a 640x480 frame every box is pulled
    upward by a quarter of its distance from the top and loses a quarter of its
    height, which is enough to cut a patch of background instead of the object.

    Padding is applied to the bottom and right only, so the origin is shared and
    no offset correction is needed once both axes use the square's side.
    """
    side = int(max(width, height))
    return side, side


def clip_box(box, width: int, height: int) -> tuple[float, float, float, float]:
    """Clamp a box to the frame, ordering the corners.

    A box may legitimately extend into the padded region, which does not exist
    in the original frame, so predictions are clipped back to real pixels.
    """
    x0, x1 = sorted((float(box[0]), float(box[2])))
    y0, y1 = sorted((float(box[1]), float(box[3])))
    return (
        min(max(x0, 0.0), float(width)),
        min(max(y0, 0.0), float(height)),
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
    )


def box_overlap(box_a, box_b) -> float:
    """Intersection of two boxes as a fraction of the smaller box's area.

    Chosen over intersection over union because the boxes being compared are
    frequently nested rather than merely overlapping: a detection on a pot's
    handle lies wholly inside the detection on the pot, which this measure scores
    as complete overlap while intersection over union would score it as almost
    none. Returns 0.0 when either box is degenerate.
    """
    ax0, ay0, ax1, ay1 = (float(v) for v in box_a)
    bx0, by0, bx1, by1 = (float(v) for v in box_b)
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    if smaller <= 0.0:
        return 0.0
    return (inter_w * inter_h) / smaller


@dataclass
class Instance:
    """One detected instance of the queried noun.

    Attributes:
        score: detection confidence.
        box: (x_min, y_min, x_max, y_max) in pixels.
    """

    score: float
    box: tuple[float, float, float, float]

    @property
    def center_x(self) -> float:
        return 0.5 * (self.box[0] + self.box[2])

    @property
    def center_y(self) -> float:
        return 0.5 * (self.box[1] + self.box[3])

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def suppress_overlapping(instances, *, overlap: float = DEFAULT_OVERLAP):
    """Keep one detection per object, discarding boxes that cover an earlier one.

    Greedy suppression in descending score order: the highest-scoring box is
    kept, and any later box overlapping a kept box by more than `overlap` is
    dropped as a redundant view of the same instance rather than as a second one.

    Without this step every consumer that counts instances counts boxes instead.
    The detector emits a dense set of candidate boxes and applies no suppression
    of its own, so a single confidently detected object yields a descending
    cascade of boxes on itself, which reads exactly like a second, slightly less
    certain instance.

    Pure and model-free, so the merging rule can be tested against boxes chosen
    to be nested, adjacent, or disjoint.
    """
    kept: list = []
    for instance in sorted(instances, key=lambda i: float(i.score), reverse=True):
        if any(box_overlap(instance.box, other.box) > overlap for other in kept):
            continue
        kept.append(instance)
    return kept


def detect_instances(image, noun: str, *, score_thresh: float = DEFAULT_SCORE_THRESH,
                     overlap: float = DEFAULT_OVERLAP):
    """Detect instances of `noun` in `image`, returning scores with boxes.

    `image` may be a PIL image or an HxWx3 array. The query is the target noun
    phrased as "a photo of a <noun>", the standard OWLv2 prompt form. Instances
    are returned in descending score order, one per object: overlapping boxes are
    suppressed here rather than by each caller, so a count of returned instances
    is a count of objects. Passing `overlap=1.0` disables the suppression and
    returns the detector's raw output.

    The boxes are what the duplicate-target proposal discards and the scene
    construction needs: the position of each instance is what defines the
    expected direction for a scene, so it is recorded rather than inferred from
    the instruction.
    """
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    processor, model, device = _load_owl()
    query = f"a photo of a {noun}"
    inputs = processor(text=[[query]], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Boxes are normalised against the padded square, not the frame.
    target_sizes = torch.tensor([padded_target_size(*image.size)], device=device)
    # The combined `Owlv2Processor` gained `post_process_object_detection` only in
    # later transformers releases; on the pinned build it lives on the wrapped
    # image processor, so resolve whichever is present.
    post_process = getattr(processor, "post_process_object_detection", None)
    if post_process is None:
        post_process = processor.image_processor.post_process_object_detection
    results = post_process(
        outputs, target_sizes=target_sizes, threshold=score_thresh
    )[0]
    instances = [
        Instance(score=float(score), box=clip_box(box, *image.size))
        for score, box in zip(results["scores"].tolist(), results["boxes"].tolist())
    ]
    return suppress_overlapping(instances, overlap=overlap)


def _sam_mask(image, *, box=None, point=None):
    """Run the segmentation model on one prompt, returning a boolean HxW mask.

    Returns None when the model is unavailable or the mask comes back empty. An
    empty mask is a failure rather than a valid empty region, so it is reported
    the same way and the caller decides how to degrade.
    """
    import numpy as np
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    loaded = _load_sam()
    if loaded is None:
        return None
    processor, model, device = loaded

    prompt = {}
    if box is not None:
        prompt["input_boxes"] = [[[float(v) for v in clip_box(box, *image.size)]]]
    if point is not None:
        prompt["input_points"] = [[[float(point[0]), float(point[1])]]]

    inputs = processor(image, return_tensors="pt", **prompt).to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    mask = np.asarray(masks[0][0][0], dtype=bool)
    return mask if mask.any() else None


def segment_instance(image, box):
    """Mask of the object inside `box`, or None when segmentation is unavailable.

    Returns a boolean HxW array covering the whole frame, true on the object.

    A bounding box is a poor cutout for compositing: the rectangle carries the
    table, shadows, and any neighbouring object that happens to fall inside it,
    so the pasted copy reads as a pasted rectangle rather than as a second
    object. Prompting with the detected box gives the instance outline, and
    compositing along that outline leaves only the object itself.

    The box prompt is what makes this reliable without supervision: the region
    to segment is already known from detection, so the model is only asked to
    find the boundary within it rather than to decide what is salient.
    """
    return _sam_mask(image, box=box)


def segment_surface(image, point):
    """Mask of the surface at `point`, or None when segmentation is unavailable.

    Used to find the extent of the table an object rests on. The prompt is a
    point rather than a box because the surface has no detected box: an
    open-vocabulary query for a table returns a rectangle spanning most of the
    frame, which constrains nothing, whereas a point known to lie on the surface
    identifies it directly.

    The natural prompt is a point just below the object's own footprint, which is
    on the supporting surface whenever the object is resting on it.
    """
    return _sam_mask(image, point=point)


def count_instances(image, noun: str, *, score_thresh: float = DEFAULT_SCORE_THRESH,
                    overlap: float = DEFAULT_OVERLAP):
    """Return the per-instance detection scores for `noun` in `image`.

    Thin wrapper over `detect_instances` for callers that need only the scores.
    One score per object, in descending order, since the overlapping boxes the
    detector emits for a single object are suppressed upstream.
    """
    return [inst.score for inst in
            detect_instances(image, noun, score_thresh=score_thresh,
                             overlap=overlap)]


def run_auto_pass(
    out_dir: str,
    *,
    categories=(CATEGORY_REFERENT,),
    only_pairable: bool = True,
    score_thresh: float = DEFAULT_SCORE_THRESH,
    high: float = DEFAULT_HIGH,
    low: float = DEFAULT_LOW,
    limit: int | None = None,
) -> dict:
    """Propose `duplicate_target` for candidate scenes and write it back.

    Iterates scenes that are still `unreviewed` for `duplicate_target` in the
    given categories, counts the target noun in each cached frame, classifies
    the result, and writes `duplicate_target`, `duplicate_score`,
    `duplicate_note`, and `duplicate_source='auto'` in a single manifest update.
    Scenes whose target noun cannot be extracted are marked `unclear` and left
    for manual confirmation. Returns a summary of the proposal counts.
    """
    queue = review_queue(
        out_dir,
        status=None,
        duplicate_status="unreviewed",
        categories=list(categories),
        only_pairable=only_pairable,
        limit=limit,
    )

    annotations: dict = {}
    counts = {"yes": 0, "no": 0, "unclear": 0}
    noun_failures = 0

    for item in queue:
        ep = item["episode_index"]
        instruction = item["instruction"]
        rel = item["image_path"]
        noun = extract_target_noun(instruction)
        if not noun or not rel:
            annotations[ep] = {
                "duplicate_target": "unclear",
                "duplicate_score": "",
                "duplicate_note": ("noun extraction failed" if not noun
                                   else "no cached frame"),
                "duplicate_source": DUPLICATE_SOURCE_AUTO,
            }
            counts["unclear"] += 1
            noun_failures += 1
            continue

        image_path = os.path.join(out_dir, rel)
        scores = count_instances(image_path_to_image(image_path),
                                 noun, score_thresh=score_thresh)
        label, confidence = classify_counts(scores, high=high, low=low)
        annotations[ep] = {
            "duplicate_target": label,
            "duplicate_score": round(confidence, 4),
            "duplicate_note": f"auto: noun={noun}, n_instances={len(scores)}, "
                              f"s2={round(confidence, 4)}",
            "duplicate_source": DUPLICATE_SOURCE_AUTO,
        }
        counts[label] += 1

    if annotations:
        update_manifest_annotations(out_dir, annotations)

    summary = {
        "total": len(queue),
        "yes": counts["yes"],
        "no": counts["no"],
        "unclear": counts["unclear"],
        "noun_failures": noun_failures,
        "thresholds": {"score_thresh": score_thresh, "high": high, "low": low},
    }
    print(f"[run_auto_pass] {summary['total']} scenes -> "
          f"yes={summary['yes']} no={summary['no']} unclear={summary['unclear']} "
          f"(noun_failures={noun_failures})")
    return summary


def image_path_to_image(path: str):
    """Open an image file as a PIL image (thin wrapper for testability)."""
    from PIL import Image
    return Image.open(path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Automated duplicate-target proposal")
    ap.add_argument("--cache-dir", required=True,
                    help="directory holding manifest.csv and frames/")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-thresh", type=float, default=DEFAULT_SCORE_THRESH)
    ap.add_argument("--high", type=float, default=DEFAULT_HIGH)
    ap.add_argument("--low", type=float, default=DEFAULT_LOW)
    args = ap.parse_args()

    run_auto_pass(
        args.cache_dir,
        score_thresh=args.score_thresh,
        high=args.high,
        low=args.low,
        limit=args.limit,
    )
