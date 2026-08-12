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
     confirmation in the scene-review notebook.

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

# Detection and decision thresholds. A box is
# only counted when its score clears DEFAULT_SCORE_THRESH. The proposal is then
# decided from the second-highest surviving score: two confident instances imply
# a duplicate target, a weak or absent second instance implies a single target,
# and the band between the two is left for manual confirmation.
DEFAULT_SCORE_THRESH = 0.10
DEFAULT_HIGH = 0.30
DEFAULT_LOW = 0.15

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

# Module-level model cache so the detector is loaded once per process.
_OWL_CACHE: dict = {}


def extract_target_noun(instruction: str, tags=None) -> str | None:
    """Return the head noun the spatial term selects, or None if undetermined.

    A regex heuristic is used first because the instructions are short
    imperatives (for example "grab the red block on the right"), a form that
    general-purpose parsers routinely mis-tag by reading the leading verb as a
    noun or the object noun as a verb. The heuristic strips action verbs,
    determiners, prepositions, and spatial cues and returns the content word
    immediately preceding the first spatial cue, which for these instructions is
    the selected object. A spaCy parse is used only as a fallback when the
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


def _extract_with_spacy(text: str) -> str | None:
    """Head noun of the first object noun chunk via spaCy, or None if spaCy is
    unavailable or finds no noun chunk."""
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
    location_words = SPATIAL_TOKENS | _SPATIAL_PHRASE_WORDS
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.text.lower() in location_words:
            continue
        if root.pos_ in {"NOUN", "PROPN"}:
            return root.lemma_.lower() or root.text.lower()
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

    before = [w for w in words[:cue_index] if w not in _STOP_WORDS]
    if before:
        return before[-1]
    remaining = [w for w in words if w not in _STOP_WORDS]
    return remaining[0] if remaining else None


def classify_counts(
    scores,
    *,
    high: float = DEFAULT_HIGH,
    low: float = DEFAULT_LOW,
) -> tuple[str, float]:
    """Map detection scores to a duplicate-target proposal and a confidence.

    The decision is driven by the second-highest detection score, which stands
    for the confidence that a second instance of the object exists:

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


def _load_owl():
    """Load and cache the OWLv2 processor and model on the available device."""
    if "owl" in _OWL_CACHE:
        return _OWL_CACHE["owl"]
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(OWLV2_CHECKPOINT)
    model = Owlv2ForObjectDetection.from_pretrained(OWLV2_CHECKPOINT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _OWL_CACHE["owl"] = (processor, model, device)
    return _OWL_CACHE["owl"]


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


def detect_instances(image, noun: str, *, score_thresh: float = DEFAULT_SCORE_THRESH):
    """Detect instances of `noun` in `image`, returning scores with boxes.

    `image` may be a PIL image or an HxWx3 array. The query is the target noun
    phrased as "a photo of a <noun>", the standard OWLv2 prompt form. Instances
    are returned in descending score order.

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
    target_sizes = torch.tensor([image.size[::-1]], device=device)
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
        Instance(score=float(score), box=tuple(float(v) for v in box))
        for score, box in zip(results["scores"].tolist(), results["boxes"].tolist())
    ]
    instances.sort(key=lambda inst: inst.score, reverse=True)
    return instances


def count_instances(image, noun: str, *, score_thresh: float = DEFAULT_SCORE_THRESH):
    """Return the per-box detection scores for `noun` in `image` above threshold.

    Thin wrapper over `detect_instances` for callers that need only the scores.
    Scores are returned in descending order.
    """
    return [inst.score for inst in
            detect_instances(image, noun, score_thresh=score_thresh)]


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
            "duplicate_note": f"auto: noun={noun}, n_boxes={len(scores)}, "
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
