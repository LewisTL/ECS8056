"""
tests/test_data.py: unit tests for the grasp-detection and category heuristic
in data.py. Uses synthetic step objects only; no network access or the
TensorFlow/TFDS stack is required.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from data import (
    CATEGORY_OTHER,
    CATEGORY_PLACEMENT,
    CATEGORY_REFERENT,
    CATEGORY_SOURCE_HEURISTIC,
    DUPLICATE_SOURCE_AUTO,
    DUPLICATE_SOURCE_MANUAL,
    FEASIBLE_DEFAULT,
    MANIFEST_FIELDS,
    PROTOBUF_SPEC,
    SPLIT_CONSTRUCTION,
    SPLIT_UNASSIGNED,
    SPLIT_VALIDATION,
    _categorise,
    append_manifest_rows,
    classify_instruction,
    grasp_index,
    harvest_records,
    load_manifest,
    make_pair,
    protobuf_runtime_problem,
    read_harvest_state,
    release_components,
    review_queue,
    review_summary,
    split_counts,
    update_manifest_annotations,
    validation_eligible,
)
from detect_duplicates import (Instance, box_overlap, classify_counts, clip_box,
                               extract_manipulated_noun, extract_target_noun,
                               is_object_noun, padded_target_size,
                               suppress_overlapping)


def _step(open_gripper: bool) -> dict:
    """Build a synthetic step exposing only the field grasp_index reads."""
    return {"action": {"open_gripper": open_gripper}}


def test_grasp_index_normal_open_close_open():
    steps = [_step(True), _step(True), _step(False), _step(False), _step(True)]
    assert grasp_index(steps) == 2


def test_grasp_index_never_closes():
    steps = [_step(True), _step(True), _step(True)]
    assert grasp_index(steps) is None


def test_grasp_index_closed_from_start():
    steps = [_step(False), _step(False), _step(False)]
    assert grasp_index(steps) is None


def test_grasp_index_transition_on_final_step():
    steps = [_step(True), _step(True), _step(False)]
    assert grasp_index(steps) is None


def test_grasp_index_multiple_close_open_cycles():
    # The first transition is at index 1; the later transition at index 3 (a
    # regrasp) must be ignored.
    steps = [_step(True), _step(False), _step(True), _step(False), _step(True)]
    assert grasp_index(steps) == 1


def test_categorise_referent_selection_direct():
    assert _categorise(["left"], []) == CATEGORY_REFERENT


def test_categorise_placement_relation_direct():
    assert _categorise(["right"], ["to the right"]) == CATEGORY_PLACEMENT


def test_classify_instruction_referent_selection():
    tags = classify_instruction("pick up the cup on the left")
    assert tags.category == "referent_selection"


def test_classify_instruction_placement_relation():
    tags = classify_instruction("move the cloth to the right of the colander")
    assert tags.category == "placement_relation"


def test_make_pair_single_term_swap():
    result = make_pair("pick up the cup on the left")
    assert result == ("left", "pick up the cup on the right")


def test_make_pair_rejects_zero_terms():
    assert make_pair("pick up the cup") is None


def test_make_pair_rejects_multiple_terms():
    assert make_pair("move the left cup to the right") is None


def test_make_pair_phrase_priority_over_token():
    # "in front of" must match as a phrase, not the bare token "front".
    result = make_pair("put the spoon in front of the bowl")
    assert result is not None
    term, swapped = result
    assert term == "in front of"
    assert swapped == "put the spoon behind the bowl"


def _write_manifest(tmp_path, rows: list[dict]) -> str:
    """Write a minimal manifest.csv under tmp_path and return the directory.

    Rows default to the validation split, since that is the scope manual review
    covers; a test exercising the split filter sets the field explicitly.
    """
    out_dir = str(tmp_path)
    path = os.path.join(out_dir, "manifest.csv")
    normalised = []
    for row in rows:
        full = {col: "" for col in MANIFEST_FIELDS}
        full.update(row)
        if not full["feasible_both"]:
            full["feasible_both"] = FEASIBLE_DEFAULT
        if not full["category_source"]:
            full["category_source"] = CATEGORY_SOURCE_HEURISTIC
        # Absent means "the usual case"; an explicit empty string means a frame
        # cached before the roles existed, which the split filter must exclude.
        if "split" not in row:
            full["split"] = SPLIT_VALIDATION
        normalised.append(full)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in normalised:
            writer.writerow(row)
    return out_dir


def test_review_queue_priority_and_filters(tmp_path):
    out_dir = _write_manifest(tmp_path, [
        {
            "episode_index": "10",
            "instruction": "put the cloth to the right of the colander",
            "category": CATEGORY_PLACEMENT,
            "grasp_image_path": "frames/ep_000010_grasp.png",
            "feasible_both": "unreviewed",
        },
        {
            "episode_index": "3",
            "instruction": "pick up the cup on the left",
            "category": CATEGORY_REFERENT,
            "image_path": "frames/ep_000003.png",
            "feasible_both": "unreviewed",
        },
        {
            "episode_index": "7",
            "instruction": "pick up the cup on the right",
            "category": CATEGORY_REFERENT,
            "image_path": "frames/ep_000007.png",
            "feasible_both": "yes",
        },
        {
            "episode_index": "20",
            "instruction": "put the spoon next to the bowl",
            "category": CATEGORY_PLACEMENT,
            "image_path": "frames/ep_000020.png",
            "feasible_both": "unreviewed",
        },
        {
            "episode_index": "30",
            "instruction": "pick up the cup",
            "category": CATEGORY_OTHER,
            "image_path": "frames/ep_000030.png",
            "feasible_both": "unreviewed",
        },
    ])

    queue = review_queue(out_dir, status="unreviewed", only_pairable=True)
    # Referent first, then placement-with-grasp; non-pairable and already-yes excluded.
    assert [int(it["episode_index"]) for it in queue] == [3, 10]
    assert queue[0]["instr_b"] == "pick up the cup on the right"
    assert queue[0]["spatial_term"] == "left"

    yes_only = review_queue(out_dir, status="yes", only_pairable=True)
    assert [int(it["episode_index"]) for it in yes_only] == [7]

    with_non_pairable = review_queue(
        out_dir, status="unreviewed", only_pairable=False, categories=[CATEGORY_OTHER]
    )
    assert [int(it["episode_index"]) for it in with_non_pairable] == [30]
    assert with_non_pairable[0]["pairable"] is False


def test_review_summary_counts(tmp_path):
    out_dir = _write_manifest(tmp_path, [
        {
            "episode_index": "1",
            "instruction": "pick up the cup on the left",
            "category": CATEGORY_REFERENT,
            "feasible_both": "unreviewed",
        },
        {
            "episode_index": "2",
            "instruction": "pick up the cup on the right",
            "category": CATEGORY_REFERENT,
            "feasible_both": "yes",
            "duplicate_target": "yes",
        },
        {
            "episode_index": "3",
            "instruction": "pick up the cup",
            "category": CATEGORY_OTHER,
            "feasible_both": "unreviewed",
        },
    ])
    summary = review_summary(out_dir)
    assert summary["total"] == 3
    assert summary["pairable"] == 2
    assert summary["non_pairable"] == 1
    assert summary["referent_pairable_unreviewed"] == 1
    assert summary["referent_pairable_yes"] == 1
    assert summary["by_category_feasibility"][(CATEGORY_REFERENT, "yes")] == 1
    # Episode 2 is referent, pairable, feasible on both sides, and duplicate.
    assert summary["primary_eligible"] == 1
    assert summary["referent_pairable_dup_yes"] == 1
    assert summary["referent_pairable_dup_unreviewed"] == 1


def test_review_queue_duplicate_status_filter(tmp_path):
    out_dir = _write_manifest(tmp_path, [
        {
            "episode_index": "1",
            "instruction": "pick up the cup on the left",
            "category": CATEGORY_REFERENT,
            "image_path": "frames/ep_000001.png",
            "feasible_both": "unreviewed",
            "duplicate_target": "unclear",
        },
        {
            "episode_index": "2",
            "instruction": "pick up the cup on the right",
            "category": CATEGORY_REFERENT,
            "image_path": "frames/ep_000002.png",
            "feasible_both": "unreviewed",
            "duplicate_target": "yes",
        },
    ])
    # Status=None keeps every feasibility value; the duplicate filter selects
    # only the auto-flagged borderline scene.
    unclear = review_queue(out_dir, status=None, duplicate_status="unclear")
    assert [int(it["episode_index"]) for it in unclear] == [1]


def test_update_manifest_annotations_rejects_bad_feasible(tmp_path):
    out_dir = _write_manifest(tmp_path, [
        {
            "episode_index": "1",
            "instruction": "pick up the cup on the left",
            "category": CATEGORY_REFERENT,
            "feasible_both": "unreviewed",
        },
    ])
    with pytest.raises(ValueError, match="feasible_both"):
        update_manifest_annotations(out_dir, {1: {"feasible_both": "maybe"}})


def test_update_manifest_annotations_rejects_bad_duplicate(tmp_path):
    out_dir = _write_manifest(tmp_path, [
        {
            "episode_index": "1",
            "instruction": "pick up the cup on the left",
            "category": CATEGORY_REFERENT,
            "feasible_both": "unreviewed",
        },
    ])
    with pytest.raises(ValueError, match="duplicate_target"):
        update_manifest_annotations(out_dir, {1: {"duplicate_target": "maybe"}})


def test_update_manifest_annotations_writes_duplicate_source(tmp_path):
    out_dir = _write_manifest(tmp_path, [
        {
            "episode_index": "1",
            "instruction": "pick up the cup on the left",
            "category": CATEGORY_REFERENT,
            "feasible_both": "unreviewed",
        },
        {
            "episode_index": "2",
            "instruction": "pick up the cup on the right",
            "category": CATEGORY_REFERENT,
            "feasible_both": "unreviewed",
        },
    ])
    update_manifest_annotations(out_dir, {
        1: {"duplicate_target": "yes", "duplicate_source": DUPLICATE_SOURCE_AUTO},
        2: {"duplicate_target": "no", "duplicate_source": DUPLICATE_SOURCE_MANUAL},
    })
    items = {int(it["episode_index"]): it
             for it in review_queue(out_dir, status=None, only_pairable=False)}
    assert items[1]["duplicate_target"] == "yes"
    assert items[1]["duplicate_source"] == DUPLICATE_SOURCE_AUTO
    assert items[2]["duplicate_target"] == "no"
    assert items[2]["duplicate_source"] == DUPLICATE_SOURCE_MANUAL


# --------------------------------------------------------------------------- #
# Set roles and the harvest
# --------------------------------------------------------------------------- #
def test_validation_eligible_requires_lateral_referent_pair():
    # A lateral term selecting a referent: usable unaltered.
    assert validation_eligible("pick up the cup on the left")
    assert validation_eligible("take the leftmost fork")
    # A destination phrase: the term names where the object goes, and the first
    # action is the same reach either way.
    assert not validation_eligible("move the pot to the right of the spoon")
    # A non-lateral swap is not comparable with the constructed experiments.
    assert not validation_eligible("put the spoon in front of the bowl")
    # No swappable term at all.
    assert not validation_eligible("put the corn in the pot")
    # Two swappable terms would not be a minimal pair.
    assert not validation_eligible("move the left cup to the right")


class _FakeTags:
    def __init__(self, instruction):
        tags = classify_instruction(instruction)
        self.category = tags.category
        self.spatial_tokens = tags.spatial_tokens
        self.spatial_phrases = tags.spatial_phrases
        self.has_transfer = tags.has_transfer
        self.has_spatial = tags.has_spatial
        self.is_multi_object = tags.is_multi_object
        self.matched = tags.matched


class _FakeRecord:
    """The subset of EpisodeRecord the harvest reads, without TFDS or a stream."""

    def __init__(self, episode_index: int, instruction: str):
        self.episode_index = episode_index
        self.instruction = instruction
        self.image = np.zeros((4, 6, 3), dtype=np.uint8)
        self.gt_vector = np.zeros(7, dtype=np.float32)
        self.num_steps = 12
        self.tags = _FakeTags(instruction)
        self.grasp_frame_index = None
        self.grasp_image = None
        self.grasp_gt_vector = None


def _records(specs):
    return [_FakeRecord(index, instruction) for index, instruction in specs]


def test_harvest_assigns_disjoint_roles(tmp_path):
    out_dir = str(tmp_path)
    summary = harvest_records(
        _records([
            (0, "pick up the cup on the left"),      # validation eligible
            (1, "put the corn in the pot"),          # construction base
            (2, "move the pot to the right of the spoon"),  # construction base
            (3, "take the leftmost fork"),           # validation eligible
        ]),
        out_dir, validation_target=None, construction_target=None, verbose=False,
    )
    rows = {int(r["episode_index"]): r for r in load_manifest(out_dir)}
    assert rows[0]["split"] == SPLIT_VALIDATION
    assert rows[3]["split"] == SPLIT_VALIDATION
    assert rows[1]["split"] == SPLIT_CONSTRUCTION
    assert rows[2]["split"] == SPLIT_CONSTRUCTION
    assert summary["totals"] == {SPLIT_VALIDATION: 2, SPLIT_CONSTRUCTION: 2}
    # Every cached frame carries exactly one role, so no frame can serve both.
    assert not (set(r["episode_index"] for r in load_manifest(out_dir)
                    if r["split"] == SPLIT_VALIDATION)
                & set(r["episode_index"] for r in load_manifest(out_dir)
                      if r["split"] == SPLIT_CONSTRUCTION))


def test_harvest_keeps_eligible_frames_out_of_the_construction_pool(tmp_path):
    """A frame that could be a validation trial is never used as a base frame.

    Otherwise enlarging the validation set later would mean either re-streaming or
    admitting frames that had already been composited into stimuli.
    """
    out_dir = str(tmp_path)
    harvest_records(
        _records([(0, "pick up the cup on the left"),
                  (1, "pick up the mug on the right"),
                  (2, "put the corn in the pot")]),
        out_dir, validation_target=1, construction_target=None, verbose=False,
    )
    rows = {int(r["episode_index"]): r["split"] for r in load_manifest(out_dir)}
    # The second eligible frame arrives after the validation target is met and is
    # still cached as validation rather than falling through to construction.
    assert rows[1] == SPLIT_VALIDATION


def test_harvest_stops_once_both_targets_are_met(tmp_path):
    out_dir = str(tmp_path)
    summary = harvest_records(
        _records([(0, "pick up the cup on the left"),
                  (1, "put the corn in the pot"),
                  (2, "put the fork on the plate"),
                  (3, "put the knife on the board")]),
        out_dir, validation_target=1, construction_target=1, verbose=False,
    )
    assert summary["targets_met"]
    assert summary["streamed"] == 2
    assert summary["next_split_start"] == 2


def test_harvest_resumes_without_duplicating(tmp_path):
    out_dir = str(tmp_path)
    first = _records([(0, "pick up the cup on the left"),
                      (1, "put the corn in the pot")])
    harvest_records(first, out_dir, validation_target=None,
                    construction_target=None, verbose=False)
    # A resumed session re-offers what it already cached plus something new.
    summary = harvest_records(
        first + _records([(2, "put the fork on the plate")]),
        out_dir, validation_target=None, construction_target=None, verbose=False,
    )
    indices = [int(r["episode_index"]) for r in load_manifest(out_dir)]
    assert sorted(indices) == [0, 1, 2]
    assert len(indices) == 3
    assert summary["skipped_existing"] == 2
    assert read_harvest_state(out_dir)["next_split_start"] == 3


def test_harvest_state_records_the_stream_position_past_uncached_episodes(tmp_path):
    """The manifest cannot supply the resume point on its own.

    Once the construction target is met, later episodes are streamed and cached by
    neither role, so the highest cached index understates how far the stream got and
    a resumed run would re-stream what it already rejected.
    """
    out_dir = str(tmp_path)
    harvest_records(
        _records([(0, "put the corn in the pot"),
                  (1, "put the fork on the plate"),
                  (2, "put the knife on the board")]),
        out_dir, validation_target=None, construction_target=1, verbose=False,
    )
    cached = [int(r["episode_index"]) for r in load_manifest(out_dir)]
    assert cached == [0]
    assert read_harvest_state(out_dir)["next_split_start"] == 3


def test_append_manifest_rows_widens_a_narrow_header(tmp_path):
    """Appending under an assumed header is what corrupted the prediction log.

    A file whose header does not cover every column being written is rewritten
    under the union first, so the rows stay readable by name.
    """
    out_dir = str(tmp_path)
    path = os.path.join(out_dir, "manifest.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index", "instruction"])
        writer.writeheader()
        writer.writerow({"episode_index": "7", "instruction": "old row"})
    append_manifest_rows(out_dir, [{"episode_index": "8", "instruction": "new row",
                                    "split": SPLIT_CONSTRUCTION}])
    rows = {int(r["episode_index"]): r for r in load_manifest(out_dir)}
    assert rows[7]["instruction"] == "old row"
    assert rows[7]["split"] == SPLIT_UNASSIGNED
    assert rows[8]["split"] == SPLIT_CONSTRUCTION


def test_review_queue_scopes_to_the_validation_split(tmp_path):
    out_dir = _write_manifest(tmp_path, [
        {"episode_index": "1", "instruction": "pick up the cup on the left",
         "category": CATEGORY_REFERENT, "split": SPLIT_VALIDATION},
        {"episode_index": "2", "instruction": "pick up the mug on the left",
         "category": CATEGORY_REFERENT, "split": SPLIT_CONSTRUCTION},
        {"episode_index": "3", "instruction": "pick up the pan on the left",
         "category": CATEGORY_REFERENT, "split": SPLIT_UNASSIGNED},
    ])
    assert [it["episode_index"] for it in review_queue(out_dir)] == ["1"]
    # Construction frames are not reviewed as validation trials, and a frame
    # cached before the roles existed belongs to neither scope.
    assert len(review_queue(out_dir, splits=None)) == 3
    assert len(review_queue(out_dir, splits=(SPLIT_CONSTRUCTION,))) == 1


def test_review_summary_reports_splits_and_scopes_its_counters(tmp_path):
    out_dir = _write_manifest(tmp_path, [
        {"episode_index": "1", "instruction": "pick up the cup on the left",
         "category": CATEGORY_REFERENT, "feasible_both": "yes",
         "duplicate_target": "yes", "split": SPLIT_VALIDATION},
        {"episode_index": "2", "instruction": "pick up the mug on the left",
         "category": CATEGORY_REFERENT, "feasible_both": "yes",
         "duplicate_target": "yes", "split": SPLIT_CONSTRUCTION},
    ])
    summary = review_summary(out_dir)
    assert summary["by_split"] == {SPLIT_VALIDATION: 1, SPLIT_CONSTRUCTION: 1}
    assert summary["total"] == 2
    assert summary["in_scope"] == 1
    assert summary["primary_eligible"] == 1
    assert review_summary(out_dir, splits=None)["primary_eligible"] == 2


def test_split_counts_treats_a_missing_split_as_unassigned():
    assert split_counts([{"split": SPLIT_VALIDATION}, {}, {"split": ""}]) == {
        SPLIT_VALIDATION: 1, SPLIT_UNASSIGNED: 2}


def test_classify_counts_bands():
    # Two confident detections above the high threshold -> duplicate.
    label, conf = classify_counts([0.9, 0.8, 0.2], high=0.3, low=0.15)
    assert label == "yes"
    assert conf == 0.8
    # A single strong detection with no credible second instance -> single.
    label, conf = classify_counts([0.9, 0.05], high=0.3, low=0.15)
    assert label == "no"
    # A borderline second detection -> left for manual confirmation.
    label, _ = classify_counts([0.9, 0.2], high=0.3, low=0.15)
    assert label == "unclear"
    # No detections at all -> single target.
    label, conf = classify_counts([], high=0.3, low=0.15)
    assert label == "no"
    assert conf == 0.0


def test_extract_target_noun():
    # The object selected by the spatial term, not the location word.
    assert extract_target_noun("pick up the cup on the left") == "cup"
    assert extract_target_noun("grab the red block on the right") == "block"
    assert extract_target_noun("") is None


def test_extract_manipulated_noun_takes_the_object_not_the_landmark():
    """The manipulated object, which differs from the referent when a landmark is
    present. Synthesised construction needs the object the episode demonstrates is
    graspable, not the thing a spatial term happens to select.
    """
    assert extract_manipulated_noun("put the corn in the pot on the left") == "corn"
    assert extract_manipulated_noun("take the lid off the pot") == "lid"
    assert extract_manipulated_noun("put the sushi on the plate") == "sushi"
    # The referent extractor disagrees on exactly these, which is the point.
    assert extract_target_noun("put the corn in the pot on the left") == "pot"


def test_extract_manipulated_noun_skips_modifiers_and_particles():
    assert extract_manipulated_noun("grab the red block") == "block"
    assert extract_manipulated_noun("move the silver pot to the left") == "pot"
    assert extract_manipulated_noun("pick up the cup") == "cup"
    assert extract_manipulated_noun("put down the cloth") == "cloth"


def test_extract_manipulated_noun_rejects_state_words():
    """An adverb read as an object would be written into a synthesised instruction."""
    assert extract_manipulated_noun("flip the pot upright") == "pot"
    assert extract_manipulated_noun("turn the cup over") == "cup"
    assert extract_manipulated_noun("") is None


def test_extract_manipulated_noun_passes_over_a_trailing_adverb():
    """The head position at the end of the phrase is where adverbs land.

    Taking the last content word is what skips leading modifiers, and it is also
    what promotes a manner adverb into the query. The observed failures are
    reproduced here: each was sent to the detector as an object name.
    """
    assert extract_manipulated_noun("move the pot diagonally") == "pot"
    assert extract_manipulated_noun("push the bowl forward") == "bowl"
    assert extract_manipulated_noun("move the cup directly to the left") == "cup"
    assert extract_manipulated_noun("slide the plate all the way over") == "plate"
    # A landmark introduced by a location word still closes the object phrase.
    assert extract_manipulated_noun("put the corn outside the pot") == "corn"


def test_extract_noun_returns_nothing_rather_than_a_word_that_names_no_object():
    """A frame is better dropped than queried with a word that names no object."""
    assert extract_manipulated_noun("move it to the left") is None
    assert extract_manipulated_noun("push forward") is None


def test_is_object_noun_rejects_adverbs_it_was_never_given():
    """Manner adverbs are open-class; the suffix rule covers the ones not listed."""
    assert is_object_noun("banana")
    assert is_object_noun("pot")
    assert not is_object_noun("vertically")
    assert not is_object_noun("gingerly")
    # Object names sharing the suffix are exempted rather than dropping the rule.
    assert is_object_noun("jelly")


def test_box_overlap_scores_a_nested_box_as_full_overlap():
    """Intersection over union would score a part inside a whole as almost none.

    The detector returns a box on an object and further boxes on its parts, and
    they have to read as the same instance.
    """
    whole = (100.0, 100.0, 200.0, 200.0)
    part = (140.0, 140.0, 170.0, 170.0)
    assert box_overlap(whole, part) == pytest.approx(1.0)
    assert box_overlap(part, whole) == pytest.approx(1.0)
    assert box_overlap(whole, (300.0, 300.0, 400.0, 400.0)) == 0.0
    # Boxes sharing only an edge have no area in common.
    assert box_overlap(whole, (200.0, 100.0, 300.0, 200.0)) == 0.0
    # A degenerate box cannot be said to overlap anything.
    assert box_overlap(whole, (150.0, 150.0, 150.0, 200.0)) == 0.0


def test_suppress_overlapping_keeps_one_detection_per_object():
    """Counting boxes is not counting instances.

    The detector emits a dense set of boxes and suppresses none of them, so a
    single confident object yields a descending cascade on itself that reads
    exactly like a second, less certain instance.
    """
    cascade = [Instance(score=0.28, box=(110.0, 105.0, 190.0, 195.0)),
               Instance(score=0.50, box=(100.0, 100.0, 200.0, 200.0)),
               Instance(score=0.19, box=(140.0, 140.0, 170.0, 170.0))]
    kept = suppress_overlapping(cascade)
    assert [i.score for i in kept] == [0.50]

    # A genuine second object elsewhere in the frame survives.
    two = cascade + [Instance(score=0.31, box=(300.0, 100.0, 380.0, 200.0))]
    assert [i.score for i in suppress_overlapping(two)] == [0.50, 0.31]
    # The highest-scoring box of a merged group is the one kept, since it is the
    # box the construction will cut.
    assert suppress_overlapping(cascade)[0].box == (100.0, 100.0, 200.0, 200.0)


def test_suppression_is_what_makes_the_second_score_mean_a_second_instance():
    """The duplicate-target proposal reads the second score as a second object."""
    cascade = [Instance(score=0.50, box=(100.0, 100.0, 200.0, 200.0)),
               Instance(score=0.40, box=(105.0, 105.0, 195.0, 195.0))]
    raw = classify_counts([i.score for i in cascade], high=0.3, low=0.15)[0]
    kept = classify_counts([i.score for i in suppress_overlapping(cascade)],
                           high=0.3, low=0.15)[0]
    assert raw == "yes"
    assert kept == "no"


def test_padded_target_size_uses_the_square_the_detector_pads_to():
    """OWLv2 boxes are normalised against a padded square, not the frame.

    The image processor pads to `max(height, width)` before resizing, so
    unnormalising with the frame's own height scales the shorter axis by too
    small a factor. On a 640x480 frame that pulls every box a quarter of the way
    toward the top and costs it a quarter of its height, which is the difference
    between cutting the object and cutting the background above it.
    """
    assert padded_target_size(640, 480) == (640, 640)
    assert padded_target_size(480, 640) == (640, 640)
    # A square frame needs no padding, so the size is unchanged.
    assert padded_target_size(256, 256) == (256, 256)


def test_padded_target_size_would_shift_a_tabletop_box_upward_if_ignored():
    """The magnitude of the defect, stated as the analysis would have seen it."""
    width, height = 640, 480
    side = padded_target_size(width, height)[0]
    # A box around an object sitting mid-frame, in normalised coordinates.
    y_norm = 340.0 / side
    correct = y_norm * side
    naive = y_norm * height
    assert correct - naive > 80.0


def test_clip_box_orders_corners_and_keeps_boxes_inside_the_frame():
    assert clip_box((10, 20, 30, 40), 100, 100) == (10.0, 20.0, 30.0, 40.0)
    # Predictions may land in the padded region, which is not real pixels.
    assert clip_box((-5, -5, 150, 150), 100, 80) == (0.0, 0.0, 100.0, 80.0)
    # Reversed corners are normalised rather than producing a negative extent.
    assert clip_box((60, 70, 20, 30), 100, 100) == (20.0, 30.0, 60.0, 70.0)


def test_release_components_ignores_suffixes():
    assert release_components("6.31.1") == (6, 31, 1)
    assert release_components("7.36.0") == (7, 36, 0)
    assert release_components("6.32.0rc1") == (6, 32, 0)
    assert release_components("unknown") == ()


def test_protobuf_runtime_accepts_the_supported_window():
    assert protobuf_runtime_problem("6.31.1") is None
    assert protobuf_runtime_problem("6.33.6") is None


def test_protobuf_runtime_rejects_a_runtime_below_the_generated_code():
    problem = protobuf_runtime_problem("5.29.6")
    assert problem is not None
    assert PROTOBUF_SPEC in problem
    assert "older" in problem


def test_protobuf_runtime_rejects_a_runtime_past_the_tfds_upper_bound():
    """Protobuf 7 removed the descriptor attribute released TFDS reads."""
    problem = protobuf_runtime_problem("7.36.0")
    assert problem is not None
    assert PROTOBUF_SPEC in problem
    assert "FieldDescriptor.label" in problem


def test_protobuf_runtime_defers_on_an_unparseable_version():
    """An unrecognised version string must not block streaming by itself."""
    assert protobuf_runtime_problem("") is None
    assert protobuf_runtime_problem("unknown") is None
