"""
tests/test_data.py: unit tests for the grasp-detection and category heuristic
in data.py. Uses synthetic step objects only; no network access or the
TensorFlow/TFDS stack is required.
"""

from __future__ import annotations

import csv
import os

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
    _categorise,
    classify_instruction,
    grasp_index,
    make_pair,
    review_queue,
    review_summary,
    update_manifest_annotations,
)
from detect_duplicates import classify_counts, extract_target_noun


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
    """Write a minimal manifest.csv under tmp_path and return the directory."""
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
