"""
tests/test_data.py: unit tests for the grasp-detection and category heuristic
in data.py. Uses synthetic step objects only; no network access or the
TensorFlow/TFDS stack is required.
"""

from __future__ import annotations

from data import (
    CATEGORY_PLACEMENT,
    CATEGORY_REFERENT,
    _categorise,
    classify_instruction,
    grasp_index,
)


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
