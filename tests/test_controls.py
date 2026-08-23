"""
tests/test_controls.py: unit tests for the stimulus transforms and the condition
factorial in controls.py, and for the spatial-axis mapping in data.py.

Synthetic images and instructions only; no model, no network, and no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from controls import (
    CONDITIONS,
    DEFAULT_CONDITIONS,
    IMAGE_MIRROR,
    IMAGE_ORIGINAL,
    ROLE_A,
    ROLE_B,
    ROLE_NEUTRAL,
    apply_image_transform,
    build_scene_swap,
    conditions_for,
    grey_image,
    mirror_image,
    noise_image,
    plan_stimuli,
    strip_spatial_term,
    substitute_spatial_term,
    swapped_scene_id,
)
from data import (
    AXIS_INDEX,
    AXIS_LATERAL,
    AXIS_SCENE_DEPENDENT,
    SWAP,
    TERM_AXIS,
    is_lateral_term,
    term_axis,
    term_axis_index,
)


def _image(width=8, height=6, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Axis mapping
# --------------------------------------------------------------------------- #
def test_term_axis_covers_every_swappable_term():
    assert set(SWAP) <= set(TERM_AXIS)


def test_antonyms_share_an_axis():
    # A pair that contrasted two different axes could not have one expected
    # component, so the swap would have no well-defined direction.
    for term, opposite in SWAP.items():
        assert term_axis(term) == term_axis(opposite), (term, opposite)


def test_lateral_terms_map_to_dx():
    for term in ("left", "right", "leftmost", "rightmost"):
        assert is_lateral_term(term)
        assert term_axis_index(term) == AXIS_INDEX[AXIS_LATERAL] == 0


def test_scene_dependent_terms_have_no_fixed_component():
    for term in ("closer to", "farther from"):
        assert term_axis(term) == AXIS_SCENE_DEPENDENT
        assert term_axis_index(term) is None


def test_term_axis_is_case_insensitive_and_safe_on_unknown_terms():
    assert term_axis("  LEFT ") == AXIS_LATERAL
    assert term_axis("sideways") is None
    assert term_axis_index("sideways") is None


# --------------------------------------------------------------------------- #
# Image transforms
# --------------------------------------------------------------------------- #
def test_mirror_is_an_involution():
    image = _image()
    assert np.array_equal(np.asarray(mirror_image(mirror_image(image))),
                          np.asarray(image))


def test_mirror_reverses_horizontal_order():
    image = _image()
    assert np.array_equal(np.asarray(mirror_image(image)),
                          np.asarray(image)[:, ::-1, :])


def test_mirror_changes_an_asymmetric_image():
    array = np.zeros((4, 6, 3), dtype=np.uint8)
    array[:, 0] = 255
    image = Image.fromarray(array)
    assert not np.array_equal(np.asarray(mirror_image(image)), array)


def test_grey_and_noise_preserve_size():
    image = _image(width=10, height=7)
    assert grey_image(image).size == image.size
    assert noise_image(image).size == image.size


def test_noise_is_deterministic_in_its_seed():
    image = _image()
    assert np.array_equal(np.asarray(noise_image(image, seed=3)),
                          np.asarray(noise_image(image, seed=3)))
    assert not np.array_equal(np.asarray(noise_image(image, seed=3)),
                              np.asarray(noise_image(image, seed=4)))


def test_apply_image_transform_dispatch():
    image = _image()
    assert np.array_equal(np.asarray(apply_image_transform(IMAGE_ORIGINAL, image)),
                          np.asarray(image))
    assert np.array_equal(np.asarray(apply_image_transform(IMAGE_MIRROR, image)),
                          np.asarray(mirror_image(image)))
    with pytest.raises(ValueError):
        apply_image_transform("upside_down", image)


# --------------------------------------------------------------------------- #
# Scene swap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [2, 3, 5, 17, 64])
def test_derangement_never_assigns_a_scene_to_itself(size):
    ids = [f"s{i}" for i in range(size)]
    mapping = build_scene_swap(ids, seed=1)
    assert set(mapping) == set(ids)
    assert all(key != value for key, value in mapping.items())


def test_derangement_is_a_permutation():
    ids = [f"s{i}" for i in range(20)]
    mapping = build_scene_swap(ids, seed=2)
    assert sorted(mapping.values()) == sorted(ids)


def test_derangement_is_deterministic():
    ids = [f"s{i}" for i in range(12)]
    assert build_scene_swap(ids, seed=7) == build_scene_swap(ids, seed=7)


def test_derangement_requires_two_scenes():
    with pytest.raises(ValueError):
        build_scene_swap(["only"], seed=0)


def test_swapped_scene_id_matches_the_mapping():
    ids = [f"s{i}" for i in range(6)]
    mapping = build_scene_swap(ids, seed=0)
    assert swapped_scene_id("s2", ids, seed=0) == mapping["s2"]


# --------------------------------------------------------------------------- #
# Instruction transforms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("instruction,term,expected", [
    ("pick up the cup on the left", "left", "pick up the cup"),
    ("grab the red block on the right", "right", "grab the red block"),
    ("push the cloth to the right", "right", "push the cloth"),
    ("Move the silver pot to the back", "back", "Move the silver pot"),
    # Modifier of a following noun: the term goes, the carrier stays.
    ("put the carrot in the top drawer", "top", "put the carrot in the drawer"),
    ("take the leftmost bowl", "leftmost", "take the bowl"),
    ("lift the bottom plate", "bottom", "lift the plate"),
    # Relational prepositions govern a landmark, which must come out too.
    ("put the spoon behind the bowl", "behind", "put the spoon"),
    ("put the cup in front of the bowl", "in front of", "put the cup"),
    ("move the can closer to the plate", "closer to", "move the can"),
])
def test_strip_removes_the_spatial_content_and_stays_grammatical(
        instruction, term, expected):
    assert strip_spatial_term(instruction, term) == expected


def test_strip_removes_a_stranded_landmark():
    result = strip_spatial_term("Place the can to the left of the pot.", "left")
    assert result == "Place the can."


def test_strip_never_leaves_a_swappable_term_behind():
    for instruction, term in (("pick up the cup on the left", "left"),
                              ("put the carrot in the top drawer", "top"),
                              ("put the spoon behind the bowl", "behind")):
        result = strip_spatial_term(instruction, term)
        assert result is not None
        assert not any(other in result.lower() for other in SWAP)


def test_strip_never_ends_on_a_dangling_word():
    dangling = {"the", "a", "an", "of", "to", "on", "in", "at", "from"}
    for term in SWAP:
        for template in ("pick up the cup on the {t}", "move it {t}",
                         "put the bowl {t} the plate", "grab the {t} spoon"):
            result = strip_spatial_term(template.format(t=term), term)
            if result is not None:
                assert result.lower().split()[-1].strip(".,;") not in dangling


def test_strip_refuses_rather_than_returning_a_fragment():
    # No such term in the instruction, and a result too short to be an
    # instruction at all.
    assert strip_spatial_term("pick up the cup", "left") is None
    assert strip_spatial_term("left", "left") is None
    assert strip_spatial_term("", "left") is None
    assert strip_spatial_term("pick up the cup on the left", "") is None


def test_substitute_only_applies_where_it_reads_naturally():
    assert (substitute_spatial_term("pick up the cup on the left", "left")
            == "pick up the cup on the table")
    # A substitution here would produce "the table drawer" or "the table of the
    # pot", so the transform declines instead.
    assert substitute_spatial_term("put the carrot in the top drawer", "top") is None
    assert substitute_spatial_term("Place the can to the left of the pot.",
                                   "left") is None
    assert substitute_spatial_term("put the spoon behind the bowl", "behind") is None


def test_substitute_preserves_length_roughly():
    original = "pick up the cup on the left"
    result = substitute_spatial_term(original, "left")
    assert abs(len(result.split()) - len(original.split())) <= 1


# --------------------------------------------------------------------------- #
# Condition factorial
# --------------------------------------------------------------------------- #
def test_mirror_conditions_apply_only_to_lateral_terms():
    lateral = {c.name for c in conditions_for("left")}
    depth = {c.name for c in conditions_for("behind")}
    assert "mirror" in lateral and "mirror_neutral" in lateral
    assert "mirror" not in depth and "mirror_neutral" not in depth


def test_default_conditions_exclude_the_out_of_distribution_ablations():
    # Grey and noise images can drive the model to a constant action, which
    # would mimic an absent lexical prior; the swapped-scene control replaces
    # them in the default set.
    assert "grey" not in DEFAULT_CONDITIONS and "noise" not in DEFAULT_CONDITIONS
    assert "swapped_scene" in DEFAULT_CONDITIONS


def test_every_condition_declares_known_roles():
    for condition in CONDITIONS.values():
        assert condition.roles
        assert set(condition.roles) <= {ROLE_A, ROLE_B, ROLE_NEUTRAL}


def _scene(scene_id="s0", term="left"):
    return {
        "scene_id": scene_id,
        "spatial_term": term,
        "instr_a": f"pick up the cup on the {term}",
        "instr_b": f"pick up the cup on the {SWAP[term]}",
    }


def test_plan_stimuli_expands_the_lateral_factorial():
    scenes = [_scene("s0"), _scene("s1"), _scene("s2")]
    swap_map = build_scene_swap([s["scene_id"] for s in scenes], seed=0)
    stimuli = plan_stimuli(scenes[0], swap_map=swap_map)
    assert {(s.condition, s.role) for s in stimuli} == {
        ("baseline", "a"), ("baseline", "b"),
        ("neutral", "n"),
        ("mirror", "a"), ("mirror_neutral", "n"),
        ("swapped_scene", "a"), ("swapped_scene", "b"),
    }


def test_swapped_scene_stimuli_draw_a_different_scenes_image():
    scenes = [_scene(f"s{i}") for i in range(4)]
    swap_map = build_scene_swap([s["scene_id"] for s in scenes], seed=0)
    for scene in scenes:
        for stimulus in plan_stimuli(scene, swap_map=swap_map):
            if stimulus.condition == "swapped_scene":
                assert stimulus.image_scene_id != scene["scene_id"]
            else:
                assert stimulus.image_scene_id == scene["scene_id"]


def test_swapped_scene_is_skipped_without_an_assignment():
    stimuli = plan_stimuli(_scene(), swap_map=None)
    assert all(s.condition != "swapped_scene" for s in stimuli)


def test_neutral_conditions_are_skipped_when_the_strip_refuses():
    scene = {
        "scene_id": "s0",
        "spatial_term": "left",
        # No clean removal: nothing remains but the term's carrier.
        "instr_a": "left",
        "instr_b": "right",
    }
    stimuli = plan_stimuli(scene, swap_map=None)
    assert all(s.role != ROLE_NEUTRAL for s in stimuli)


def test_plan_stimuli_is_deterministic():
    scenes = [_scene(f"s{i}") for i in range(5)]
    ids = [s["scene_id"] for s in scenes]
    first = plan_stimuli(scenes[0], swap_map=build_scene_swap(ids, seed=0))
    second = plan_stimuli(scenes[0], swap_map=build_scene_swap(ids, seed=0))
    assert first == second
