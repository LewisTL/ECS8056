"""
tests/test_compose.py: unit tests for the constructed-scene geometry, the
compositing, and the validation sample in compose_scenes.py.

Synthetic images only. The open-vocabulary detector is imported lazily inside
`locate_gripper` and `run_construction`, so nothing here downloads a model or
needs a GPU.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

import compose_scenes
from data import SPLIT_CONSTRUCTION, SPLIT_VALIDATION, make_pair

from compose_scenes import (
    CONFIG_OPPOSITE,
    CONFIG_SAME_LEFT,
    CONFIG_SAME_RIGHT,
    CONFIGURATIONS,
    CONSTRUCTED_FIELDS,
    DECISION_APPROVED,
    DECISION_REJECTED,
    DECISION_UNLABELLED,
    DEFAULT_FEATHER,
    DEFAULT_MIN_GAP,
    DEFAULT_PADDING,
    GATE_BAD_NOUN,
    GATE_MULTI,
    GATE_NO_INSTANCE,
    GATE_OK,
    GATE_WEAK,
    GRIPPER_SOURCE_DETECTED,
    GRIPPER_SOURCE_FALLBACK,
    GRIPPER_Y_UNRECORDED_FRACTION,
    IMAGE_X_TO_LATERAL_SIGN,
    MODE_INHERITED,
    MODE_SYNTHESISED,
    POOL_APPROVED,
    POOL_FROZEN,
    REJECT_NO_SECOND_INSTANCE,
    REJECT_PASTE_IMPLAUSIBLE,
    SINGLE_HIGH,
    SINGLE_SECOND_MAX,
    SYNTHESISED_TEMPLATE,
    agreement_pool,
    append_constructed_manifest,
    apply_validation_labels,
    approval_summary,
    arrangement_target_signs,
    resolve_scene_arrangement,
    approve,
    approved_ids,
    DEFAULT_TARGETS,
    pair_eligible_ids,
    project_yield,
    action_image_delta,
    build_review_rows,
    build_scene,
    cut_patch,
    diagnose_detection,
    evaluation_scenes,
    gate_base_frame,
    gripper_image_xy,
    sample_candidates,
    load_evaluation_set,
    load_review,
    plan_vertical,
    select_base_scenes,
    select_evaluation_set,
    summarise_diagnosis,
    surface_span,
    sync_review,
    draw_validation_sample,
    load_constructed_manifest,
    load_validation_sample,
    locate_gripper,
    manipulation_rate,
    paste_duplicate,
    plan_placement,
    resolve_targets,
    run_construction,
    validation_agreement,
    write_constructed_manifest,
    write_evaluation_set,
    write_review,
    write_validation_sample,
)

WIDTH, HEIGHT = 320, 240
GRIPPER_X = 160.0
BOX_HALF = 20
PATCH_WIDTH = 2 * BOX_HALF + 2 * DEFAULT_PADDING


def _manifest_rows(specs):
    """Cached manifest rows, in the construction role unless stated otherwise.

    Construction reads only frames the harvest assigned that role, so a fixture
    without one would be eligible for nothing.
    """
    return [{**spec, "split": spec.get("split", SPLIT_CONSTRUCTION)}
            for spec in specs]


def _frame(center_x: int):
    """A textured background with one solid, asymmetric object at `center_x`."""
    columns = np.arange(WIDTH)
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[..., 0] = (150 + 40 * np.sin(columns / 17.0)).astype(np.uint8)
    array[..., 1] = 140
    array[..., 2] = 120
    array[110:150, center_x - BOX_HALF:center_x + BOX_HALF] = (30, 90, 200)
    # An off-centre highlight, so a mirrored copy is distinguishable.
    array[113:121, center_x - BOX_HALF + 3:center_x - BOX_HALF + 11] = (240, 240, 255)
    box = (center_x - BOX_HALF, 110, center_x + BOX_HALF, 150)
    return Image.fromarray(array), box


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
def test_leftward_terms_name_the_smaller_image_x():
    for term in ("left", "leftmost"):
        assert resolve_targets(term, 200.0, 80.0) == (80.0, 200.0)


def test_rightward_terms_name_the_larger_image_x():
    for term in ("right", "rightmost"):
        assert resolve_targets(term, 200.0, 80.0) == (200.0, 80.0)


def test_target_resolution_does_not_depend_on_argument_order():
    assert resolve_targets("left", 80.0, 200.0) == resolve_targets("left", 200.0, 80.0)


# --------------------------------------------------------------------------- #
# Placement geometry
# --------------------------------------------------------------------------- #
def test_opposite_places_the_instances_either_side_of_the_start():
    placement = plan_placement(CONFIG_OPPOSITE, "left", 100.0, GRIPPER_X,
                               PATCH_WIDTH, WIDTH)
    assert placement is not None
    assert (placement.x_source - GRIPPER_X) * (placement.x_pasted - GRIPPER_X) < 0
    assert placement.offset_a * placement.offset_b < 0


def test_same_side_keeps_both_instances_on_one_side():
    placement = plan_placement(CONFIG_SAME_LEFT, "left", 120.0, GRIPPER_X,
                               PATCH_WIDTH, WIDTH)
    assert placement is not None
    assert placement.x_source < GRIPPER_X and placement.x_pasted < GRIPPER_X
    assert placement.offset_a * placement.offset_b > 0


def test_same_side_is_unavailable_on_the_side_the_source_does_not_occupy():
    # The source instance cannot be moved, so a left-side arrangement is
    # impossible when the only real instance sits on the right.
    assert plan_placement(CONFIG_SAME_LEFT, "left", 200.0, GRIPPER_X,
                          PATCH_WIDTH, WIDTH) is None
    assert plan_placement(CONFIG_SAME_RIGHT, "left", 120.0, GRIPPER_X,
                          PATCH_WIDTH, WIDTH) is None


def test_instances_never_overlap():
    for configuration in CONFIGURATIONS:
        for source_x in range(40, WIDTH - 40, 10):
            placement = plan_placement(configuration, "left", float(source_x),
                                       GRIPPER_X, PATCH_WIDTH, WIDTH)
            if placement is None:
                continue
            assert placement.separation >= DEFAULT_MIN_GAP * PATCH_WIDTH
            assert placement.separation >= PATCH_WIDTH


def test_placement_stays_inside_the_frame():
    for configuration in CONFIGURATIONS:
        for source_x in range(30, WIDTH - 30, 5):
            placement = plan_placement(configuration, "left", float(source_x),
                                       GRIPPER_X, PATCH_WIDTH, WIDTH)
            if placement is None:
                continue
            half = PATCH_WIDTH / 2.0
            assert 0 <= placement.x_pasted - half
            assert placement.x_pasted + half <= WIDTH


def test_placement_refuses_rather_than_clamping_when_there_is_no_room():
    # A patch wider than the frame cannot be placed anywhere legally.
    assert plan_placement(CONFIG_OPPOSITE, "left", 100.0, GRIPPER_X,
                          WIDTH * 2, WIDTH) is None


def test_expected_sign_follows_the_geometry_not_the_wording():
    left = plan_placement(CONFIG_OPPOSITE, "left", 100.0, GRIPPER_X,
                          PATCH_WIDTH, WIDTH)
    right = plan_placement(CONFIG_OPPOSITE, "right", 100.0, GRIPPER_X,
                           PATCH_WIDTH, WIDTH)
    assert left.expected_sign_image == -right.expected_sign_image
    assert left.expected_sign_image == int(np.sign(left.x_target_a - left.x_target_b))


def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError):
        plan_placement("diagonal", "left", 100.0, GRIPPER_X, PATCH_WIDTH, WIDTH)


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #
def _mask_for(box, shape=(HEIGHT, WIDTH)):
    """A silhouette narrower than its box, so the two cutouts differ."""
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = (int(v) for v in box)
    inset = 6
    mask[y0 + 4:y1 - 4, x0 + inset:x1 - inset] = True
    return mask


def test_cut_patch_includes_the_padding_and_clips_at_the_border():
    image, box = _frame(100)
    patch, crop, alpha = cut_patch(image, box, padding=DEFAULT_PADDING)
    assert patch.size == (crop[2] - crop[0], crop[3] - crop[1])
    assert crop[0] == box[0] - DEFAULT_PADDING
    assert alpha is None
    _, edge_crop, _ = cut_patch(image, (0, 0, 20, 20), padding=DEFAULT_PADDING)
    assert edge_crop[0] == 0 and edge_crop[1] == 0


def test_cut_patch_returns_the_mask_window_aligned_to_the_crop():
    image, box = _frame(100)
    patch, crop, alpha = cut_patch(image, box, padding=DEFAULT_PADDING,
                                   mask=_mask_for(box))
    assert alpha is not None
    assert alpha.shape == (patch.size[1], patch.size[0])
    # The silhouette is inset from the box, so the crop's border stays empty.
    assert not alpha[0].any() and not alpha[:, 0].any()


def test_paste_lands_where_it_was_asked_to():
    image, box = _frame(100)
    _, paste_box, _ = paste_duplicate(image, box, 220.0)
    center = 0.5 * (paste_box[0] + paste_box[2])
    assert abs(center - 220.0) <= 1.0


def test_paste_changes_the_target_region_and_leaves_the_source_alone():
    image, box = _frame(100)
    composited, paste_box, _ = paste_duplicate(image, box, 220.0)
    before, after = np.asarray(image), np.asarray(composited)
    target = after[paste_box[1]:paste_box[3], paste_box[0]:paste_box[2]]
    original = before[paste_box[1]:paste_box[3], paste_box[0]:paste_box[2]]
    assert not np.array_equal(target, original)
    assert np.array_equal(after[110:150, 80:120], before[110:150, 80:120])


def test_paste_is_mirrored():
    image, box = _frame(100)
    plain, _, _ = paste_duplicate(image, box, 220.0, mirror=False, feather=0)
    flipped, _, _ = paste_duplicate(image, box, 220.0, mirror=True, feather=0)
    assert not np.array_equal(np.asarray(plain), np.asarray(flipped))


def test_paste_is_deterministic():
    image, box = _frame(100)
    first, _, _ = paste_duplicate(image, box, 220.0)
    second, _, _ = paste_duplicate(image, box, 220.0)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_paste_preserves_the_image_size_and_mode():
    image, box = _frame(100)
    composited, _, _ = paste_duplicate(image, box, 220.0)
    assert composited.size == image.size and composited.mode == image.mode


def test_mask_paste_transfers_the_object_without_its_surrounding_background():
    """The point of the mask: only the object crosses, not the rectangle.

    Cutting along the box carries the table inside the rectangle with it, which
    is what makes a box paste read as a pasted rectangle. With a silhouette the
    pixels outside it must be left exactly as the background already was.
    """
    image, box = _frame(100)
    mask = _mask_for(box)
    composited, paste_box, mode = paste_duplicate(
        image, box, 220.0, mask=mask, feather=0)
    assert mode == "mask"

    before, after = np.asarray(image), np.asarray(composited)
    x0, y0, x1, y1 = paste_box
    # The silhouette as the paste saw it: cropped, then mirrored with the patch.
    patch_alpha = cut_patch(image, box, padding=DEFAULT_PADDING, mask=mask)[2][:, ::-1]
    outside = ~patch_alpha
    region_before = before[y0:y1, x0:x1]
    region_after = after[y0:y1, x0:x1]
    assert np.array_equal(region_after[outside], region_before[outside])
    assert not np.array_equal(region_after[patch_alpha], region_before[patch_alpha])


def test_box_paste_transfers_the_whole_rectangle():
    """Without a mask every pixel of the rectangle is replaced.

    This is the fallback's defect, asserted so the difference from the mask path
    is explicit rather than assumed.
    """
    image, box = _frame(100)
    composited, paste_box, mode = paste_duplicate(image, box, 220.0, feather=0)
    assert mode == "box"
    before, after = np.asarray(image), np.asarray(composited)
    x0, y0, x1, y1 = paste_box
    changed = (after[y0:y1, x0:x1] != before[y0:y1, x0:x1]).any(axis=-1)
    # The background is a horizontal gradient, so almost every column differs.
    assert changed.mean() > 0.5


def test_an_empty_mask_falls_back_to_the_box_rather_than_pasting_nothing():
    image, box = _frame(100)
    empty = np.zeros((HEIGHT, WIDTH), dtype=bool)
    composited, _, mode = paste_duplicate(image, box, 220.0, mask=empty)
    assert mode == "box"
    assert not np.array_equal(np.asarray(composited), np.asarray(image))


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #
def test_build_scene_records_the_geometry_it_actually_used():
    image, box = _frame(100)
    result = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                         0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X)
    assert result is not None
    composited, scene = result
    left, _, right, _ = (float(v) for v in scene.paste_box.split(","))
    assert abs(0.5 * (left + right) - scene.x_pasted) <= 1.0
    assert scene.x_source == 100.0
    assert scene.configuration == CONFIG_OPPOSITE
    assert scene.separation_px == abs(scene.x_pasted - scene.x_source)
    assert scene.offset_a_px == scene.x_target_a - scene.x_gripper
    assert composited.size == image.size


def test_build_scene_records_which_cutout_was_used():
    image, box = _frame(100)
    _, boxed = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                           0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X)
    _, masked = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                            0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X,
                            mask=_mask_for(box))
    assert boxed.cutout_mode == "box"
    assert masked.cutout_mode == "mask"
    assert "cutout_mode" in CONSTRUCTED_FIELDS


def test_build_scene_seats_the_duplicate_on_the_surface():
    """The recorded rows have to reflect the shift that was applied.

    Without a surface the duplicate keeps the source's row, which is the
    fronto-parallel assumption. With one it moves, and `y_shift_px` records by how
    much so the placement is auditable from the manifest alone.
    """
    image, box = _frame(100)
    surface = _oblique_surface()

    _, flat = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                          0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X)
    assert flat.surface_source == "none"
    assert flat.y_shift_px == 0.0
    assert flat.y_base_pasted == flat.y_base_source

    _, seated = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                            0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X,
                            surface=surface)
    assert seated.surface_source == "detected"
    assert seated.y_shift_px != 0.0
    assert seated.y_base_pasted == seated.y_base_source + seated.y_shift_px
    # The recorded base lands on the surface at the pasted column.
    assert surface[int(seated.y_base_pasted), int(round(seated.x_pasted))]


def test_build_scene_refuses_a_placement_off_the_surface():
    """A configuration that would paste into mid-air is skipped, not approximated.

    The surface is cut away at the column the duplicate would occupy, so there is
    nowhere on the table to put it.
    """
    image, box = _frame(100)
    surface = _oblique_surface()

    result = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                         0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X,
                         surface=surface)
    assert result is not None
    x_pasted = int(round(result[1].x_pasted))

    surface[:, x_pasted - 2:x_pasted + 3] = False
    assert build_scene(image, "12", "pick up the cup on the left", "cup", box,
                       0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X,
                       surface=surface) is None


def test_the_vertical_shift_leaves_the_lateral_geometry_untouched():
    """The measurement reads only lateral coordinates, so seating the duplicate
    must not disturb them."""
    image, box = _frame(100)
    _, flat = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                          0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X)
    _, seated = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                            0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X,
                            surface=_oblique_surface())
    for field in ("x_source", "x_pasted", "x_target_a", "x_target_b", "x_gripper",
                  "offset_a_px", "offset_b_px", "separation_px",
                  "expected_sign_image", "target_sign_a_image",
                  "target_sign_b_image"):
        assert getattr(flat, field) == getattr(seated, field)


def test_build_scene_labels_the_configuration_consistently_with_the_offsets():
    for configuration in CONFIGURATIONS:
        for source_x in (80, 120, 200, 240):
            image, box = _frame(source_x)
            result = build_scene(image, "12", "pick up the cup on the left", "cup",
                                 box, 0.7, configuration, x_gripper=GRIPPER_X)
            if result is None:
                continue
            _, scene = result
            same_side = scene.offset_a_px * scene.offset_b_px > 0
            assert same_side == (configuration != CONFIG_OPPOSITE)


def test_build_scene_swaps_the_instruction():
    image, box = _frame(100)
    _, scene = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                           0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X)
    assert scene.instr_a == "pick up the cup on the left"
    assert scene.instr_b == "pick up the cup on the right"
    assert scene.spatial_term == "left"


def test_build_scene_refuses_non_lateral_and_unpairable_instructions():
    image, box = _frame(100)
    # A depth term: the construction places instances left and right, so it
    # carries no expectation here.
    assert build_scene(image, "12", "put the cup behind the bowl", "cup", box, 0.7,
                       CONFIG_OPPOSITE, x_gripper=GRIPPER_X) is None
    # No swappable term at all.
    assert build_scene(image, "12", "pick up the cup", "cup", box, 0.7,
                       CONFIG_OPPOSITE, x_gripper=GRIPPER_X) is None
    # Two swappable terms would not be a minimal pair.
    assert build_scene(image, "12", "move the left cup to the right", "cup", box,
                       0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X) is None


def test_select_base_scenes_excludes_placement_instructions_by_default():
    """Placement scenes cannot seed a construction.

    In "move the pot to the right of the spoon" the term governs the landmark,
    not the object being duplicated, so the swap does not choose between the two
    instances and the trial measures nothing. Leaving these in was silently
    producing incoherent stimuli.
    """
    rows = _manifest_rows([
        {"episode_index": "1", "instruction": "pick up the cup on the left",
         "category": "referent_selection", "image_path": "frames/a.png"},
        {"episode_index": "2", "instruction": "move the pot to the right of the spoon",
         "category": "placement_relation", "image_path": "frames/b.png"},
    ])
    selected = select_base_scenes(rows)
    assert [row["episode_index"] for row in selected] == ["1"]
    # The wider pool stays reachable for inspection.
    assert len(select_base_scenes(rows, categories=None)) == 2


def test_synthesised_mode_opens_the_pool_to_frames_with_no_spatial_term():
    """The inherited pool is a thin slice of the cache and yielded nothing usable.

    A synthesised instruction needs only a nameable manipulated object, so a frame
    is eligible whatever its own instruction said and whatever its category.
    """
    rows = _manifest_rows([
        {"episode_index": "1", "instruction": "put the corn in the pot",
         "category": "placement_relation", "image_path": "frames/a.png"},
        {"episode_index": "2", "instruction": "fold the towel",
         "category": "other", "image_path": "frames/b.png"},
    ])
    assert select_base_scenes(rows) == []

    selected = select_base_scenes(rows, instruction_mode=MODE_SYNTHESISED)
    assert len(selected) == 2
    # The manipulated object, not the landmark.
    assert selected[0]["target_noun"] == "corn"
    assert selected[0]["instruction"] == "pick up the corn on the left"
    assert all(r["instruction_source"] == MODE_SYNTHESISED for r in selected)


def test_synthesised_mode_refuses_to_write_an_unsatisfiable_instruction():
    """A fixture cannot be picked up, whatever the arrangement.

    Writing "pick up the microwave on the left" would reintroduce the impossible
    request confound the constructed set exists to remove: non-compliance would
    then say nothing about spatial grounding.
    """
    rows = _manifest_rows([
        {"episode_index": "1", "instruction": "open the microwave",
         "category": "other", "image_path": "frames/a.png"},
        {"episode_index": "2", "instruction": "put the spoon in the sink",
         "category": "other", "image_path": "frames/b.png"},
        {"episode_index": "3", "instruction": "wipe the counter",
         "category": "other", "image_path": "frames/c.png"},
    ])
    selected = select_base_scenes(rows, instruction_mode=MODE_SYNTHESISED)
    # Only the spoon is liftable; the microwave and the counter are fixtures.
    assert [r["target_noun"] for r in selected] == ["spoon"]


def test_synthesised_instructions_still_form_a_lateral_minimal_pair():
    """The written wording has to survive the machinery it feeds."""
    made = make_pair(SYNTHESISED_TEMPLATE.format(noun="cup"))
    assert made is not None
    term, swapped = made
    assert term == "left" and swapped == "pick up the cup on the right"


def test_selection_records_the_noun_and_source_in_both_modes():
    rows = _manifest_rows([
        {"episode_index": "1", "instruction": "pick up the cup on the left",
         "category": "referent_selection", "image_path": "frames/a.png"}])
    inherited = select_base_scenes(rows)[0]
    assert inherited["target_noun"] == "cup"
    assert inherited["instruction_source"] == MODE_INHERITED
    assert inherited["instruction"] == "pick up the cup on the left"


def test_unknown_instruction_mode_is_rejected():
    with pytest.raises(ValueError):
        select_base_scenes([], instruction_mode="invented")


def test_select_base_scenes_prefers_a_manual_category_correction():
    rows = _manifest_rows([
        {"episode_index": "3", "instruction": "pick up the cup on the left",
         "category": "placement_relation",
         "category_manual": "referent_selection", "image_path": "frames/c.png"},
    ])
    assert len(select_base_scenes(rows)) == 1


def test_construction_never_builds_on_a_validation_frame():
    """The two roles are disjoint, and this is where that is enforced.

    A frame composited into an experimental stimulus cannot also be evidence that
    the finding holds on unedited data, so a frame reserved for validation is
    refused as a base frame rather than filtered out further downstream.
    """
    rows = _manifest_rows([
        {"episode_index": "1", "instruction": "pick up the cup on the left",
         "category": "referent_selection", "image_path": "frames/a.png",
         "split": SPLIT_VALIDATION},
        {"episode_index": "2", "instruction": "pick up the mug on the left",
         "category": "referent_selection", "image_path": "frames/b.png",
         "split": SPLIT_CONSTRUCTION},
    ])
    assert [r["episode_index"] for r in select_base_scenes(rows)] == ["2"]
    assert [r["episode_index"] for r in
            select_base_scenes(rows, instruction_mode=MODE_SYNTHESISED)] == ["2"]
    # A frame cached before the roles existed belongs to neither, so it cannot
    # enter the experiments by default.
    unassigned = _manifest_rows([
        {"episode_index": "3", "instruction": "pick up the cup on the left",
         "category": "referent_selection", "image_path": "frames/c.png",
         "split": ""}])
    assert select_base_scenes(unassigned) == []
    assert len(select_base_scenes(unassigned, splits=None)) == 1


def test_build_scene_is_deterministic():
    image, box = _frame(100)
    first = build_scene(image, "12", "pick up the cup on the left", "cup", box, 0.7,
                        CONFIG_OPPOSITE, x_gripper=GRIPPER_X)
    second = build_scene(image, "12", "pick up the cup on the left", "cup", box, 0.7,
                         CONFIG_OPPOSITE, x_gripper=GRIPPER_X)
    assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
    assert first[1] == second[1]


def test_construct_ids_are_unique_across_configurations():
    ids = set()
    for configuration in CONFIGURATIONS:
        for source_x in (80, 120, 200):
            image, box = _frame(source_x)
            result = build_scene(image, str(source_x),
                                 "pick up the cup on the left", "cup", box, 0.7,
                                 configuration, x_gripper=GRIPPER_X)
            if result is None:
                continue
            assert result[1].construct_id not in ids
            ids.add(result[1].construct_id)


def test_manifest_round_trip(tmp_path):
    image, box = _frame(100)
    scenes = [build_scene(image, "12", "pick up the cup on the left", "cup", box,
                          0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X)[1]]
    write_constructed_manifest(scenes, str(tmp_path))
    rows = load_constructed_manifest(str(tmp_path))
    assert len(rows) == 1
    assert set(rows[0]) == set(CONSTRUCTED_FIELDS)
    assert rows[0]["construct_id"] == scenes[0].construct_id
    assert float(rows[0]["x_pasted"]) == pytest.approx(scenes[0].x_pasted)


def test_build_scene_records_the_supplied_gripper_centre():
    image, box = _frame(100)
    _, scene = build_scene(image, "12", "pick up the cup on the left", "cup", box,
                           0.7, CONFIG_OPPOSITE, x_gripper=GRIPPER_X,
                           y_gripper=48.0)
    assert scene.x_gripper == GRIPPER_X
    assert scene.y_gripper == 48.0
    assert "y_gripper" in CONSTRUCTED_FIELDS


def test_action_image_delta_projects_only_the_identified_lateral_channel():
    # Component 1 is the identified lateral channel and maps to image x
    # through the inverted sign convention.
    px, py = action_image_delta(0.0, 0.01, 0.0)
    assert px == pytest.approx(0.01 * IMAGE_X_TO_LATERAL_SIGN)
    assert py == pytest.approx(0.0)

    # The other components have no identified image direction and contribute
    # nothing to the overlay.
    px, py = action_image_delta(0.01, 0.0, 0.03)
    assert px == pytest.approx(0.0)
    assert py == pytest.approx(0.0)

    scaled = action_image_delta(0.01, 0.02, 0.03, scale=10.0)
    unit = action_image_delta(0.01, 0.02, 0.03, scale=1.0)
    assert scaled[0] == pytest.approx(10.0 * unit[0])
    assert scaled[1] == pytest.approx(10.0 * unit[1])


def test_gripper_image_xy_prefers_the_recorded_row():
    scene = {"x_gripper": 160.0, "y_gripper": 44.0}
    assert gripper_image_xy(scene, 240.0) == (160.0, 44.0)


def test_gripper_image_xy_uses_the_upper_frame_when_the_row_was_not_recorded():
    x, y = gripper_image_xy({"x_gripper": 160.0}, 240.0)
    assert x == 160.0
    assert y == pytest.approx(240.0 * GRIPPER_Y_UNRECORDED_FRACTION)
    x, y = gripper_image_xy({"x_gripper": 160.0, "y_gripper": ""}, 240.0)
    assert y == pytest.approx(240.0 * GRIPPER_Y_UNRECORDED_FRACTION)


def test_locate_gripper_returns_the_box_centre(monkeypatch):
    from detect_duplicates import Instance

    def detect(image, query, score_thresh=0.0):
        return [Instance(score=0.8, box=(100.0, 20.0, 140.0, 60.0))]

    monkeypatch.setattr("detect_duplicates.detect_instances", detect)
    image, _ = _frame(100)
    x, y, source, score = locate_gripper(image)
    assert source == GRIPPER_SOURCE_DETECTED
    assert x == pytest.approx(120.0)
    assert y == pytest.approx(40.0)
    assert score == pytest.approx(0.8)


def test_locate_gripper_falls_back_to_the_image_centre(monkeypatch):
    monkeypatch.setattr("detect_duplicates.detect_instances",
                        lambda *args, **kwargs: [])
    image, _ = _frame(100)
    x, y, source, score = locate_gripper(image)
    assert source == GRIPPER_SOURCE_FALLBACK
    assert x == pytest.approx(image.width / 2.0)
    assert y == pytest.approx(image.height / 2.0)
    assert score == 0.0


# --------------------------------------------------------------------------- #
# Validation sample
# --------------------------------------------------------------------------- #
def _scene_rows(n=30):
    rows = []
    for i in range(n):
        configuration = CONFIG_OPPOSITE if i % 2 else CONFIG_SAME_LEFT
        rows.append({
            "construct_id": f"c{i:06d}_{configuration}_left",
            "configuration": configuration,
            "instr_a": "pick up the cup on the left",
            "gripper_source": "detected",
            "cutout_mode": "mask" if i % 3 else "box",
        })
    return rows


def test_validation_sample_is_stratified_and_deterministic():
    rows = _scene_rows()
    first = draw_validation_sample(rows, n=10, seed=0)
    second = draw_validation_sample(rows, n=10, seed=0)
    assert [r["construct_id"] for r in first] == [r["construct_id"] for r in second]
    assert {r["configuration"] for r in first} == {CONFIG_OPPOSITE, CONFIG_SAME_LEFT}


def test_validation_sample_never_repeats_a_scene():
    drawn = draw_validation_sample(_scene_rows(), n=30, seed=1)
    ids = [r["construct_id"] for r in drawn]
    assert len(ids) == len(set(ids))


def test_validation_sample_caps_at_the_pool_size():
    assert len(draw_validation_sample(_scene_rows(6), n=50, seed=0)) == 6


def test_validation_labels_survive_a_redraw(tmp_path):
    rows = _scene_rows()
    write_validation_sample(draw_validation_sample(rows, n=10, seed=0), str(tmp_path))
    labelled = load_validation_sample(str(tmp_path))
    labelled[0]["human_configuration"] = labelled[0]["auto_configuration"]
    labelled[0]["human_gripper_ok"] = "yes"
    write_validation_sample(labelled, str(tmp_path))

    write_validation_sample(draw_validation_sample(rows, n=10, seed=0), str(tmp_path))
    after = load_validation_sample(str(tmp_path))
    assert sum(1 for r in after if r["human_configuration"]) == 1


def test_validation_sample_keeps_scenes_already_labelled():
    """Labelling is scarce, so a redraw must not discard the work.

    The pool grows as screening continues, and a draw that ignored the labels
    already collected would select a different set every time it ran.
    """
    rows = _scene_rows()
    first = draw_validation_sample(rows, n=10, seed=0)
    kept = {r["construct_id"] for r in first}

    # A different seed alone would select a largely different set.
    fresh = draw_validation_sample(rows, n=10, seed=5)
    assert {r["construct_id"] for r in fresh} != kept

    again = draw_validation_sample(rows, n=10, seed=5, retain=kept)
    assert kept <= {r["construct_id"] for r in again}
    assert len(again) == 10


def test_validation_sample_drops_retained_scenes_outside_the_pool():
    """A label on a scene the pool excludes cannot enter the agreement number."""
    rows = _scene_rows(10)
    absent = "c999999_opposite_left"
    drawn = draw_validation_sample(rows, n=4, seed=0, retain={absent})
    assert absent not in {r["construct_id"] for r in drawn}


def test_validation_sample_records_the_pool_it_came_from():
    drawn = draw_validation_sample(_scene_rows(), n=6, seed=0, pool=POOL_APPROVED)
    assert all(r["pool"] == POOL_APPROVED for r in drawn)


def test_a_redraw_updates_the_pool_but_keeps_the_labels(tmp_path):
    """The pool is a pipeline column, not a label, so a redraw refreshes it.

    A sample drawn before the freeze and redrawn after it describes the frozen
    set, and a stale column would attribute the agreement number to the wrong
    set.
    """
    out_dir = str(tmp_path)
    rows = _scene_rows()
    write_validation_sample(
        draw_validation_sample(rows, n=6, seed=0, pool=POOL_APPROVED), out_dir)
    labelled = load_validation_sample(out_dir)
    labelled[0]["human_configuration"] = labelled[0]["auto_configuration"]
    write_validation_sample(labelled, out_dir)

    write_validation_sample(
        draw_validation_sample(rows, n=6, seed=0, pool=POOL_FROZEN), out_dir)
    after = load_validation_sample(out_dir)
    assert all(r["pool"] == POOL_FROZEN for r in after)
    assert sum(1 for r in after if r["human_configuration"]) == 1


def test_apply_validation_labels_annotates_the_manifest_without_touching_geometry(tmp_path):
    """Hand labels land on the frame's own row; the recorded geometry does not move.

    A disagreement must stay visible on the manifest rather than replacing the
    recorded arrangement, and a scene without a label must stay empty so an
    unlabelled scene cannot read as an agreeing one.
    """
    out_dir = str(tmp_path)
    scenes = _pool(n_left=1, n_right=1)
    write_constructed_manifest(scenes, out_dir)
    write_validation_sample(draw_validation_sample(scenes, n=2, seed=0), out_dir)

    sample = load_validation_sample(out_dir)
    sample[0]["human_configuration"] = "unclear"
    sample[0]["human_two_instances"] = "yes"
    sample[0]["human_gripper_ok"] = "yes"
    sample[0]["human_paste_plausible"] = "no"
    write_validation_sample(sample, out_dir)

    result = apply_validation_labels(out_dir)
    assert result["labelled"] == 1
    assert result["configuration_disagreements"] == 1

    rows = {r["construct_id"]: r for r in load_constructed_manifest(out_dir)}
    annotated = rows[sample[0]["construct_id"]]
    assert annotated["human_configuration"] == "unclear"
    assert annotated["human_paste_plausible"] == "no"
    assert annotated["configuration"] == sample[0]["auto_configuration"]
    untouched = [r for cid, r in rows.items()
                 if cid != sample[0]["construct_id"]]
    assert all(r["human_configuration"] == "" for r in untouched)


def test_apply_validation_labels_counts_agreement_as_no_disagreement(tmp_path):
    out_dir = str(tmp_path)
    scenes = _pool(n_left=1, n_right=1)
    write_constructed_manifest(scenes, out_dir)
    write_validation_sample(draw_validation_sample(scenes, n=1, seed=0), out_dir)

    sample = load_validation_sample(out_dir)
    sample[0]["human_configuration"] = sample[0]["auto_configuration"]
    write_validation_sample(sample, out_dir)

    result = apply_validation_labels(out_dir)
    assert result["labelled"] == 1
    assert result["configuration_disagreements"] == 0


def _arrangement_scene(**overrides):
    scene = {
        "construct_id": "c000001_opposite_left",
        "configuration": CONFIG_OPPOSITE,
        "expected_sign_image": 1,
        "target_sign_a_image": 1,
        "target_sign_b_image": -1,
        "human_configuration": "",
    }
    scene.update(overrides)
    return scene


def test_arrangement_target_signs_follow_the_arrangement():
    assert arrangement_target_signs(CONFIG_SAME_LEFT, 1) == (-1, -1)
    assert arrangement_target_signs(CONFIG_SAME_RIGHT, 1) == (1, 1)
    assert arrangement_target_signs(CONFIG_OPPOSITE, 1) == (1, -1)
    assert arrangement_target_signs(CONFIG_OPPOSITE, -1) == (-1, 1)


def test_unlabelled_and_agreeing_scenes_keep_the_recorded_arrangement():
    for label in ("", CONFIG_OPPOSITE):
        resolved = resolve_scene_arrangement(
            _arrangement_scene(human_configuration=label))
        assert resolved == {
            "configuration": CONFIG_OPPOSITE,
            "target_sign_a_image": 1,
            "target_sign_b_image": -1,
            "relabelled": False,
        }


def test_an_unclear_label_drops_the_scene():
    assert resolve_scene_arrangement(
        _arrangement_scene(human_configuration="unclear")) is None


def test_a_disagreeing_label_wins_and_rederives_the_target_signs():
    """The human arrangement decides how the scene is probed.

    An opposite scene relabelled same-side means the gripper was mislocalised,
    so the gripper-relative signs are re-derived from the label rather than
    kept from the discredited coordinates. The instances' relative order is
    exact by construction, so a scene relabelled opposite splits its signs by
    `expected_sign_image`.
    """
    to_same_side = resolve_scene_arrangement(
        _arrangement_scene(human_configuration=CONFIG_SAME_RIGHT))
    assert to_same_side == {
        "configuration": CONFIG_SAME_RIGHT,
        "target_sign_a_image": 1,
        "target_sign_b_image": 1,
        "relabelled": True,
    }

    to_opposite = resolve_scene_arrangement(_arrangement_scene(
        configuration=CONFIG_SAME_LEFT, expected_sign_image=-1,
        target_sign_a_image=-1, target_sign_b_image=-1,
        human_configuration=CONFIG_OPPOSITE))
    assert to_opposite == {
        "configuration": CONFIG_OPPOSITE,
        "target_sign_a_image": -1,
        "target_sign_b_image": 1,
        "relabelled": True,
    }


def test_validation_agreement_counts_only_labelled_rows():
    rows = draw_validation_sample(_scene_rows(), n=10, seed=0)
    assert validation_agreement(rows)["labelled"] == 0

    for i, row in enumerate(rows):
        row["human_configuration"] = (row["auto_configuration"] if i % 2
                                      else "unclear")
        row["human_gripper_ok"] = "yes"
    result = validation_agreement(rows)
    assert result["labelled"] == len(rows)
    assert result["configuration_agreement"] == pytest.approx(0.5)
    assert result["gripper_agreement"] == pytest.approx(1.0)


def test_validation_sample_carries_the_cutout_mode():
    rows = draw_validation_sample(_scene_rows(), n=10, seed=0)
    assert all(r["auto_cutout_mode"] in {"mask", "box"} for r in rows)


def test_validation_agreement_separates_paste_quality_by_cutout():
    """A box cutout is a different stimulus and is reported separately.

    Pooling the two would average away the distinction that decides whether any
    box fallbacks are usable at all.
    """
    rows = draw_validation_sample(_scene_rows(), n=30, seed=0)
    for row in rows:
        row["human_configuration"] = row["auto_configuration"]
        # The outline cutouts read as objects; the box cutouts do not.
        row["human_paste_plausible"] = (
            "yes" if row["auto_cutout_mode"] == "mask" else "no")

    result = validation_agreement(rows)
    assert result["by_cutout"]["mask"]["paste_plausible_rate"] == pytest.approx(1.0)
    assert result["by_cutout"]["box"]["paste_plausible_rate"] == pytest.approx(0.0)
    # The pooled rate sits between the two and would hide the difference.
    assert 0.0 < result["paste_plausible_rate"] < 1.0


# --------------------------------------------------------------------------- #
# Support surface geometry
# --------------------------------------------------------------------------- #
def _oblique_surface(shape=(HEIGHT, WIDTH)):
    """A trapezoidal table, as an obliquely viewed surface appears in the frame.

    The far and near edges both rise from left to right, so the rows the surface
    occupies depend on the column. This is the geometry that a constant-y paste
    gets wrong.
    """
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for x in range(width):
        t = x / (width - 1)
        top = int(round(120 - 40 * t))
        bottom = int(round(220 - 40 * t))
        mask[top:bottom + 1, x] = True
    return mask


def test_surface_span_returns_the_rows_the_surface_occupies():
    mask = _oblique_surface()
    left = surface_span(mask, 0)
    right = surface_span(mask, WIDTH - 1)
    assert left == (120, 220)
    assert right == (80, 180)
    # Outside the frame there is no column to report.
    assert surface_span(mask, -1) is None
    assert surface_span(mask, WIDTH) is None


def test_surface_span_ignores_specks_away_from_the_surface():
    """The longest run, not the full extent.

    A segmentation may pick up an unrelated speck in the same column, and taking
    the extent would then report every row between it and the table as surface.
    """
    mask = _oblique_surface()
    mask[10, 100] = True
    assert surface_span(mask, 100) == (120 - int(round(40 * 100 / (WIDTH - 1))),
                                       220 - int(round(40 * 100 / (WIDTH - 1))))


def test_surface_span_reports_a_column_with_no_surface_as_absent():
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    assert surface_span(mask, WIDTH // 2) is None


def test_plan_vertical_moves_the_base_with_the_surface():
    """The whole point: a lateral copy has to change row to stay on the table.

    The surface rises 40 px from the left edge to the right, so a copy moved from
    one to the other has to rise with it. Holding the row constant is what left
    duplicates floating above the table or standing in front of its near edge.
    """
    mask = _oblique_surface()
    y_base = 200.0
    planned = plan_vertical(mask, 0.0, y_base, WIDTH - 1)
    assert planned is not None
    assert planned == pytest.approx(160, abs=1)
    # And the result is on the surface, which the constant-row choice is not.
    assert mask[planned, WIDTH - 1]
    assert not mask[int(y_base), WIDTH - 1]


def test_plan_vertical_preserves_relative_depth_on_the_surface():
    """Constant fraction of the span, so apparent distance is roughly unchanged
    and the patch needs no rescaling."""
    mask = _oblique_surface()
    top_source, bottom_source = surface_span(mask, 10)
    y_base = top_source + 0.25 * (bottom_source - top_source)

    planned = plan_vertical(mask, 10.0, y_base, 300)
    top_target, bottom_target = surface_span(mask, 300)
    fraction = (planned - top_target) / (bottom_target - top_target)
    assert fraction == pytest.approx(0.25, abs=0.02)


def test_plan_vertical_refuses_a_column_with_no_surface():
    """A duplicate cannot be placed where there is no table to place it on."""
    mask = _oblique_surface()
    mask[:, 300] = False
    assert plan_vertical(mask, 10.0, 200.0, 300) is None
    # And an absent source column is equally unusable.
    assert plan_vertical(_oblique_surface(), 10.0, 200.0, WIDTH + 50) is None


def test_plan_vertical_clamps_a_base_recorded_off_the_surface():
    """A base slightly outside the run still yields a placement on the surface."""
    mask = _oblique_surface()
    planned = plan_vertical(mask, 10.0, 5.0, 300)
    assert planned is not None
    assert mask[planned, 300]


def test_paste_applies_the_vertical_shift():
    image, box = _frame(100)
    _, level, _ = paste_duplicate(image, box, 220.0, dy=0.0)
    _, raised, _ = paste_duplicate(image, box, 220.0, dy=-30.0)
    assert raised[1] == level[1] - 30
    assert raised[3] == level[3] - 30


# --------------------------------------------------------------------------- #
# Base-frame gate and detection diagnostic
# --------------------------------------------------------------------------- #
class _Detection:
    def __init__(self, score):
        self.score = score


def test_gate_names_the_reason_a_frame_cannot_seed_a_construction():
    assert gate_base_frame([]) == GATE_NO_INSTANCE
    assert gate_base_frame([_Detection(SINGLE_HIGH - 0.01)]) == GATE_WEAK
    assert gate_base_frame([_Detection(0.9)]) == GATE_OK
    # A credible second instance means the frame is already a two-instance trial
    # with a geometry that was not chosen.
    assert gate_base_frame([_Detection(0.9),
                            _Detection(SINGLE_SECOND_MAX)]) == GATE_MULTI
    # A weak second detection is noise, not an instance.
    assert gate_base_frame([_Detection(0.9),
                            _Detection(SINGLE_SECOND_MAX - 0.01)]) == GATE_OK


def test_gate_thresholds_are_arguments_so_they_can_be_calibrated():
    """The score scale belongs to the detector, not to this data.

    A frame the default rejected as weak must become admissible under a threshold
    chosen from the observed distribution, without editing the module.
    """
    weak = [_Detection(SINGLE_HIGH - 0.02)]
    assert gate_base_frame(weak) == GATE_WEAK
    assert gate_base_frame(weak, single_high=SINGLE_HIGH - 0.05) == GATE_OK


def test_gate_ignores_the_order_detections_arrive_in():
    ordered = gate_base_frame([_Detection(0.9), _Detection(0.05)])
    reversed_ = gate_base_frame([_Detection(0.05), _Detection(0.9)])
    assert ordered == reversed_ == GATE_OK


def test_gate_counts_objects_because_boxes_reach_it_already_suppressed():
    """The cascade of boxes a detector emits for one object is not two instances.

    `detect_instances` suppresses overlapping boxes, so the second score the gate
    sees belongs to a second object. The behaviour is verified end to end here:
    the raw output of the detector on a single object would otherwise be ruled
    `multi_instance`, which is what the suppression exists to prevent.
    """
    from detect_duplicates import Instance, suppress_overlapping

    one_object = [Instance(score=0.50, box=(100.0, 100.0, 200.0, 200.0)),
                  Instance(score=0.28, box=(110.0, 105.0, 190.0, 195.0)),
                  Instance(score=0.19, box=(140.0, 140.0, 170.0, 170.0))]
    assert gate_base_frame(one_object) == GATE_MULTI
    assert gate_base_frame(suppress_overlapping(one_object)) == GATE_OK

    two_objects = [Instance(score=0.50, box=(100.0, 100.0, 200.0, 200.0)),
                   Instance(score=0.28, box=(240.0, 100.0, 300.0, 200.0))]
    assert gate_base_frame(suppress_overlapping(two_objects)) == GATE_MULTI


def test_sampling_shuffles_before_it_truncates():
    """A prefix of the manifest is a few Bridge domains, not a sample of them.

    The diagnostic calibrates the thresholds on what this returns, so it has to
    draw the same way the build does.
    """
    pool = [{"episode_index": str(i)} for i in range(50)]
    drawn = sample_candidates(pool, limit=10, seed=0)
    assert len(drawn) == 10
    assert drawn != pool[:10]
    assert drawn == sample_candidates(pool, limit=10, seed=0)
    assert sample_candidates(pool, limit=10, shuffle=False) == pool[:10]


def test_diagnosis_refuses_a_query_that_does_not_name_an_object(tmp_path,
                                                               monkeypatch):
    """An adverb read as a noun is an extraction fault, not a weak detection.

    Running the detector on it would return a low-scoring box on an arbitrary
    region, which is indistinguishable from a real object scored weakly and would
    drag the quantiles the threshold is chosen from.
    """
    import detect_duplicates

    cache_dir, rows = _cache(tmp_path, [80, 240])
    rows[0]["target_noun"] = "diagonally"
    rows[1]["target_noun"] = "cup"

    def detect(image, noun, **kwargs):
        assert noun == "cup", "a non-object query reached the detector"
        return [detect_duplicates.Instance(score=0.4, box=(60.0, 110.0,
                                                           100.0, 150.0))]

    monkeypatch.setattr(detect_duplicates, "detect_instances", detect)
    records = diagnose_detection(cache_dir, rows, shuffle=False)
    assert [r["verdict"] for r in records] == [GATE_BAD_NOUN, GATE_OK]
    assert records[0]["scores"] == []

    summary = summarise_diagnosis(records)
    assert summary["recoverable_by_lowering_single_high"] == 0
    assert summary["with_any_detection"] == 1


def test_diagnosis_rules_at_the_threshold_the_build_uses(tmp_path, monkeypatch):
    """Scores are reported from the floor, but the verdict is the build's.

    Detecting at the floor is what makes a near-miss visible; ruling at the floor
    would report a distribution of verdicts no build ever produces.
    """
    import detect_duplicates

    cache_dir, rows = _cache(tmp_path, [80])
    rows[0]["target_noun"] = "cup"

    def detect(image, noun, **kwargs):
        return [detect_duplicates.Instance(score=0.08, box=(60.0, 110.0,
                                                            100.0, 150.0))]

    monkeypatch.setattr(detect_duplicates, "detect_instances", detect)
    record = diagnose_detection(cache_dir, rows, shuffle=False)[0]
    # The detection is reported, since a near miss is evidence about the
    # threshold, but the build would never see it and so the frame holds nothing.
    assert record["scores"] == [0.08]
    assert record["verdict"] == GATE_NO_INSTANCE
    assert diagnose_detection(cache_dir, rows, shuffle=False,
                              working_thresh=0.02)[0]["verdict"] == GATE_WEAK


def test_diagnosis_summary_separates_a_wrong_query_from_a_strict_threshold():
    """The two causes of a zero yield need opposite responses.

    A frame with no detection at all indicts the queried noun; a frame whose best
    score sits just under the threshold indicts the threshold. The summary has to
    count them apart and say where a workable threshold would sit.
    """
    records = [
        {"verdict": GATE_NO_INSTANCE, "scores": []},
        {"verdict": GATE_NO_INSTANCE, "scores": []},
        {"verdict": GATE_WEAK, "scores": [0.20]},
        {"verdict": GATE_WEAK, "scores": [0.22]},
        {"verdict": GATE_WEAK, "scores": [0.24]},
        {"verdict": GATE_OK, "scores": [0.55]},
    ]
    summary = summarise_diagnosis(records)
    assert summary["frames"] == 6
    assert summary["with_any_detection"] == 4
    assert summary["recoverable_by_lowering_single_high"] == 3
    assert summary["by_verdict"][GATE_NO_INSTANCE] == 2
    # The quantiles are what a threshold is chosen from.
    assert 0.20 <= summary["best_score_quantiles"]["p50"] <= 0.24
    assert summary["best_score_max"] == pytest.approx(0.55)


def test_diagnosis_summary_handles_a_run_with_no_detections_at_all():
    summary = summarise_diagnosis([{"verdict": GATE_NO_INSTANCE, "scores": []}])
    assert summary["with_any_detection"] == 0
    assert summary["best_score_quantiles"] == {}


# --------------------------------------------------------------------------- #
# Construction run: resume, targets, and sampling
# --------------------------------------------------------------------------- #
def _cache(tmp_path, centers):
    """A cache directory of frames plus the manifest rows that name them."""
    cache_dir = str(tmp_path / "cache")
    os.makedirs(os.path.join(cache_dir, "frames"), exist_ok=True)
    rows = []
    for index, center in enumerate(centers):
        image, _ = _frame(center)
        rel = os.path.join("frames", f"ep_{index:06d}.png")
        image.save(os.path.join(cache_dir, rel))
        rows.append({"episode_index": str(index),
                     "instruction": "pick up the cup on the left",
                     "category": "referent_selection", "image_path": rel,
                     "split": SPLIT_CONSTRUCTION})
    return cache_dir, rows


def _stub_perception(monkeypatch, centers):
    """Stand in for the detector, the segmenter, and the gripper locator.

    The construction geometry is covered by its own tests; what these exercise is
    the run's bookkeeping, which is where an interrupted session does its damage.
    """
    import detect_duplicates

    by_center = {center: (center - BOX_HALF, 110, center + BOX_HALF, 150)
                 for center in centers}

    def detect(image, noun, **kwargs):
        # The stub reads the frame it was handed to find its own object, so each
        # base frame keeps the arrangement the fixture gave it.
        array = np.asarray(image)
        for center, box in by_center.items():
            if tuple(array[130, center]) == (30, 90, 200):
                return [detect_duplicates.Instance(score=0.9, box=box)]
        return []

    def segment(image, box):
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[box[1]:box[3], box[0]:box[2]] = True
        return mask

    monkeypatch.setattr(detect_duplicates, "detect_instances", detect)
    monkeypatch.setattr(detect_duplicates, "segment_instance", segment)
    monkeypatch.setattr(compose_scenes, "locate_gripper",
                        lambda image: (image.width / 2.0, image.height / 2.0,
                                       "detected", 0.9))
    monkeypatch.setattr(
        compose_scenes, "locate_surface",
        lambda image, box: np.ones((image.height, image.width), dtype=bool))


def test_construction_stops_once_every_target_is_met(tmp_path, monkeypatch):
    """Construction is the expensive stage, so it stops when it has enough.

    Each frame supports its opposite arrangement and one same-side arrangement, so
    the run continues until both cells are full rather than exhausting the pool.
    """
    centers = [100, 100, 100, 100]
    cache_dir, rows = _cache(tmp_path, centers)
    _stub_perception(monkeypatch, centers)

    summary = run_construction(
        cache_dir, str(tmp_path / "built"), manifest_rows=rows, shuffle=False,
        targets={CONFIG_OPPOSITE: 2, CONFIG_SAME_LEFT: 2, CONFIG_SAME_RIGHT: 0},
        verbose=False)
    assert summary["goals_met"]
    assert summary["processed"] == 2
    assert summary["by_configuration"] == {CONFIG_OPPOSITE: 2, CONFIG_SAME_LEFT: 2}


def test_construction_overbuilds_by_the_stated_factor(tmp_path, monkeypatch):
    """Approval rejects some scenes, so more are built than the set will hold."""
    centers = [100] * 6
    cache_dir, rows = _cache(tmp_path, centers)
    _stub_perception(monkeypatch, centers)

    summary = run_construction(
        cache_dir, str(tmp_path / "built"), manifest_rows=rows, shuffle=False,
        targets={CONFIG_OPPOSITE: 2, CONFIG_SAME_LEFT: 2, CONFIG_SAME_RIGHT: 0},
        overbuild=2.0, verbose=False)
    assert summary["goals"] == {CONFIG_OPPOSITE: 4, CONFIG_SAME_LEFT: 4,
                               CONFIG_SAME_RIGHT: 0}
    assert summary["by_configuration"][CONFIG_SAME_LEFT] == 4


def test_construction_resumes_from_the_manifest_it_wrote(tmp_path, monkeypatch):
    """A disconnected session must not cost the scenes already built.

    Rows are appended as scenes are produced, and a later run skips the base frames
    the manifest already accounts for.
    """
    centers = [100, 100, 100]
    cache_dir, rows = _cache(tmp_path, centers)
    _stub_perception(monkeypatch, centers)
    out_dir = str(tmp_path / "built")

    first = run_construction(cache_dir, out_dir, manifest_rows=rows[:2],
                             shuffle=False, flush_every=1, verbose=False)
    assert first["constructed"] == 4
    assert len(load_constructed_manifest(out_dir)) == 4

    second = run_construction(cache_dir, out_dir, manifest_rows=rows,
                              shuffle=False, flush_every=1, verbose=False)
    assert second["skipped_existing"] == 2
    assert second["constructed"] == 2
    ids = [row["construct_id"] for row in load_constructed_manifest(out_dir)]
    assert len(ids) == len(set(ids)) == 6


def test_appending_widens_a_narrow_header_instead_of_misaligning_rows(tmp_path):
    """Appending under assumed column names is what corrupted the prediction log.

    A manifest whose header predates a column is rewritten under the union before
    anything is appended, so every row stays readable by name.
    """
    out_dir = str(tmp_path)
    path = os.path.join(out_dir, "constructed_manifest.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("construct_id,configuration\nc000001_opposite_left,opposite\n")

    append_constructed_manifest(
        [{"construct_id": "c000002_opposite_left", "configuration": CONFIG_OPPOSITE,
          "base_scene_id": "2", "x_source": 100.0}], out_dir)
    rows = {row["construct_id"]: row for row in load_constructed_manifest(out_dir)}
    assert rows["c000001_opposite_left"]["configuration"] == CONFIG_OPPOSITE
    assert rows["c000001_opposite_left"]["base_scene_id"] == ""
    assert rows["c000002_opposite_left"]["base_scene_id"] == "2"


def test_construction_samples_the_pool_rather_than_taking_a_prefix(tmp_path,
                                                                  monkeypatch):
    """Bridge episodes are ordered by scene and task.

    A truncated run over a prefix of the manifest would cover a few domains and
    would not represent the pool it was drawn from, so candidates are shuffled
    under the seed before the limit applies.
    """
    centers = [100] * 6
    cache_dir, rows = _cache(tmp_path, centers)
    _stub_perception(monkeypatch, centers)

    def bases(out_dir, **kwargs):
        run_construction(cache_dir, out_dir, manifest_rows=rows, limit=2,
                         verbose=False, **kwargs)
        return {row["base_scene_id"] for row in load_constructed_manifest(out_dir)}

    ordered = bases(str(tmp_path / "prefix"), shuffle=False)
    sampled = bases(str(tmp_path / "sampled"), shuffle=True, seed=0)
    again = bases(str(tmp_path / "again"), shuffle=True, seed=0)
    assert ordered == {"0", "1"}
    assert sampled != ordered
    assert sampled == again


# --------------------------------------------------------------------------- #
# Approval screen and the frozen experimental set
# --------------------------------------------------------------------------- #
def _pool(n_left: int = 4, n_right: int = 4, opposite_only: int = 0):
    """A built set: `n_left` frames with a left-hand object, `n_right` right-hand.

    Each frame carries its opposite scene and its one achievable same-side scene,
    which is what a base frame actually yields. `opposite_only` frames carry the
    opposite scene alone, standing for a frame whose same-side placement did not
    fit.
    """
    scenes = []
    base = 0
    for count, side in ((n_left, CONFIG_SAME_LEFT), (n_right, CONFIG_SAME_RIGHT)):
        for _ in range(count):
            for configuration in (CONFIG_OPPOSITE, side):
                scenes.append({
                    "construct_id": f"c{base:06d}_{configuration}_left",
                    "base_scene_id": str(base),
                    "configuration": configuration,
                    "instr_a": "pick up the cup on the left",
                    "gripper_source": "detected",
                    "cutout_mode": "mask",
                })
            base += 1
    for _ in range(opposite_only):
        scenes.append({
            "construct_id": f"c{base:06d}_{CONFIG_OPPOSITE}_left",
            "base_scene_id": str(base),
            "configuration": CONFIG_OPPOSITE,
            "instr_a": "pick up the cup on the left",
            "gripper_source": "detected",
            "cutout_mode": "mask",
        })
        base += 1
    return scenes


def _approve_all(rows):
    for row in rows:
        approve(rows, row["construct_id"], decision=DECISION_APPROVED)
    return rows


def test_review_rows_cover_every_scene_not_a_sample():
    """The screen decides inclusion per scene, so it is a census.

    Geometry agreement is a separate measurement on the eligible pool.
    """
    scenes = _pool()
    rows = build_review_rows(scenes)
    assert len(rows) == len(scenes)
    assert {r["construct_id"] for r in rows} == {s["construct_id"] for s in scenes}
    assert all(r["decision"] == DECISION_UNLABELLED for r in rows)


def test_review_preserves_decisions_when_the_set_grows(tmp_path):
    """Construction continues across sessions, so the file grows under the labels."""
    out_dir = str(tmp_path)
    first = _pool(n_left=1, n_right=1)
    rows = sync_review(first, out_dir)
    approve(rows, first[0]["construct_id"], decision=DECISION_REJECTED,
            reason=REJECT_PASTE_IMPLAUSIBLE, notes="seam visible")
    write_review(rows, out_dir)

    grown = sync_review(first + _pool(n_left=1, n_right=0)[:2], out_dir)
    kept = {r["construct_id"]: r for r in grown}[first[0]["construct_id"]]
    assert kept["decision"] == DECISION_REJECTED
    assert kept["reject_reason"] == REJECT_PASTE_IMPLAUSIBLE
    assert kept["notes"] == "seam visible"
    assert len(grown) > len(first)


def test_rejection_requires_a_stated_reason():
    """A scene excluded for no stated cause cannot be reported as a funnel."""
    rows = build_review_rows(_pool(n_left=1, n_right=0))
    with pytest.raises(ValueError, match="reason"):
        approve(rows, rows[0]["construct_id"], decision=DECISION_REJECTED)
    with pytest.raises(ValueError, match="reason"):
        approve(rows, rows[0]["construct_id"], decision=DECISION_REJECTED,
                reason="looks odd")
    with pytest.raises(ValueError, match="decision"):
        approve(rows, rows[0]["construct_id"], decision="maybe")


def test_approval_summary_reports_rates_per_configuration():
    """A screen that admits the arrangements at different rates threatens the
    comparison they are supposed to support, so the rates are never pooled."""
    scenes = _pool(n_left=2, n_right=2)
    rows = _approve_all(build_review_rows(scenes))
    for row in rows:
        if row["configuration"] == CONFIG_SAME_LEFT:
            approve(rows, row["construct_id"], decision=DECISION_REJECTED,
                    reason=REJECT_NO_SECOND_INSTANCE)

    summary = approval_summary(rows, targets={CONFIG_SAME_LEFT: 2})
    assert summary["reviewed"] == len(rows)
    assert summary["by_configuration"][CONFIG_OPPOSITE]["approval_rate"] == 1.0
    assert summary["by_configuration"][CONFIG_SAME_LEFT]["approval_rate"] == 0.0
    assert summary["by_configuration"][CONFIG_SAME_LEFT]["still_needed"] == 2
    assert summary["by_reject_reason"] == {REJECT_NO_SECOND_INSTANCE: 2}
    assert summary["unlabelled"] == 0


def test_approval_summary_counts_an_unreviewed_scene_as_neither():
    rows = build_review_rows(_pool(n_left=1, n_right=1))
    approve(rows, rows[0]["construct_id"], decision=DECISION_APPROVED)
    summary = approval_summary(rows)
    assert summary["approved"] == 1
    assert summary["rejected"] == 0
    assert summary["unlabelled"] == len(rows) - 1
    assert summary["approval_rate"] == pytest.approx(1.0)


def test_eligibility_excludes_a_frame_that_yielded_one_arrangement():
    """A scene without a sibling cannot enter the set, so screening it buys nothing."""
    scenes = _pool(n_left=2, n_right=1, opposite_only=3)
    eligible = pair_eligible_ids(scenes)
    assert len(eligible) == 6            # three paired frames, two scenes each
    unpaired = {s["construct_id"] for s in scenes} - eligible
    assert len(unpaired) == 3
    assert all(s["configuration"] == CONFIG_OPPOSITE for s in scenes
               if s["construct_id"] in unpaired)


def test_the_projection_measures_frames_rather_than_scenes():
    """Approval compounds within a frame, so the scene rate overstates the yield.

    Rejecting one member withdraws the frame, which is what the targets consume.
    """
    scenes = _pool(n_left=2, n_right=2)
    rows = _approve_all(build_review_rows(scenes))
    approve(rows, "c000000_opposite_left", decision=DECISION_REJECTED,
            reason=REJECT_PASTE_IMPLAUSIBLE)

    projection = project_yield(scenes, rows,
                               targets={CONFIG_OPPOSITE: 8, CONFIG_SAME_LEFT: 4,
                                        CONFIG_SAME_RIGHT: 4})
    assert projection["base_frames"] == 4
    assert projection["paired_frames"] == 4
    assert projection["decided_pairs"] == 4
    assert projection["approved_pairs"] == 3
    assert projection["joint_approval_rate"] == pytest.approx(0.75)
    assert projection["approved_by_side"] == {CONFIG_SAME_LEFT: 1,
                                             CONFIG_SAME_RIGHT: 2}
    # Eight paired frames wanted, three held: five more must survive the screen,
    # so seven pairs have to be built at the measured rate.
    assert projection["paired_frames_short"] == 5
    assert projection["pairs_to_build"] == 7
    assert projection["scenes_to_screen"] == 14


def test_the_projection_sizes_a_harvest_from_the_candidate_pool():
    """The manifest holds frames that produced a scene, not those attempted.

    Without the attempted count the requirement can only be stated in frames that
    yield scenes, which is not the number a harvest is sized by.
    """
    scenes = _pool(n_left=2, n_right=2, opposite_only=4)
    rows = _approve_all(build_review_rows(scenes))
    targets = {CONFIG_OPPOSITE: 16, CONFIG_SAME_LEFT: 8, CONFIG_SAME_RIGHT: 8}

    blind = project_yield(scenes, rows, targets=targets)
    assert blind["candidates_to_process"] is None
    assert blind["frames_to_yield_scenes"] == 24   # 12 short at a pair rate of 0.5

    sized = project_yield(scenes, rows, targets=targets, candidates_processed=80)
    assert sized["candidates_to_process"] == 240   # 4 pairs from 80 attempts
    assert sized["candidates_to_process"] > sized["frames_to_yield_scenes"]


def test_the_projection_counts_a_frame_with_an_unseen_scene_as_undecided():
    """A rate taken over half-screened frames would move as screening continued."""
    scenes = _pool(n_left=1, n_right=1)
    rows = build_review_rows(scenes)
    for row in rows[:2]:                  # one frame decided, the other untouched
        approve(rows, row["construct_id"], decision=DECISION_APPROVED)

    projection = project_yield(scenes, rows, targets=DEFAULT_TARGETS)
    assert projection["paired_frames"] == 2
    assert projection["decided_pairs"] == 1
    assert projection["approved_pairs"] == 1
    assert projection["joint_approval_rate"] == pytest.approx(1.0)


def test_an_unmeasurable_requirement_is_left_unstated():
    """A projection from a rate of zero would report a finite, invented number."""
    scenes = _pool(n_left=1, n_right=1)
    rows = build_review_rows(scenes)
    for row in rows:
        approve(rows, row["construct_id"], decision=DECISION_REJECTED,
                reason=REJECT_PASTE_IMPLAUSIBLE)

    projection = project_yield(scenes, rows, targets=DEFAULT_TARGETS,
                               candidates_processed=10)
    assert projection["approved_pairs"] == 0
    assert projection["joint_approval_rate"] == pytest.approx(0.0)
    assert projection["pairs_to_build"] is None
    assert projection["scenes_to_screen"] is None
    assert projection["candidates_to_process"] is None


def test_a_filled_target_asks_for_nothing_further():
    scenes = _pool(n_left=2, n_right=2)
    rows = _approve_all(build_review_rows(scenes))
    projection = project_yield(scenes, rows,
                               targets={CONFIG_OPPOSITE: 4, CONFIG_SAME_LEFT: 2,
                                        CONFIG_SAME_RIGHT: 2})
    assert projection["paired_frames_short"] == 0
    assert projection["pairs_to_build"] == 0
    assert projection["candidates_to_process"] == 0


def test_selection_pairs_each_same_side_scene_with_its_own_opposite():
    """The contrast is within base frame, which is what makes it interpretable.

    Both members come from one frame, so the background, the object, the cutout,
    and the instruction are identical across the arrangement change and scene
    identity cannot explain a difference.
    """
    scenes = _pool(n_left=4, n_right=4)
    rows = _approve_all(build_review_rows(scenes))
    targets = {CONFIG_OPPOSITE: 4, CONFIG_SAME_LEFT: 2, CONFIG_SAME_RIGHT: 2}

    result = select_evaluation_set(scenes, approved_ids(rows), targets=targets, seed=0)
    assert result["counts"] == targets
    assert result["complete"]
    assert result["shortfall"] == {}

    by_base: dict = {}
    for row in result["rows"]:
        by_base.setdefault(row["base_scene_id"], set()).add(row["configuration"])
    assert len(by_base) == 4
    for configurations in by_base.values():
        assert CONFIG_OPPOSITE in configurations
        assert len(configurations) == 2
    assert all(row["paired"] == "yes" for row in result["rows"])


def test_selection_needs_both_scenes_of_a_frame_approved():
    """Rejecting one scene of a frame withdraws the frame, not just that scene.

    Keeping the survivor would put an unpaired scene in the set and break the
    within-frame comparison the pairing exists to provide.
    """
    scenes = _pool(n_left=2, n_right=2)
    rows = _approve_all(build_review_rows(scenes))
    # The opposite scene of the first left-hand frame fails the screen.
    approve(rows, "c000000_opposite_left", decision=DECISION_REJECTED,
            reason=REJECT_PASTE_IMPLAUSIBLE)

    result = select_evaluation_set(
        scenes, approved_ids(rows),
        targets={CONFIG_OPPOSITE: 2, CONFIG_SAME_LEFT: 2, CONFIG_SAME_RIGHT: 1},
        seed=0)
    assert "0" not in {row["base_scene_id"] for row in result["rows"]}
    assert result["shortfall"][CONFIG_SAME_LEFT] == 1


def test_selection_reports_a_shortfall_rather_than_padding_it():
    """The scarcer side is the binding constraint and is reported as such.

    Padding the cell from unpaired scenes would change what the comparison
    controls for while leaving the count looking complete.
    """
    scenes = _pool(n_left=1, n_right=4, opposite_only=6)
    rows = _approve_all(build_review_rows(scenes))
    targets = {CONFIG_OPPOSITE: 6, CONFIG_SAME_LEFT: 3, CONFIG_SAME_RIGHT: 3}

    result = select_evaluation_set(scenes, approved_ids(rows), targets=targets, seed=0)
    assert result["shortfall"][CONFIG_SAME_LEFT] == 2
    assert not result["complete"]
    # The opposite cell follows the frames the same-side cells could fill, so it
    # is short by the same amount rather than made up from spare frames.
    assert result["counts"][CONFIG_OPPOSITE] == 4
    assert result["shortfall"][CONFIG_OPPOSITE] == 2


def test_selection_excludes_scenes_that_were_never_reviewed():
    scenes = _pool(n_left=2, n_right=2)
    rows = build_review_rows(scenes)
    approve(rows, "c000000_same_side_left_left", decision=DECISION_APPROVED)
    approve(rows, "c000000_opposite_left", decision=DECISION_APPROVED)
    result = select_evaluation_set(
        scenes, approved_ids(rows),
        targets={CONFIG_OPPOSITE: 2, CONFIG_SAME_LEFT: 2, CONFIG_SAME_RIGHT: 2},
        seed=0)
    assert {row["base_scene_id"] for row in result["rows"]} == {"0"}


def test_selection_is_reproducible_and_seed_dependent():
    scenes = _pool(n_left=6, n_right=6)
    approved = approved_ids(_approve_all(build_review_rows(scenes)))
    targets = {CONFIG_OPPOSITE: 4, CONFIG_SAME_LEFT: 2, CONFIG_SAME_RIGHT: 2}

    first = select_evaluation_set(scenes, approved, targets=targets, seed=0)
    again = select_evaluation_set(scenes, approved, targets=targets, seed=0)
    other = select_evaluation_set(scenes, approved, targets=targets, seed=7)
    ids = lambda result: [row["construct_id"] for row in result["rows"]]
    assert ids(first) == ids(again)
    assert ids(first) != ids(other)


def test_frozen_set_survives_a_round_trip_and_carries_the_pairing(tmp_path):
    out_dir = str(tmp_path)
    scenes = _pool(n_left=2, n_right=2)
    write_constructed_manifest(scenes, out_dir)
    result = select_evaluation_set(
        scenes, approved_ids(_approve_all(build_review_rows(scenes))),
        targets={CONFIG_OPPOSITE: 2, CONFIG_SAME_LEFT: 1, CONFIG_SAME_RIGHT: 1},
        seed=0)
    write_evaluation_set(result["rows"], out_dir)

    frozen = load_evaluation_set(out_dir)
    assert [r["construct_id"] for r in frozen] == [
        r["construct_id"] for r in result["rows"]]

    resolved = evaluation_scenes(out_dir)
    assert len(resolved) == len(frozen)
    assert all(row["paired"] == "yes" for row in resolved)
    # Resolved rows carry the full geometry, which is what the probe reads.
    assert all("configuration" in row for row in resolved)


def test_no_frozen_set_resolves_to_nothing_rather_than_everything(tmp_path):
    """A missing freeze must not silently fall back to the whole manifest."""
    out_dir = str(tmp_path)
    write_constructed_manifest(_pool(n_left=1, n_right=1), out_dir)
    assert evaluation_scenes(out_dir) == []


# --------------------------------------------------------------------------- #
# The agreement pool
# --------------------------------------------------------------------------- #
def test_agreement_pool_is_the_frozen_set_when_one_exists(tmp_path):
    out_dir = str(tmp_path)
    scenes = _pool(n_left=2, n_right=2)
    write_constructed_manifest(scenes, out_dir)
    result = select_evaluation_set(
        scenes, approved_ids(_approve_all(build_review_rows(scenes))),
        targets={CONFIG_OPPOSITE: 2, CONFIG_SAME_LEFT: 1, CONFIG_SAME_RIGHT: 1},
        seed=0)
    write_evaluation_set(result["rows"], out_dir)

    rows, source = agreement_pool(out_dir, scenes)
    assert source == POOL_FROZEN
    assert {r["construct_id"] for r in rows} == {
        r["construct_id"] for r in result["rows"]}


def test_agreement_pool_excludes_rejected_and_unscreened_scenes(tmp_path):
    """Agreement is a claim about the stimuli the probe runs on.

    A rejection cites faults that also make the arrangement harder to read, so
    admitting rejections would understate agreement on the set actually used, and
    an unscreened scene is of unknown quality.
    """
    out_dir = str(tmp_path)
    scenes = _pool(n_left=1, n_right=1)
    write_constructed_manifest(scenes, out_dir)
    rows = build_review_rows(scenes)
    approve(rows, scenes[0]["construct_id"], decision=DECISION_APPROVED)
    approve(rows, scenes[1]["construct_id"], decision=DECISION_REJECTED,
            reason=REJECT_PASTE_IMPLAUSIBLE)
    write_review(rows, out_dir)

    pool, source = agreement_pool(out_dir, scenes)
    assert source == POOL_APPROVED
    assert [r["construct_id"] for r in pool] == [scenes[0]["construct_id"]]


def test_agreement_pool_never_falls_back_to_the_whole_build(tmp_path):
    """The defect this guards against drew the sample from every scene built.

    An earlier version resolved the pool at the call site and substituted the
    constructed manifest when no frozen set existed, so the agreement number was
    measured over rejected and unscreened scenes.
    """
    out_dir = str(tmp_path)
    scenes = _pool(n_left=2, n_right=2)
    write_constructed_manifest(scenes, out_dir)

    pool, source = agreement_pool(out_dir, scenes)
    assert source == POOL_APPROVED
    assert pool == []
    assert draw_validation_sample(pool, n=50, seed=0) == []


# --------------------------------------------------------------------------- #
# Manipulation check
# --------------------------------------------------------------------------- #
def _manipulation_inputs(toward_pasted: bool):
    scenes, predictions = [], []
    for i in range(10):
        construct_id = f"c{i:06d}_opposite_left"
        scenes.append({"construct_id": construct_id, "x_source": 100.0,
                       "x_pasted": 220.0, "x_gripper": 160.0,
                       "configuration": CONFIG_OPPOSITE})
        # The source sits left of the start position and the duplicate right,
        # so the sign of the lateral action says which one was reached for.
        # Under the identified convention a rightward target reads negative.
        value = 0.01 * IMAGE_X_TO_LATERAL_SIGN if toward_pasted \
            else -0.01 * IMAGE_X_TO_LATERAL_SIGN
        predictions.append({"construct_id": construct_id, "c1": value,
                            "configuration": CONFIG_OPPOSITE})
    return predictions, scenes


def test_manipulation_rate_detects_reaching_for_the_duplicate():
    predictions, scenes = _manipulation_inputs(toward_pasted=True)
    result = manipulation_rate(predictions, scenes)
    assert result["n"] == 10
    assert result["rate"] == pytest.approx(1.0)


def test_manipulation_rate_is_at_floor_when_the_paste_is_ignored():
    predictions, scenes = _manipulation_inputs(toward_pasted=False)
    assert manipulation_rate(predictions, scenes)["rate"] == pytest.approx(0.0)


def test_manipulation_rate_skips_scenes_that_cannot_distinguish():
    # Both instances on the same side: one action is consistent with reaching
    # for either, so the scene carries no information here.
    scenes = [{"construct_id": "c1", "x_source": 100.0, "x_pasted": 60.0,
               "x_gripper": 160.0, "configuration": CONFIG_SAME_LEFT}]
    predictions = [{"construct_id": "c1", "c1": -0.01,
                    "configuration": CONFIG_SAME_LEFT}]
    assert manipulation_rate(predictions, scenes)["n"] == 0


def test_manipulation_rate_handles_empty_inputs():
    assert manipulation_rate([], [])["n"] == 0
