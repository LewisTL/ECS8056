"""
tests/test_model_readout.py: unit tests for the OpenVLA action-bin decoding and
the continuous readout in action_bins.py.

The tests are written against the arithmetic in
`OpenVLAForActionPrediction.predict_action`, reproduced here as
`_upstream_decode`, so a change in either implementation shows up as a
disagreement rather than as two consistent but wrong answers.

Synthetic logits only: no torch, no model download, and no GPU. The complementary
check that the reimplementation matches the real model on real inputs is
`model.verify_readout`, which the probe notebook runs as a gate before any
continuous value is used.
"""

from __future__ import annotations

import numpy as np
import pytest

from action_bins import (
    ActionReadout,
    action_token_ids,
    bin_distribution,
    bin_entropy,
    bin_widths,
    decode_bin_indices,
    denormalise_action,
    readout_from_logits,
    softmax,
)

# Matches the openvla-7b configuration: 256 bin edges give 255 centres.
N_ACTION_BINS = 256
BINS = np.linspace(-1, 1, N_ACTION_BINS)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0
N_BINS = BIN_CENTERS.shape[0]
VOCAB_SIZE = 32000
PADDED_VOCAB = 32064

# Bridge-like statistics: six masked translation and rotation dimensions and an
# unmasked gripper dimension, which de-normalisation passes through unchanged.
STATS = {
    "q01": [-0.14] * 6 + [0.0],
    "q99": [0.14] * 6 + [1.0],
    "mask": [True] * 6 + [False],
}


def _upstream_decode(token_ids):
    """The de-tokenisation performed inside OpenVLA's predict_action."""
    discretized = VOCAB_SIZE - np.asarray(token_ids)
    discretized = np.clip(discretized - 1, a_min=0, a_max=BIN_CENTERS.shape[0] - 1)
    normalized = BIN_CENTERS[discretized]
    low, high = np.array(STATS["q01"]), np.array(STATS["q99"])
    return np.where(STATS["mask"], 0.5 * (normalized + 1) * (high - low) + low,
                    normalized)


def _one_hot_logits(bin_indices, peak=60.0):
    """Logits placing essentially all mass on one action token per dimension."""
    logits = np.full((len(bin_indices), PADDED_VOCAB), -1e9)
    for dimension, index in enumerate(bin_indices):
        logits[dimension, VOCAB_SIZE - 1 - index] = peak
    return logits


# --------------------------------------------------------------------------- #
# Token layout
# --------------------------------------------------------------------------- #
def test_bin_centers_has_one_fewer_entry_than_the_bin_count():
    # An off-by-one here would shift every decoded action by half a bin.
    assert N_BINS == N_ACTION_BINS - 1 == 255


def test_token_ids_round_trip_to_their_bin_indices():
    ids = action_token_ids(VOCAB_SIZE, N_BINS)
    assert np.array_equal(decode_bin_indices(ids, VOCAB_SIZE, N_BINS),
                          np.arange(N_BINS))


def test_action_tokens_sit_at_the_tail_of_the_vocabulary():
    ids = action_token_ids(VOCAB_SIZE, N_BINS)
    assert ids.max() == VOCAB_SIZE - 1
    assert ids.min() == VOCAB_SIZE - N_BINS
    assert len(set(ids.tolist())) == N_BINS


def test_decode_clips_exactly_as_upstream_does():
    for token in (VOCAB_SIZE + 5, VOCAB_SIZE, VOCAB_SIZE - 1, VOCAB_SIZE - 128,
                  VOCAB_SIZE - N_BINS, VOCAB_SIZE - N_BINS - 50):
        decoded = decode_bin_indices([token], VOCAB_SIZE, N_BINS)[0]
        expected = np.clip(VOCAB_SIZE - token - 1, 0, N_BINS - 1)
        assert decoded == expected


# --------------------------------------------------------------------------- #
# De-normalisation
# --------------------------------------------------------------------------- #
def test_denormalise_matches_upstream():
    rng = np.random.default_rng(0)
    normalized = rng.uniform(-1, 1, size=7)
    low, high = np.array(STATS["q01"]), np.array(STATS["q99"])
    expected = np.where(STATS["mask"],
                        0.5 * (normalized + 1) * (high - low) + low, normalized)
    assert np.allclose(denormalise_action(normalized, STATS), expected)


def test_denormalise_maps_the_bin_range_onto_the_quantile_range():
    assert np.allclose(denormalise_action([-1.0] * 7, STATS)[:6], [-0.14] * 6)
    assert np.allclose(denormalise_action([1.0] * 7, STATS)[:6], [0.14] * 6)


def test_denormalise_leaves_unmasked_dimensions_alone():
    assert denormalise_action([0.0] * 7, STATS)[6] == pytest.approx(0.0)
    assert denormalise_action([0.5] * 7, STATS)[6] == pytest.approx(0.5)


def test_denormalise_defaults_to_masking_every_dimension():
    stats = {"q01": [-1.0, -2.0], "q99": [1.0, 2.0]}
    assert np.allclose(denormalise_action([0.0, 0.0], stats), [0.0, 0.0])
    assert np.allclose(denormalise_action([1.0, 1.0], stats), [1.0, 2.0])


def test_denormalisation_commutes_with_expectation():
    # The whole continuous readout rests on this: taking the expected bin centre
    # and then de-normalising must equal de-normalising every centre and then
    # taking the expectation. It holds because the transform is affine.
    rng = np.random.default_rng(1)
    probs, _ = bin_distribution(rng.normal(size=(7, PADDED_VOCAB)), VOCAB_SIZE, N_BINS)
    direct = denormalise_action(probs @ BIN_CENTERS, STATS)
    per_bin = np.stack([denormalise_action(np.full(7, c), STATS) for c in BIN_CENTERS])
    elementwise = (probs * per_bin.T).sum(axis=1)
    assert np.allclose(direct, elementwise)


def test_bin_width_is_the_quantile_span_over_the_intervals():
    widths = bin_widths(STATS, N_BINS)
    assert widths[0] == pytest.approx(0.28 / (N_BINS - 1))
    assert widths[6] == pytest.approx(1.0 / (N_BINS - 1))


# --------------------------------------------------------------------------- #
# Distribution over bins
# --------------------------------------------------------------------------- #
def test_softmax_is_normalised_and_stable_at_extreme_magnitudes():
    probs = softmax(np.array([[1000.0, 1000.0, 999.0], [-1000.0, -1000.0, -1001.0]]))
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert np.isfinite(probs).all()


def test_bin_distribution_is_normalised_over_the_action_slice():
    rng = np.random.default_rng(2)
    probs, mass = bin_distribution(rng.normal(size=(7, PADDED_VOCAB)),
                                   VOCAB_SIZE, N_BINS)
    assert probs.shape == (7, N_BINS)
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert np.all((mass > 0) & (mass < 1))


def test_action_mass_reaches_one_when_all_weight_is_on_action_tokens():
    _, mass = bin_distribution(_one_hot_logits([10] * 7), VOCAB_SIZE, N_BINS)
    assert np.allclose(mass, 1.0)


def test_action_mass_reports_weight_leaking_outside_the_slice():
    logits = _one_hot_logits([10] * 7)
    logits[:, 0] = 60.0  # equal weight on a non-action token
    _, mass = bin_distribution(logits, VOCAB_SIZE, N_BINS)
    assert np.allclose(mass, 0.5, atol=1e-6)


def test_bin_distribution_does_not_produce_nan_on_an_empty_slice():
    logits = np.full((2, PADDED_VOCAB), -np.inf)
    logits[:, 0] = 1.0
    probs, mass = bin_distribution(logits, VOCAB_SIZE, N_BINS)
    assert np.isfinite(probs).all()
    assert np.allclose(mass, 0.0)


def test_entropy_is_zero_when_certain_and_maximal_when_uniform():
    probs, _ = bin_distribution(_one_hot_logits([100] * 3), VOCAB_SIZE, N_BINS)
    assert np.allclose(bin_entropy(probs), 0.0, atol=1e-9)
    uniform = np.full((1, N_BINS), 1.0 / N_BINS)
    assert bin_entropy(uniform)[0] == pytest.approx(np.log(N_BINS))


# --------------------------------------------------------------------------- #
# Full readout
# --------------------------------------------------------------------------- #
def test_argmax_path_reproduces_upstream_decoding():
    rng = np.random.default_rng(3)
    indices = rng.integers(0, N_BINS, size=7)
    tokens = [VOCAB_SIZE - 1 - int(i) for i in indices]
    readout = readout_from_logits(tokens, _one_hot_logits(indices), BIN_CENTERS,
                                  VOCAB_SIZE, STATS)
    assert np.allclose(readout.action, _upstream_decode(tokens), atol=1e-6)
    assert np.array_equal(readout.bin_index, indices)


def test_expected_value_equals_the_argmax_when_the_model_is_certain():
    indices = [3, 100, 200, 254, 0, 128, 64]
    tokens = [VOCAB_SIZE - 1 - i for i in indices]
    readout = readout_from_logits(tokens, _one_hot_logits(indices), BIN_CENTERS,
                                  VOCAB_SIZE, STATS)
    assert np.allclose(readout.expected, readout.action, atol=1e-6)
    assert np.allclose(readout.top_prob, 1.0)


def test_expected_value_lies_between_the_bins_carrying_the_mass():
    # Two adjacent bins with equal weight: the expectation must fall between
    # them, which is the sub-bin resolution the continuous readout exists for.
    logits = np.full((1, PADDED_VOCAB), -1e9)
    logits[0, VOCAB_SIZE - 1 - 100] = 10.0
    logits[0, VOCAB_SIZE - 1 - 101] = 10.0
    stats = {"q01": [-1.0], "q99": [1.0], "mask": [True]}
    readout = readout_from_logits([VOCAB_SIZE - 1 - 100], logits, BIN_CENTERS,
                                  VOCAB_SIZE, stats)
    assert BIN_CENTERS[100] < readout.expected[0] < BIN_CENTERS[101]
    assert readout.action[0] == pytest.approx(BIN_CENTERS[100], abs=1e-6)


def test_expected_value_resolves_a_difference_the_argmax_cannot():
    # Two distributions with the same argmax but different weight on the
    # neighbouring bin decode to the same action and to different expectations.
    stats = {"q01": [-1.0], "q99": [1.0], "mask": [True]}
    readouts = []
    for neighbour_logit in (0.0, 5.0):
        logits = np.full((1, PADDED_VOCAB), -1e9)
        logits[0, VOCAB_SIZE - 1 - 128] = 6.0
        logits[0, VOCAB_SIZE - 1 - 129] = neighbour_logit
        readouts.append(readout_from_logits([VOCAB_SIZE - 1 - 128], logits,
                                            BIN_CENTERS, VOCAB_SIZE, stats))
    assert readouts[0].action[0] == pytest.approx(readouts[1].action[0])
    assert readouts[0].expected[0] != pytest.approx(readouts[1].expected[0])


def test_readout_reports_confidence_and_entropy_per_dimension():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(7, PADDED_VOCAB))
    tokens = [int(np.argmax(logits[d])) for d in range(7)]
    readout = readout_from_logits(tokens, logits, BIN_CENTERS, VOCAB_SIZE, STATS)
    assert readout.top_prob.shape == (7,)
    assert np.all((readout.top_prob > 0) & (readout.top_prob <= 1))
    assert np.all(readout.entropy >= 0)
    assert np.all(readout.entropy <= np.log(N_BINS) + 1e-9)


def test_readout_is_deterministic():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(7, PADDED_VOCAB))
    tokens = [VOCAB_SIZE - 1 - i for i in range(7)]
    first = readout_from_logits(tokens, logits, BIN_CENTERS, VOCAB_SIZE, STATS)
    second = readout_from_logits(tokens, logits, BIN_CENTERS, VOCAB_SIZE, STATS)
    assert np.array_equal(first.expected, second.expected)


# --------------------------------------------------------------------------- #
# Log fields
# --------------------------------------------------------------------------- #
def test_log_fields_cover_every_dimension_with_the_expected_names():
    readout = readout_from_logits([VOCAB_SIZE - 1 - i for i in range(7)],
                                  _one_hot_logits(list(range(7))), BIN_CENTERS,
                                  VOCAB_SIZE, STATS)
    fields = readout.to_log_fields()
    for prefix in ("c", "b", "p", "h"):
        assert [f"{prefix}{i}" for i in range(7)] == [
            k for k in fields if k.startswith(prefix) and k[1:].isdigit()]
    assert fields["action_mass_min"] == pytest.approx(1.0)
    assert all(isinstance(fields[f"b{i}"], int) for i in range(7))


def test_log_fields_are_json_safe_scalars():
    readout = readout_from_logits([VOCAB_SIZE - 1], _one_hot_logits([0]),
                                  BIN_CENTERS, VOCAB_SIZE,
                                  {"q01": [-1.0], "q99": [1.0], "mask": [True]})
    assert all(isinstance(v, (int, float)) for v in readout.to_log_fields().values())


def test_readout_is_a_dataclass_with_the_documented_fields():
    readout = readout_from_logits([VOCAB_SIZE - 1], _one_hot_logits([0]),
                                  BIN_CENTERS, VOCAB_SIZE,
                                  {"q01": [-1.0], "q99": [1.0], "mask": [True]})
    assert isinstance(readout, ActionReadout)
    for name in ("action", "expected", "bin_index", "top_prob", "entropy",
                 "action_mass"):
        assert getattr(readout, name).shape == (1,)
