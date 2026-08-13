"""
action_bins.py: OpenVLA action-token decoding and the continuous readout.

OpenVLA emits one token per action dimension, drawn from the tail of the LLM
vocabulary. Decoding takes the argmax token and looks up a bin centre, so the
executable action is quantised to the bin grid: on BridgeData V2 the lateral bin
width is roughly 1e-3, while observed lateral predictions fall between 1e-4 and
4e-3 of zero. A directional difference smaller than one bin is therefore
invisible to the argmax, and two instructions that shift the distribution
without crossing a bin boundary decode to identical actions.

The continuous readout replaces the argmax with the expected bin centre under
the model's own distribution over action tokens. It preserves the sign semantics
of the argmax while resolving sub-bin differences, so a paired comparison is no
longer floored by quantisation.

Token layout, confirmed against `OpenVLAForActionPrediction.predict_action`:
discretisation computes `vocab_size - token_id` and then clips
`discretised - 1` to `[0, n_bins - 1]`. Inverting that, bin index `i` is emitted
by token id `vocab_size - 1 - i`. Note that `bin_centers` holds
`n_action_bins - 1` entries, since it is the midpoint sequence of
`np.linspace(-1, 1, n_action_bins)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Token the model expects immediately after the prompt colon, matching the
# inputs seen at training time. Upstream `predict_action` appends it when
# absent; the continuous path must do the same or the two readouts diverge.
EMPTY_TOKEN_ID = 29871

# Degrees of freedom in an OpenVLA action: three translation, three rotation, and
# the gripper.
DOF = 7


@dataclass
class ActionReadout:
    """Both readouts for one prediction, plus distribution diagnostics.

    Attributes:
        action: de-normalised argmax action, identical to `predict_action`.
        expected: de-normalised expected-bin action, the continuous readout.
        bin_index: argmax bin index per dimension, on the `bin_centers` grid.
        top_prob: probability of the argmax bin within the action-token slice.
        entropy: entropy of the per-dimension bin distribution, in nats.
        action_mass: probability mass on action tokens before renormalisation.
            Well below one indicates the model is placing weight outside the
            action vocabulary, which invalidates the continuous value.
    """

    action: np.ndarray
    expected: np.ndarray
    bin_index: np.ndarray
    top_prob: np.ndarray
    entropy: np.ndarray
    action_mass: np.ndarray

    def to_log_fields(self) -> dict:
        """Flatten to the per-dimension columns written to the prediction log."""
        fields: dict = {}
        for i, value in enumerate(self.expected):
            fields[f"c{i}"] = float(value)
        for i, value in enumerate(self.bin_index):
            fields[f"b{i}"] = int(value)
        for i, value in enumerate(self.top_prob):
            fields[f"p{i}"] = float(value)
        for i, value in enumerate(self.entropy):
            fields[f"h{i}"] = float(value)
        fields["action_mass_min"] = float(np.min(self.action_mass))
        return fields


def readout_log_fields(n_dims: int = DOF) -> list[str]:
    """Names `to_log_fields` produces, in order, without needing a prediction.

    Declaring the schema apart from the data is what lets a log's full column set
    be known before any row exists, so a file can be created with its final header
    rather than with whichever subset the first row happened to carry.
    """
    return ([f"c{i}" for i in range(n_dims)]
            + [f"b{i}" for i in range(n_dims)]
            + [f"p{i}" for i in range(n_dims)]
            + [f"h{i}" for i in range(n_dims)]
            + ["action_mass_min"])


def action_token_ids(vocab_size: int, n_bins: int) -> np.ndarray:
    """Token ids that decode to bin indices 0 through n_bins - 1, in order.

    Inverse of the de-tokenisation in `OpenVLAForActionPrediction.predict_action`.
    """
    return vocab_size - 1 - np.arange(n_bins)


def decode_bin_indices(token_ids, vocab_size: int, n_bins: int) -> np.ndarray:
    """Map emitted token ids to bin indices, as upstream de-tokenisation does."""
    discretized = vocab_size - np.asarray(token_ids)
    return np.clip(discretized - 1, a_min=0, a_max=n_bins - 1)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def bin_distribution(logits: np.ndarray, vocab_size: int, n_bins: int):
    """Per-dimension distribution over action bins.

    Args:
        logits: (n_dims, vocab) array of next-token scores, one row per action
            dimension in emission order.
        vocab_size: the model's de-tokenisation vocabulary size, which excludes
            the padding tokens added to round the embedding matrix.
        n_bins: number of bin centres.

    Returns (probs, mass): `probs` is (n_dims, n_bins) renormalised over the
    action-token slice, and `mass` is (n_dims,) holding the probability that
    fell on action tokens before renormalisation. Renormalising keeps the
    expected bin well defined even when a little mass escapes the slice; `mass`
    is retained so that escape is visible rather than silently absorbed.
    """
    probs_full = softmax(logits)
    ids = action_token_ids(vocab_size, n_bins)
    sliced = probs_full[..., ids]
    mass = sliced.sum(axis=-1)
    # A zero row would only arise from an all-minus-infinity slice; guard so the
    # division cannot produce NaN and hide the failure downstream.
    safe = np.where(mass > 0, mass, 1.0)
    return sliced / safe[..., None], mass


def bin_entropy(probs: np.ndarray) -> np.ndarray:
    """Entropy in nats of each row of a bin distribution."""
    probs = np.asarray(probs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 0, probs * np.log(probs), 0.0)
    return -terms.sum(axis=-1)


def denormalise_action(normalized, action_norm_stats: dict) -> np.ndarray:
    """Map normalised action values into dataset units.

    Mirrors the de-normalisation in `OpenVLAForActionPrediction.predict_action`.
    The transform is affine on masked dimensions and the identity elsewhere, so
    it commutes with expectation: de-normalising the expected bin centre equals
    the expectation of the de-normalised bin centres. That equivalence is what
    makes the continuous readout comparable with the argmax action.
    """
    normalized = np.asarray(normalized, dtype=np.float64)
    low = np.asarray(action_norm_stats["q01"], dtype=np.float64)
    high = np.asarray(action_norm_stats["q99"], dtype=np.float64)
    mask = np.asarray(action_norm_stats.get("mask", np.ones_like(low, dtype=bool)))
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


def bin_widths(action_norm_stats: dict, n_bins: int) -> np.ndarray:
    """Width of one action bin per dimension, in dataset units.

    This is the resolution floor of the argmax readout: a difference between two
    predictions smaller than this cannot appear in the executable action.
    """
    low = np.asarray(action_norm_stats["q01"], dtype=np.float64)
    high = np.asarray(action_norm_stats["q99"], dtype=np.float64)
    return (high - low) / (n_bins - 1)


def readout_from_logits(
    token_ids,
    logits: np.ndarray,
    bin_centers: np.ndarray,
    vocab_size: int,
    action_norm_stats: dict,
) -> ActionReadout:
    """Assemble an ActionReadout from generated tokens and their logits.

    Kept free of torch so the whole decoding path is testable on synthetic
    logits; `model.predict_action_dist` supplies the real arrays.
    """
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    n_bins = bin_centers.shape[0]
    bin_index = decode_bin_indices(token_ids, vocab_size, n_bins)
    probs, mass = bin_distribution(logits, vocab_size, n_bins)
    return ActionReadout(
        action=denormalise_action(bin_centers[bin_index], action_norm_stats).astype(np.float32),
        expected=denormalise_action(probs @ bin_centers, action_norm_stats).astype(np.float32),
        bin_index=bin_index.astype(int),
        top_prob=probs.max(axis=-1).astype(np.float32),
        entropy=bin_entropy(probs).astype(np.float32),
        action_mass=mass.astype(np.float32),
    )
