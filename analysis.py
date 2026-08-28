"""
analysis.py: statistics for the contrastive probe and its control conditions.

The measurement is a set of paired comparisons on one translation component,
chosen per scene from the spatial term (see `data.TERM_AXIS`). Each comparison
answers a different question, and the claim about spatial grounding rests on the
pattern across them rather than on any single test:

  * `mirror_check` asks whether the lateral output moves at all when the scene is
    reflected. This is object grounding, the necessary condition, and it is what
    makes a null on the language tests interpretable rather than ambiguous.
  * `lexical_check` asks whether the term contrast survives when the instruction
    is paired with a different scene. A difference that survives is a property of
    the language alone.
  * `term_effect` measures the term's contribution against the model's own
    behaviour on the same image with the term removed.
  * `congruence_test` asks whether the term's effect tracks the recorded geometry
    or a fixed word-to-direction mapping, which is the decisive contrast.
    * `object_tracking_report` asks whether the first predicted action moves toward
    a unique named object and reverses when that object's image position reverses.
    That is the instrument check the language tests rest on: a first-step miss
    toward a named referent is evidence of a spatial-language failure only when
    this check has already shown that the first action is an object-directed
    readout. `object_tracking_by_axis` repeats the scoring on dx, dy, and dz, so
    a fail on dx can be read as a dead visual channel or as the image-x signal
    sitting on another translation component.

Four conventions apply throughout.

Statistics are computed on the continuous expected-bin readout (`c*` columns)
where available, because the argmax action is quantised at roughly the size of the
effect. A null is reported with an equivalence test rather than a non-significant
p value, since the claim in the null case is that the effect is negligible, not
that it is unproven.

The recorded geometry reaches these functions in image coordinates, and the
convention relating image position to the sign of the lateral action is applied
here through `lateral_sign`. Holding it outside the prediction log means a
revision costs a re-read rather than a repeat of every prediction, and it keeps
the one unresolved link in the chain visible at the point it is used.

Any statistic that reads the sign of a prediction accepts `min_magnitude`. Below
one action bin a sign is not a decision the model could execute, and counting
sub-bin values as choices pulls every sign-based rate toward one half, which
attenuates exactly the contrast the constructed scenes were built to expose.

Helpers are shared by the analysis and plotting notebooks so both report
identical numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Prefer the continuous readout; fall back to the argmax action for rows logged
# before the continuous columns existed.
CONTINUOUS_COLS = [f"c{i}" for i in range(7)]
ACTION_COLS = [f"a{i}" for i in range(7)]

# Recorded geometry as the probe logs it: the side of the start position a target
# sits on in image coordinates, where negative is further left in the frame.
EXPECTED_SIGN_COL = "expected_sign_image"
TARGET_SIGN_COLS = {"a": "target_sign_a_image", "b": "target_sign_b_image"}

# The side of the frame a lateral term names, in the same image coordinates. This
# is a fact about the words, independent of how image position maps to the sign of
# the action, which is why it can be used to classify an instruction as congruent
# or incongruent with its own target before anything is known about the model.
LATERAL_TERM_IMAGE_SIGN = {"left": -1, "leftmost": -1, "right": 1, "rightmost": 1}

# Image-x side of the gripper a unique object occupies. Negative is further left
# in the frame, matching the constructed-set geometry convention.
OBJECT_SIDE_LEFT = "left"
OBJECT_SIDE_RIGHT = "right"

# Translation components, in OpenVLA layout order. Index 0 is treated as lateral
# until a horizontal image flip shows which component actually reverses.
TRANSLATION_AXIS_NAMES = {0: "dx", 1: "dy", 2: "dz"}
TRANSLATION_AXES = tuple(TRANSLATION_AXIS_NAMES)

# Conditions the object-tracking gate predicts: the term-stripped instruction on
# the original frame and on its horizontal mirror. Names match the probe factorial
# so a later reader does not have to learn a second vocabulary.
TRACKING_CONDITION_ORIGINAL = "neutral"
TRACKING_CONDITION_MIRROR = "mirror_neutral"

# Columns of `object_tracking_set.csv`, the frozen geometry the gate scores
# against. Image coordinates, unconverted: the sign convention is applied in
# `object_tracking_report` through `lateral_sign`, the same way the constructed
# set keeps geometry and convention apart.
TRACKING_SET_FIELDS = (
    "scene_id", "episode_index", "instruction", "instr_neutral", "spatial_term",
    "noun", "image_path", "image_width", "image_height",
    "x_object", "y_object", "object_score",
    "x_gripper", "y_gripper", "gripper_source", "gripper_score",
    "object_side_image",
)


def value_column(axis_index: int, continuous: bool = True) -> str:
    """Name of the column holding one translation component."""
    return (CONTINUOUS_COLS if continuous else ACTION_COLS)[axis_index]


def has_continuous(df: pd.DataFrame) -> bool:
    """Whether the continuous readout is populated for every row."""
    return all(c in df.columns for c in CONTINUOUS_COLS) and not df[CONTINUOUS_COLS].isna().any().any()


def axis_value(df: pd.DataFrame, continuous: bool = True) -> pd.Series:
    """Value on each row's own expected axis, taken from `axis_index`.

    The axis differs per scene, so a single column cannot serve: a left/right
    pair is read on dx and a front/back pair on dy. Reading the axis from the
    row is what stops depth and vertical terms being pooled onto the lateral
    component, which is how the earlier analysis diluted its own measurement.
    """
    cols = CONTINUOUS_COLS if continuous else ACTION_COLS
    idx = df["axis_index"].astype(int).to_numpy()
    values = df[cols].to_numpy()
    return pd.Series(values[np.arange(len(df)), idx], index=df.index)


# --------------------------------------------------------------------------- #
# Paired tests and effect sizes
# --------------------------------------------------------------------------- #
def wilcoxon_paired(diffs) -> dict:
    """Wilcoxon signed-rank test on paired differences, with an effect size.

    Returns statistic, p value, median, and the matched-pairs rank-biserial
    correlation, which is bounded in [-1, 1] and reports the direction and
    magnitude of the shift independently of sample size. Degenerate inputs
    (fewer than two values, or all differences exactly zero) return NaN rather
    than raising, since an all-zero difference vector is itself a result worth
    displaying.
    """
    diffs = np.asarray([d for d in np.asarray(diffs, dtype=float) if np.isfinite(d)])
    out = {
        "n": int(diffs.size),
        "median": float(np.median(diffs)) if diffs.size else float("nan"),
        "mean": float(np.mean(diffs)) if diffs.size else float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
        "rank_biserial": float("nan"),
        "n_zero": int(np.sum(diffs == 0)),
    }
    nonzero = diffs[diffs != 0]
    if nonzero.size < 2:
        return out
    result = stats.wilcoxon(nonzero)
    out["statistic"] = float(result.statistic)
    out["p_value"] = float(result.pvalue)
    # Rank-biserial correlation: the signed-rank sums normalised by their total.
    ranks = stats.rankdata(np.abs(nonzero))
    total = ranks.sum()
    positive = ranks[nonzero > 0].sum()
    out["rank_biserial"] = float(2.0 * positive / total - 1.0) if total else float("nan")
    return out


def tost_equivalence(values, bound: float) -> dict:
    """Two one-sided tests for equivalence to zero within +/- `bound`.

    A non-significant difference test does not establish absence of an effect.
    Where the finding is that the model does not respond, the claim needs an
    equivalence test: both one-sided tests must reject for the mean to be
    declared negligible. The p value returned is the larger of the two, which is
    the p value of the equivalence claim.

    `bound` should be set from a quantity that makes the effect meaningful rather
    than from the data, for example one action bin width: a difference smaller
    than one bin cannot change the executable action.
    """
    values = np.asarray([v for v in np.asarray(values, dtype=float) if np.isfinite(v)])
    out = {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "bound": float(bound),
        "p_lower": float("nan"),
        "p_upper": float("nan"),
        "p_equivalence": float("nan"),
        "equivalent": False,
    }
    if values.size < 2 or np.allclose(values.std(ddof=1), 0.0):
        # With no variance the question is decided by the mean alone.
        if values.size and abs(out["mean"]) < bound:
            out.update(p_lower=0.0, p_upper=0.0, p_equivalence=0.0, equivalent=True)
        return out
    lower = stats.ttest_1samp(values, -bound, alternative="greater")
    upper = stats.ttest_1samp(values, bound, alternative="less")
    out["p_lower"] = float(lower.pvalue)
    out["p_upper"] = float(upper.pvalue)
    out["p_equivalence"] = float(max(lower.pvalue, upper.pvalue))
    out["equivalent"] = out["p_equivalence"] < 0.05
    return out


def resolution_report(values, bin_width: float) -> dict:
    """How much of a measured quantity survives the action quantisation.

    Reports the fraction of values that are exactly zero and the fraction
    smaller than one action bin. A measurement dominated by either is at the
    resolution floor of the argmax readout, and any conclusion drawn from its
    sign is unsafe. Retained as a first-class output rather than a footnote,
    because it is the defect that made the earlier results uninterpretable.
    """
    values = np.asarray([v for v in np.asarray(values, dtype=float) if np.isfinite(v)])
    if values.size == 0:
        return {"n": 0, "frac_exact_zero": float("nan"),
                "frac_below_bin": float("nan"), "n_distinct": 0,
                "bin_width": float(bin_width)}
    return {
        "n": int(values.size),
        "frac_exact_zero": float(np.mean(values == 0)),
        "frac_below_bin": float(np.mean(np.abs(values) < bin_width)),
        "n_distinct": int(np.unique(values).size),
        "bin_width": float(bin_width),
    }


# --------------------------------------------------------------------------- #
# Condition comparisons
# --------------------------------------------------------------------------- #
def _pivot_condition(df: pd.DataFrame, condition: str, continuous: bool = True) -> pd.DataFrame:
    """One row per scene for a condition, with the axis value per role."""
    sub = df[df["condition"] == condition].copy()
    if sub.empty:
        return pd.DataFrame(columns=["scene_id", "axis_index"])
    sub["value"] = axis_value(sub, continuous=continuous)
    wide = sub.pivot_table(index=["scene_id", "axis_index"], columns="role",
                           values="value", aggfunc="mean")
    return wide.reset_index()


def _decided(values, min_magnitude: float = 0.0) -> np.ndarray:
    """Whether each prediction picks a side the model could actually execute.

    A prediction of exactly zero picks no side at all. With `min_magnitude` set to
    one action bin, a prediction smaller than one bin is also treated as no
    decision, since the executable action it decodes to is the same either way and
    its sign is not something the model could act on.
    """
    magnitudes = np.abs(np.asarray(values, dtype=float))
    if min_magnitude > 0:
        return magnitudes >= min_magnitude
    return magnitudes > 0


def mirror_check(df: pd.DataFrame, continuous: bool = True) -> dict:
    """Does the lateral output reverse when the scene is reflected?

    Compares each scene's lateral value on the original image with its value on
    the horizontally flipped image, holding the instruction fixed. A model that
    reads lateral position must change sign, so the diagnostic quantity is the
    sum of the two: near zero for a reflection-antisymmetric response, and near
    twice the original value for a response that ignores the image.

    Reported for the term-bearing instruction and, separately, for the
    term-stripped instruction. The latter is the object-grounding baseline: a
    competent policy reaching for a single object flips when the object moves,
    with no spatial term involved. This is the instrument check the language
    tests are interpreted against, and it is not itself evidence of spatial
    language grounding.
    """
    original = _pivot_condition(df, "baseline", continuous)
    mirrored = _pivot_condition(df, "mirror", continuous)
    neutral = _pivot_condition(df, "neutral", continuous)
    mirror_neutral = _pivot_condition(df, "mirror_neutral", continuous)

    out: dict = {}
    for label, plain, flipped, role in (
        ("term", original, mirrored, "a"),
        ("neutral", neutral, mirror_neutral, "n"),
    ):
        if plain.empty or flipped.empty or role not in plain or role not in flipped:
            out[label] = {"n": 0}
            continue
        merged = plain[["scene_id", role]].merge(
            flipped[["scene_id", role]], on="scene_id", suffixes=("_plain", "_mirror")
        )
        a = merged[f"{role}_plain"].to_numpy()
        b = merged[f"{role}_mirror"].to_numpy()
        antisymmetry = a + b     # zero when the response reverses exactly
        invariance = a - b       # zero when the response ignores the reflection
        out[label] = {
            "n": int(len(merged)),
            "flip_rate": float(np.mean(a * b < 0)) if len(merged) else float("nan"),
            "identical_rate": float(np.mean(a == b)) if len(merged) else float("nan"),
            "antisymmetry": wilcoxon_paired(antisymmetry),
            "invariance": wilcoxon_paired(invariance),
            "mean_abs_original": float(np.mean(np.abs(a))) if len(merged) else float("nan"),
            "mean_abs_change": float(np.mean(np.abs(a - b))) if len(merged) else float("nan"),
        }
    return out


def mirror_check_by_configuration(df: pd.DataFrame, continuous: bool = True) -> dict:
    """`mirror_check` computed separately for each constructed arrangement.

    Pooling the arrangements hides the one that carries the signal. In the
    `opposite` arrangement the two instances sit either side of the start
    position, so reflecting the frame maps the layout onto itself: a response near
    the midpoint gives an antisymmetry and an invariance both near zero, and
    neither number says whether the visual channel is live. The same-side
    arrangements put both instances on one side, so a reflection genuinely moves
    the scene and the check has something to detect.

    Returns a mapping from configuration to the `mirror_check` result for it.
    Rows with no configuration, which is every unaltered Bridge scene, are grouped
    under the empty string and can be read as one stratum.
    """
    if df.empty or "configuration" not in df.columns:
        return {}
    labelled = df.assign(configuration=df["configuration"].fillna("").astype(str))
    return {str(name): mirror_check(group, continuous=continuous)
            for name, group in labelled.groupby("configuration")}


def lexical_check(df: pd.DataFrame, continuous: bool = True) -> dict:
    """Does the term contrast survive against a scene it does not describe?

    The baseline difference (instruction A minus instruction B on the original
    image) is compared with the same difference computed against another scene's
    image. A swapped-scene difference of comparable size means the response is
    driven by the words rather than by the scene, which would void the baseline
    contrast as evidence of grounding.
    """
    baseline = _pivot_condition(df, "baseline", continuous)
    swapped = _pivot_condition(df, "swapped_scene", continuous)
    out = {"baseline": {"n": 0}, "swapped_scene": {"n": 0}, "ratio": float("nan")}
    for label, frame in (("baseline", baseline), ("swapped_scene", swapped)):
        if frame.empty or "a" not in frame or "b" not in frame:
            continue
        diffs = (frame["a"] - frame["b"]).to_numpy()
        summary = wilcoxon_paired(diffs)
        summary["mean_abs"] = float(np.mean(np.abs(diffs))) if diffs.size else float("nan")
        summary["flip_rate"] = float(np.mean(frame["a"] * frame["b"] < 0))
        out[label] = summary
    base_abs = out["baseline"].get("mean_abs", float("nan"))
    swap_abs = out["swapped_scene"].get("mean_abs", float("nan"))
    if base_abs and np.isfinite(base_abs) and np.isfinite(swap_abs) and base_abs > 0:
        # Near one means the contrast is reproduced without the scene.
        out["ratio"] = float(swap_abs / base_abs)
    return out


def term_effect(df: pd.DataFrame, continuous: bool = True) -> dict:
    """The term's contribution relative to the same image with no spatial term.

    For each scene, the term-bearing predictions are expressed as deviations
    from the term-stripped prediction on the identical image. If the term
    carries directional information, the two variants deviate from that
    reference in opposite directions, so the product of the deviations is
    negative. Measuring against a within-scene reference removes whatever the
    scene contributes on its own, which the raw A minus B difference cannot.
    """
    baseline = _pivot_condition(df, "baseline", continuous)
    neutral = _pivot_condition(df, "neutral", continuous)
    if baseline.empty or neutral.empty or "n" not in neutral:
        return {"n": 0}
    merged = baseline.merge(neutral[["scene_id", "n"]], on="scene_id")
    if merged.empty:
        return {"n": 0}
    dev_a = (merged["a"] - merged["n"]).to_numpy()
    dev_b = (merged["b"] - merged["n"]).to_numpy()
    return {
        "n": int(len(merged)),
        "deviation_a": wilcoxon_paired(dev_a),
        "deviation_b": wilcoxon_paired(dev_b),
        "opposed_rate": float(np.mean(dev_a * dev_b < 0)),
        "both_zero_rate": float(np.mean((dev_a == 0) & (dev_b == 0))),
        "separation": wilcoxon_paired(dev_a - dev_b),
    }


def congruence_test(df: pd.DataFrame, continuous: bool = True, *,
                    lateral_sign: int = 1, condition: str = "baseline",
                    geometry_sign: int = 1, min_magnitude: float = 0.0) -> dict:
    """Does the paired difference point the way the recorded layout demands?

    Requires `expected_sign_image`, the recorded order of the two candidate
    targets in image coordinates. `lateral_sign` converts that to the sign the
    lateral action should take, and `geometry_sign` negates the geometry for a
    stimulus the recorded layout no longer describes: a mirrored image reverses
    every recorded side, so the mirror condition is read with `geometry_sign=-1`.

    This is a direction check, not the decisive test, and the distinction
    matters. In the `opposite` configuration the term's conventional direction
    and the actual layout coincide by construction, so a model that maps `left`
    to a fixed direction without ever looking at the scene agrees here just as
    completely as a grounded one. What the statistic does establish is that the
    difference is oriented consistently with the geometry at all, which is a
    precondition for reading anything from its sign and which also fixes the
    sign convention relating image position to the lateral action empirically.

    `absolute_congruence` is the discriminating counterpart: it scores each
    instruction against its own target's side rather than scoring the pair
    against their relative order.
    """
    empty = {"n": 0, "by_configuration": {}, "n_undecided": 0}
    pivot = _pivot_condition(df, condition, continuous)
    if pivot.empty or "a" not in pivot or "b" not in pivot:
        return empty
    if EXPECTED_SIGN_COL not in df.columns:
        return {**empty, "note": f"{EXPECTED_SIGN_COL} was not logged"}
    meta = (df[["scene_id", "configuration", EXPECTED_SIGN_COL]]
            .drop_duplicates(subset="scene_id"))
    merged = pivot.merge(meta, on="scene_id", how="left")
    recorded = pd.to_numeric(merged[EXPECTED_SIGN_COL], errors="coerce")
    merged = merged[recorded.notna() & (recorded != 0)]
    if merged.empty:
        return empty

    diff = (merged["a"] - merged["b"]).to_numpy()
    expected = (pd.to_numeric(merged[EXPECTED_SIGN_COL]).astype(float).to_numpy()
                * np.sign(lateral_sign) * np.sign(geometry_sign))
    decided = _decided(diff, min_magnitude)
    # Positive means the observed difference points the way the geometry demands.
    oriented = diff * np.sign(expected)

    scored = merged.assign(oriented=oriented, decided=decided)
    usable = scored[scored["decided"]]
    if usable.empty:
        return {**empty, "n_undecided": int(len(scored))}

    by_config: dict = {}
    for name, group in usable.groupby("configuration"):
        values = group["oriented"].to_numpy()
        by_config[str(name)] = {
            "n": int(values.size),
            "agreement": float(np.mean(values > 0)),
            "test": wilcoxon_paired(values),
        }
    oriented_usable = usable["oriented"].to_numpy()
    return {
        "n": int(len(usable)),
        "n_undecided": int(len(scored) - len(usable)),
        "min_magnitude": float(min_magnitude),
        "agreement": float(np.mean(oriented_usable > 0)),
        "test": wilcoxon_paired(oriented_usable),
        "by_configuration": by_config,
    }


def absolute_congruence(df: pd.DataFrame, continuous: bool = True, *,
                        lateral_sign: int = 1, condition: str = "baseline",
                        geometry_sign: int = 1,
                        min_magnitude: float = 0.0) -> dict:
    """Does each instruction move toward the side its own target occupies?

    Requires `target_sign_a_image` and `target_sign_b_image`, the side of the
    start position each instruction's target sits on in image coordinates,
    recorded when the scene was constructed. `lateral_sign` and `geometry_sign`
    are as in `congruence_test`.

    Scoring each instruction separately, rather than scoring the pair against
    their relative order, is what separates the two accounts. On a same-side
    scene both targets lie in one direction, so a grounded model agrees on both
    instructions while a word-to-direction mapping necessarily disagrees on one
    of them: it sends `left` one way and `right` the other, and only one of
    those can point at a target when both targets share a side. The expected
    agreement rate is therefore near one for a grounded model and near one half
    for a lexical one, and the paired difference cannot show this because the
    two accounts produce the same difference.

    Read with the sign convention in mind. Agreement near one is scene
    grounding, near one half is a word-to-direction mapping, and near zero is
    scene grounding under an inverted `lateral_sign` rather than a third account.

    Three groupings are reported. `by_configuration` keeps the arrangements apart,
    since the `opposite` arrangement cannot separate the accounts. `by_congruency`
    pools the instruction-level observations by whether the instruction's own term
    names the side its target actually occupies, which is where the discrimination
    is concentrated: the congruent instruction is agreed on by both accounts, and
    the incongruent one is where they part. Pooling that way also doubles the
    sample for the decisive number, because each same-side arrangement contributes
    one incongruent instruction whichever side it was built on.
    """
    empty = {"n": 0, "by_configuration": {}, "by_congruency": {}}
    pivot = _pivot_condition(df, condition, continuous)
    needed = set(TARGET_SIGN_COLS.values())
    if pivot.empty or not {"a", "b"} <= set(pivot.columns):
        return empty
    if not needed <= set(df.columns):
        return {**empty, "note": f"{sorted(needed)} were not logged"}

    meta_cols = ["scene_id", "configuration", *TARGET_SIGN_COLS.values()]
    if "spatial_term" in df.columns:
        meta_cols.append("spatial_term")
    meta = df[meta_cols].drop_duplicates(subset="scene_id")
    merged = pivot.merge(meta, on="scene_id", how="left")
    for col in TARGET_SIGN_COLS.values():
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=list(TARGET_SIGN_COLS.values()))
    merged = merged[(merged[TARGET_SIGN_COLS["a"]] != 0)
                    & (merged[TARGET_SIGN_COLS["b"]] != 0)]
    if merged.empty:
        return empty

    convention = np.sign(lateral_sign) * np.sign(geometry_sign)
    # The side of the action space each instruction's own target lies in.
    action_side = {role: np.sign(merged[col].to_numpy() * convention)
                   for role, col in TARGET_SIGN_COLS.items()}
    agree_a = np.sign(merged["a"].to_numpy()) == action_side["a"]
    agree_b = np.sign(merged["b"].to_numpy()) == action_side["b"]
    # A prediction that picks no side the model could execute cannot agree with
    # either target; see `_decided`.
    resolved = (_decided(merged["a"].to_numpy(), min_magnitude)
                & _decided(merged["b"].to_numpy(), min_magnitude))

    # Whether each instruction's own term names the side its target occupies.
    # Computed from the words and the recorded image position alone, so it does
    # not depend on the convention the agreement rates are scored under.
    term = (merged["spatial_term"].astype(str).str.lower().str.strip()
            if "spatial_term" in merged.columns
            else pd.Series([""] * len(merged), index=merged.index))
    term_side_a = term.map(LATERAL_TERM_IMAGE_SIGN)
    congruent_a = np.sign(merged[TARGET_SIGN_COLS["a"]].to_numpy()) == term_side_a
    # Role b carries the antonym, which names the opposite side of the frame.
    congruent_b = np.sign(merged[TARGET_SIGN_COLS["b"]].to_numpy()) == -term_side_a

    merged = merged.assign(agree_a=agree_a, agree_b=agree_b, resolved=resolved,
                           congruent_a=congruent_a, congruent_b=congruent_b,
                           known_congruency=term_side_a.notna(),
                           both=agree_a & agree_b,
                           n_agree=agree_a.astype(int) + agree_b.astype(int))

    by_config: dict = {}
    for name, group in merged.groupby("configuration"):
        usable = group[group["resolved"]]
        if usable.empty:
            by_config[str(name)] = {"n": 0}
            continue
        by_config[str(name)] = {
            "n": int(len(usable)),
            "agreement": float(usable["n_agree"].mean() / 2.0),
            "both_correct": float(usable["both"].mean()),
            "agreement_a": float(usable["agree_a"].mean()),
            "agreement_b": float(usable["agree_b"].mean()),
        }

    usable = merged[merged["resolved"]]
    by_congruency: dict = {}
    known = usable[usable["known_congruency"]]
    if len(known):
        observations = pd.DataFrame({
            "agree": np.concatenate([known["agree_a"].to_numpy(),
                                     known["agree_b"].to_numpy()]),
            "congruent": np.concatenate([known["congruent_a"].to_numpy(),
                                         known["congruent_b"].to_numpy()]),
            "configuration": pd.concat([known["configuration"],
                                        known["configuration"]]).to_numpy(),
        })
        for label, flag in (("congruent", True), ("incongruent", False)):
            group = observations[observations["congruent"] == flag]
            by_congruency[label] = {
                "n": int(len(group)),
                "agreement": (float(group["agree"].mean()) if len(group)
                              else float("nan")),
                "by_configuration": {
                    str(name): {"n": int(len(sub)),
                                "agreement": float(sub["agree"].mean())}
                    for name, sub in group.groupby("configuration")
                },
            }

    return {
        "n": int(len(usable)),
        "n_unresolved": int((~merged["resolved"]).sum()),
        "min_magnitude": float(min_magnitude),
        "agreement": float(usable["n_agree"].mean() / 2.0) if len(usable) else float("nan"),
        "both_correct": float(usable["both"].mean()) if len(usable) else float("nan"),
        "by_configuration": by_config,
        "by_congruency": by_congruency,
    }


def same_side_test(df: pd.DataFrame, continuous: bool = True, *,
                   condition: str = "baseline",
                   min_magnitude: float = 0.0) -> dict:
    """Separate scene grounding from a word-to-direction mapping.

    On a constructed scene with both instances on the same side of the start
    position, the two accounts make opposite predictions and nothing else
    distinguishes them:

      * A model that grounds the term selects between two targets that both lie
        in the same direction, so both instructions produce a same-signed action
        and they differ only in magnitude.
      * A model that maps `left` to one direction and `right` to the other
        produces oppositely-signed actions, since the arrangement plays no part.

    On the `opposite` configuration the two accounts agree, which is precisely
    why the earlier sign-flip metric could not tell them apart: it scored the
    behaviour they share. Contrasting the agreement rate between configurations
    is the decisive comparison, and it is only available because the same-side
    arrangement was constructed rather than sought.

    Reports, per configuration, how often the two instructions produce
    same-signed actions, and the difference between the configurations both
    unpaired and, where `base_scene_id` is present, paired within the base frame
    the two arrangements were built from. The paired form is the one the
    constructed set was designed to support: the frozen evaluation set draws each
    same-side scene and its `opposite` counterpart from the same frame, so the
    paired contrast holds the background, the object, the cutout, and the
    instruction fixed and leaves the arrangement as the only difference. The
    unpaired contrast is retained because it covers scenes whose counterpart is
    missing from the log.

    This statistic reads only whether two predictions share a sign, so it is
    unaffected by the convention relating image position to the sign of the
    action.
    """
    pivot = _pivot_condition(df, condition, continuous)
    if pivot.empty or "a" not in pivot or "b" not in pivot:
        return {"n": 0, "by_configuration": {}}
    meta_cols = ["scene_id", "configuration"]
    if "base_scene_id" in df.columns:
        meta_cols.append("base_scene_id")
    meta = df[meta_cols].drop_duplicates(subset="scene_id")
    merged = pivot.merge(meta, on="scene_id", how="left")
    merged = merged[merged["configuration"].notna() & (merged["configuration"] != "")]
    if merged.empty:
        return {"n": 0, "by_configuration": {}}

    merged = merged.assign(
        same_sign=(merged["a"] * merged["b"] > 0),
        resolved=(_decided(merged["a"].to_numpy(), min_magnitude)
                  & _decided(merged["b"].to_numpy(), min_magnitude)),
        magnitude_gap=(merged["a"].abs() - merged["b"].abs()).abs(),
    )

    by_config: dict = {}
    for name, group in merged.groupby("configuration"):
        usable = group[group["resolved"]]
        by_config[str(name)] = {
            "n": int(len(usable)),
            "n_unresolved": int(len(group) - len(usable)),
            "same_sign_rate": float(usable["same_sign"].mean()) if len(usable) else float("nan"),
            "median_magnitude_gap": float(usable["magnitude_gap"].median())
            if len(usable) else float("nan"),
        }

    # Grounded and lexical accounts differ in how the same-sign rate moves
    # between the two arrangements, so the contrast is the quantity of interest.
    same_side = merged[merged["configuration"].str.startswith("same_side")
                       & merged["resolved"]]
    opposite = merged[(merged["configuration"] == "opposite") & merged["resolved"]]
    contrast = {"n_same_side": int(len(same_side)), "n_opposite": int(len(opposite)),
                "p_value": float("nan"), "difference": float("nan")}
    if len(same_side) and len(opposite):
        contrast["difference"] = float(same_side["same_sign"].mean()
                                       - opposite["same_sign"].mean())
        table = np.array([
            [int(same_side["same_sign"].sum()), int((~same_side["same_sign"]).sum())],
            [int(opposite["same_sign"].sum()), int((~opposite["same_sign"]).sum())],
        ])
        if table.sum(axis=1).min() > 0 and table.sum(axis=0).min() > 0:
            contrast["p_value"] = float(stats.fisher_exact(table)[1])
    return {
        "n": int(len(merged)),
        "min_magnitude": float(min_magnitude),
        "by_configuration": by_config,
        "contrast": contrast,
        "contrast_paired": _paired_contrast(same_side, opposite),
    }


def _paired_contrast(same_side: pd.DataFrame, opposite: pd.DataFrame) -> dict:
    """The same-side against opposite contrast, paired within the base frame.

    Each base frame contributes at most one pair, so the comparison holds the
    frame fixed and the arrangement is the only thing that differs. Only the
    discordant pairs carry information about the difference, which is McNemar's
    test; the exact binomial form is used because the discordant count can be
    small. Pairs are dropped rather than approximated when a frame supplied only
    one of the two arrangements, and the count of those is reported.
    """
    out = {"n_pairs": 0, "n_discordant": 0, "difference": float("nan"),
           "p_value": float("nan"), "n_unpaired": 0}
    if "base_scene_id" not in same_side.columns:
        return {**out, "note": "base_scene_id was not logged, so no pairing exists"}
    left = same_side.dropna(subset=["base_scene_id"]).drop_duplicates("base_scene_id")
    right = opposite.dropna(subset=["base_scene_id"]).drop_duplicates("base_scene_id")
    if left.empty or right.empty:
        return out
    merged = left[["base_scene_id", "same_sign"]].merge(
        right[["base_scene_id", "same_sign"]], on="base_scene_id",
        suffixes=("_same_side", "_opposite"))
    out["n_unpaired"] = int(len(left) + len(right) - 2 * len(merged))
    if merged.empty:
        return out
    a = merged["same_sign_same_side"].to_numpy()
    b = merged["same_sign_opposite"].to_numpy()
    only_same_side = int(np.sum(a & ~b))
    only_opposite = int(np.sum(b & ~a))
    discordant = only_same_side + only_opposite
    out.update({
        "n_pairs": int(len(merged)),
        "n_discordant": discordant,
        "only_same_side": only_same_side,
        "only_opposite": only_opposite,
        "difference": float(np.mean(a) - np.mean(b)),
    })
    if discordant:
        out["p_value"] = float(
            stats.binomtest(only_same_side, discordant, 0.5).pvalue)
    return out


def determinism_check(df: pd.DataFrame, continuous: bool = True) -> dict:
    """Confirm repeated identical inputs produce identical outputs.

    Decoding is greedy with fixed seeds, so repeats of the same stimulus must
    agree exactly. Establishing that means any nonzero difference elsewhere is
    attributable to the manipulation rather than to run-to-run variation.
    """
    cols = CONTINUOUS_COLS if continuous else ACTION_COLS
    keys = ["scene_id", "condition", "role", "instruction"]
    present = [k for k in keys if k in df.columns]
    if not present or df.empty:
        return {"repeated_stimuli": 0, "max_spread": 0.0, "deterministic": True}

    grouped = df.groupby(present)
    sizes = grouped[present[0]].transform("size")
    repeated = sizes > 1
    if not bool(repeated.any()):
        return {"repeated_stimuli": 0, "max_spread": 0.0, "deterministic": True}

    values = grouped[cols]
    spread = values.transform("max") - values.transform("min")
    worst = float(np.nanmax(spread[repeated].to_numpy()))
    return {
        "repeated_stimuli": int(df[repeated].groupby(present).ngroups),
        "max_spread": worst,
        "deterministic": worst == 0.0,
    }


# --------------------------------------------------------------------------- #
# First-action object tracking
# --------------------------------------------------------------------------- #
def object_side_image(x_object, x_gripper) -> int:
    """Sign of the object's offset from the gripper in image x.

    Negative: the object sits further left in the frame. Zero: they share a
    column, so no lateral direction is defined.
    """
    offset = float(x_object) - float(x_gripper)
    if offset == 0:
        return 0
    return int(np.sign(offset))


def mirrored_x(x, image_width) -> float:
    """Image-x of a point after a horizontal flip of a frame of `image_width`."""
    return float(image_width) - float(x)


def toward_object(dx, x_object, x_gripper, *, min_magnitude: float = 0.0,
                  lateral_sign: int = 1):
    """Whether a lateral action points at the object.

    Returns True, False, or None. None when the object shares the gripper's
    column, so no lateral target exists, or when the action is smaller than
    `min_magnitude` and so is not a side the model could execute.

    `lateral_sign` is the convention relating image-x to the sign of `dx`: a
    positive value means an object further right in the frame should produce a
    positive action. It is applied here rather than baked into the geometry so a
    revision is a re-read, matching the constructed-set statistics.
    """
    side = object_side_image(x_object, x_gripper)
    if side == 0:
        return None
    if not bool(_decided([dx], min_magnitude)[0]):
        return None
    expected = side * int(np.sign(lateral_sign))
    return int(np.sign(dx)) == expected


def object_tracking_candidates(rows) -> dict:
    """Validation-role rows that can enter the first-action object-tracking gate.

    The wording filter is applied here; instance count is not. A scene labelled
    `duplicate_target='yes'` is excluded because two instances make a term-free
    instruction ambiguous, and detection in Notebook 04 would only confirm that.
    Every other validation row with a clean term strip and a nameable object is
    returned, so the detector, not the harvest label, decides that the scene
    holds exactly one instance.

    Returns `rows` (enriched with `spatial_term`, `instr_neutral`, and `noun`)
    and `skipped`, a count of rows dropped for each reason, so the notebook can
    print the funnel rather than a silent shortfall.
    """
    from controls import strip_spatial_term
    from data import SPLIT_VALIDATION, make_pair
    from detect_duplicates import extract_target_noun

    out = []
    skipped = {"split": 0, "duplicate": 0, "pair": 0, "strip": 0, "noun": 0}
    for row in rows:
        if str(row.get("split", "")).strip() != SPLIT_VALIDATION:
            skipped["split"] += 1
            continue
        if str(row.get("duplicate_target", "")).strip() == "yes":
            skipped["duplicate"] += 1
            continue
        instruction = str(row.get("instruction") or "")
        made = make_pair(instruction)
        if made is None:
            skipped["pair"] += 1
            continue
        term, _ = made
        neutral = strip_spatial_term(instruction, term)
        if not neutral:
            skipped["strip"] += 1
            continue
        noun = extract_target_noun(instruction)
        if not noun:
            skipped["noun"] += 1
            continue
        item = dict(row)
        item["spatial_term"] = term
        item["instr_neutral"] = neutral
        item["noun"] = noun
        out.append(item)
    return {"rows": out, "skipped": skipped, "n": len(out)}


def _tracking_rate(flags) -> float:
    """Mean of a boolean sequence, or NaN when nothing was decided."""
    values = np.asarray(list(flags), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _side_summary(rows) -> dict:
    """Toward-object and flip rates for one object-side stratum."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "n_undecided_original": 0,
            "n_undecided_mirror": 0,
            "toward_original": float("nan"),
            "toward_mirror": float("nan"),
            "flip_rate": float("nan"),
        }
    orig = [r["toward_original"] for r in rows]
    mir = [r["toward_mirror"] for r in rows]
    decided_orig = [v for v in orig if v is not None]
    decided_mir = [v for v in mir if v is not None]
    flips = [r["flip"] for r in rows if r["flip"] is not None]
    return {
        "n": n,
        "n_undecided_original": int(sum(v is None for v in orig)),
        "n_undecided_mirror": int(sum(v is None for v in mir)),
        "toward_original": _tracking_rate(decided_orig),
        "toward_mirror": _tracking_rate(decided_mir),
        "flip_rate": _tracking_rate(flips),
    }


def object_tracking_report(predictions: pd.DataFrame, geometry: pd.DataFrame, *,
                           continuous: bool = True, min_magnitude: float = 0.0,
                           lateral_sign: int = 1, axis_index: int = 0,
                           original_condition: str = TRACKING_CONDITION_ORIGINAL,
                           mirror_condition: str = TRACKING_CONDITION_MIRROR,
                           ) -> dict:
    """Does one translation component track a unique object under a mirror?

    `axis_index` selects `c0`/`a0` (dx), `c1`/`a1` (dy), or `c2`/`a2` (dz). The
    default is dx, which `TERM_AXIS` treats as lateral. Scoring dy or dz the same
    way asks whether that component behaves as the image-x channel: a horizontal
    flip reverses image x, so the true lateral component must reverse and must
    point at the object on both sides of the gripper.

    A component that keeps one sign on the original and mirrored frames is not
    that channel. Scoring it as lateral produces a high toward-object rate when
    the object sits left of the gripper and a low rate when it sits right, which
    is indistinguishable from an image-left prior. The fraction of negative
    decided values on each frame is reported so that pattern can be read as a
    constant sign rather than as tracking.

    `predictions` carries one row per (scene, condition). `geometry` carries
    `x_object`, `x_gripper`, and `image_width` per `scene_id`. The mirrored
    object and gripper positions are computed from those three columns.

    Rates are reported overall and split by the object's side of the gripper.
    Undecided predictions (sub-bin, or an object on the gripper's column) are
    excluded from each rate and counted separately.
    """
    empty = {
        "n": 0, "n_undecided_original": 0, "n_undecided_mirror": 0,
        "min_magnitude": float(min_magnitude), "lateral_sign": int(lateral_sign),
        "axis_index": int(axis_index),
        "toward_original": float("nan"), "toward_mirror": float("nan"),
        "flip_rate": float("nan"), "identical_rate": float("nan"),
        "negative_rate_original": float("nan"),
        "negative_rate_mirror": float("nan"),
        "mean_original": float("nan"), "mean_mirror": float("nan"),
        "by_object_side": {OBJECT_SIDE_LEFT: _side_summary([]),
                           OBJECT_SIDE_RIGHT: _side_summary([])},
    }
    if predictions is None or predictions.empty or geometry is None or geometry.empty:
        return empty
    if "scene_id" not in predictions.columns or "scene_id" not in geometry.columns:
        return empty

    stem = "c" if continuous else "a"
    value_col = f"{stem}{int(axis_index)}"
    if value_col not in predictions.columns:
        alt = f"a{int(axis_index)}" if continuous else f"c{int(axis_index)}"
        value_col = alt if alt in predictions.columns else None
    if value_col is None:
        return empty

    needed = {"x_object", "x_gripper", "image_width"}
    if not needed <= set(geometry.columns):
        return {**empty, "note": f"{sorted(needed)} were not recorded"}

    orig = predictions[predictions["condition"] == original_condition]
    mir = predictions[predictions["condition"] == mirror_condition]
    if orig.empty or mir.empty:
        return empty

    orig_val = orig.groupby("scene_id")[value_col].mean()
    mir_val = mir.groupby("scene_id")[value_col].mean()
    geo = geometry.drop_duplicates(subset="scene_id").set_index("scene_id")

    scored = []
    for scene_id in orig_val.index.intersection(mir_val.index).intersection(geo.index):
        row = geo.loc[scene_id]
        x_object = float(row["x_object"])
        x_gripper = float(row["x_gripper"])
        width = float(row["image_width"])
        side = object_side_image(x_object, x_gripper)
        if side == 0:
            continue
        dx_orig = float(orig_val.loc[scene_id])
        dx_mir = float(mir_val.loc[scene_id])
        toward_orig = toward_object(
            dx_orig, x_object, x_gripper,
            min_magnitude=min_magnitude, lateral_sign=lateral_sign)
        toward_mir = toward_object(
            dx_mir, mirrored_x(x_object, width), mirrored_x(x_gripper, width),
            min_magnitude=min_magnitude, lateral_sign=lateral_sign)
        orig_decided = bool(_decided([dx_orig], min_magnitude)[0])
        mir_decided = bool(_decided([dx_mir], min_magnitude)[0])
        flip = None
        if orig_decided and mir_decided:
            flip = bool(dx_orig * dx_mir < 0)
        scored.append({
            "scene_id": scene_id,
            "side": OBJECT_SIDE_LEFT if side < 0 else OBJECT_SIDE_RIGHT,
            "toward_original": toward_orig,
            "toward_mirror": toward_mir,
            "flip": flip,
            "identical": bool(dx_orig == dx_mir) if orig_decided and mir_decided else None,
            "value_original": dx_orig,
            "value_mirror": dx_mir,
            "decided_original": orig_decided,
            "decided_mirror": mir_decided,
        })

    if not scored:
        return empty

    by_side = {
        OBJECT_SIDE_LEFT: _side_summary([r for r in scored if r["side"] == OBJECT_SIDE_LEFT]),
        OBJECT_SIDE_RIGHT: _side_summary([r for r in scored if r["side"] == OBJECT_SIDE_RIGHT]),
    }
    overall = _side_summary(scored)
    identical = [r["identical"] for r in scored if r["identical"] is not None]
    orig_decided_vals = [r["value_original"] for r in scored if r["decided_original"]]
    mir_decided_vals = [r["value_mirror"] for r in scored if r["decided_mirror"]]
    return {
        "n": overall["n"],
        "n_undecided_original": overall["n_undecided_original"],
        "n_undecided_mirror": overall["n_undecided_mirror"],
        "min_magnitude": float(min_magnitude),
        "lateral_sign": int(lateral_sign),
        "axis_index": int(axis_index),
        "toward_original": overall["toward_original"],
        "toward_mirror": overall["toward_mirror"],
        "flip_rate": overall["flip_rate"],
        "identical_rate": _tracking_rate(identical),
        "negative_rate_original": _tracking_rate(v < 0 for v in orig_decided_vals),
        "negative_rate_mirror": _tracking_rate(v < 0 for v in mir_decided_vals),
        "mean_original": (float(np.mean(orig_decided_vals))
                          if orig_decided_vals else float("nan")),
        "mean_mirror": (float(np.mean(mir_decided_vals))
                        if mir_decided_vals else float("nan")),
        "by_object_side": by_side,
    }


def object_tracking_by_axis(predictions: pd.DataFrame, geometry: pd.DataFrame, *,
                            continuous: bool = True, lateral_sign: int = 1,
                            bin_widths=None, min_magnitude: float = 0.0,
                            original_condition: str = TRACKING_CONDITION_ORIGINAL,
                            mirror_condition: str = TRACKING_CONDITION_MIRROR,
                            ) -> dict:
    """Score dx, dy, and dz as if each were the image-x channel.

    A horizontal flip reverses image x. The component that is lateral must
    reverse and must track the object on both sides of the gripper. Components
    that keep one sign are not that channel. `bin_widths`, when given, is a
    sequence of per-component floors so dy and dz are not judged against the
    lateral bin width. The gate itself still reads dx; this scan is the check
    that dx is the right component to have gated on.
    """
    out = {}
    for axis in TRANSLATION_AXES:
        if bin_widths is not None:
            mag = float(bin_widths[axis])
        else:
            mag = float(min_magnitude)
        report = object_tracking_report(
            predictions, geometry, continuous=continuous, min_magnitude=mag,
            lateral_sign=lateral_sign, axis_index=axis,
            original_condition=original_condition,
            mirror_condition=mirror_condition)
        report["name"] = TRANSLATION_AXIS_NAMES[axis]
        out[axis] = report
    return out


def object_tracking_gate(report, *, min_toward: float = 0.7,
                         min_flip: float = 0.5, min_n_per_side: int = 10) -> dict:
    """Pass/fail for the first-action object-tracking instrument check.

    Passes only when toward-object holds on both sides of the gripper, on both
    the original and the mirrored frame, the action reverses under the mirror,
    and each side has enough scenes that a rate is not a handful of draws. A
    pooled rate is not sufficient: that is how an image-left prior masquerades
    as tracking.
    """
    reasons = []
    sides = report.get("by_object_side") or {}
    for name in (OBJECT_SIDE_LEFT, OBJECT_SIDE_RIGHT):
        side = sides.get(name) or {}
        n = int(side.get("n") or 0)
        n_decided = n - int(side.get("n_undecided_original") or 0)
        if n_decided < min_n_per_side:
            reasons.append(
                f"{name}: {n_decided} decided scenes, below the "
                f"{min_n_per_side} this check requires")
            continue
        for key, label in (("toward_original", "original toward-object"),
                           ("toward_mirror", "mirrored toward-object"),
                           ("flip_rate", "flip rate")):
            rate = side.get(key, float("nan"))
            floor = min_flip if key == "flip_rate" else min_toward
            if not (np.isfinite(rate) and rate >= floor):
                shown = f"{rate:.1%}" if np.isfinite(rate) else "undefined"
                reasons.append(f"{name} {label} {shown}, below {floor:.0%}")
    overall_flip = report.get("flip_rate", float("nan"))
    if not (np.isfinite(overall_flip) and overall_flip >= min_flip):
        shown = f"{overall_flip:.1%}" if np.isfinite(overall_flip) else "undefined"
        reasons.append(f"overall flip rate {shown}, below {min_flip:.0%}")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "min_toward": float(min_toward),
        "min_flip": float(min_flip),
        "min_n_per_side": int(min_n_per_side),
        "n": int(report.get("n") or 0),
    }
