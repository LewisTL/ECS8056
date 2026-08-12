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

Two conventions apply throughout. Statistics are computed on the continuous
expected-bin readout (`c*` columns) where available, because the argmax action is
quantised at roughly the size of the effect. And a null is reported with an
equivalence test rather than a non-significant p value, since the claim in the
null case is that the effect is negligible, not that it is unproven.

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


def congruence_test(df: pd.DataFrame, continuous: bool = True) -> dict:
    """Does the paired difference point the way the recorded layout demands?

    Requires `expected_sign`, the direction the difference between the two
    instructions should take given where the two candidate targets sit.

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
    baseline = _pivot_condition(df, "baseline", continuous)
    if baseline.empty or "a" not in baseline or "b" not in baseline:
        return {"n": 0, "by_configuration": {}}
    meta = (df[["scene_id", "configuration", "expected_sign"]]
            .drop_duplicates(subset="scene_id"))
    merged = baseline.merge(meta, on="scene_id", how="left")
    merged = merged[merged["expected_sign"].notna() & (merged["expected_sign"] != 0)]
    if merged.empty:
        return {"n": 0, "by_configuration": {}}

    diff = (merged["a"] - merged["b"]).to_numpy()
    expected = merged["expected_sign"].astype(float).to_numpy()
    # Positive means the observed difference points the way the geometry demands.
    oriented = diff * np.sign(expected)

    by_config: dict = {}
    for name, group in merged.assign(oriented=oriented).groupby("configuration"):
        values = group["oriented"].to_numpy()
        by_config[str(name)] = {
            "n": int(values.size),
            "agreement": float(np.mean(values > 0)),
            "test": wilcoxon_paired(values),
        }
    return {
        "n": int(len(merged)),
        "agreement": float(np.mean(oriented > 0)),
        "test": wilcoxon_paired(oriented),
        "by_configuration": by_config,
    }


def absolute_congruence(df: pd.DataFrame, continuous: bool = True) -> dict:
    """Does each instruction move toward the side its own target occupies?

    Requires `target_sign_a` and `target_sign_b`, the side of the start position
    each instruction's target sits on, recorded when the scene was constructed.

    Scoring each instruction separately, rather than scoring the pair against
    their relative order, is what separates the two accounts. On a same-side
    scene both targets lie in one direction, so a grounded model agrees on both
    instructions while a word-to-direction mapping necessarily disagrees on one
    of them: it sends `left` one way and `right` the other, and only one of
    those can point at a target when both targets share a side. The expected
    agreement rate is therefore near one for a grounded model and near one half
    for a lexical one, and the paired difference cannot show this because the
    two accounts produce the same difference.

    On the `opposite` configuration the accounts agree, which is why the rate is
    reported per configuration rather than pooled.
    """
    baseline = _pivot_condition(df, "baseline", continuous)
    needed = {"target_sign_a", "target_sign_b"}
    if baseline.empty or not {"a", "b"} <= set(baseline.columns):
        return {"n": 0, "by_configuration": {}}
    if not needed <= set(df.columns):
        return {"n": 0, "by_configuration": {},
                "note": "target_sign_a and target_sign_b were not logged"}

    meta = (df[["scene_id", "configuration", "target_sign_a", "target_sign_b"]]
            .drop_duplicates(subset="scene_id"))
    merged = baseline.merge(meta, on="scene_id", how="left")
    merged = merged[merged["target_sign_a"].notna() & merged["target_sign_b"].notna()]
    merged = merged[(merged["target_sign_a"] != 0) & (merged["target_sign_b"] != 0)]
    if merged.empty:
        return {"n": 0, "by_configuration": {}}

    agree_a = np.sign(merged["a"].to_numpy()) == np.sign(
        merged["target_sign_a"].astype(float).to_numpy())
    agree_b = np.sign(merged["b"].to_numpy()) == np.sign(
        merged["target_sign_b"].astype(float).to_numpy())
    # A zero prediction picks no side, so it cannot agree with either target.
    resolved = (merged["a"] != 0).to_numpy() & (merged["b"] != 0).to_numpy()

    merged = merged.assign(agree_a=agree_a, agree_b=agree_b, resolved=resolved,
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
    return {
        "n": int(len(usable)),
        "n_unresolved": int((~merged["resolved"]).sum()),
        "agreement": float(usable["n_agree"].mean() / 2.0) if len(usable) else float("nan"),
        "both_correct": float(usable["both"].mean()) if len(usable) else float("nan"),
        "by_configuration": by_config,
    }


def same_side_test(df: pd.DataFrame, continuous: bool = True) -> dict:
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
    same-signed actions, and the difference between the configurations with a
    proportion test.
    """
    baseline = _pivot_condition(df, "baseline", continuous)
    if baseline.empty or "a" not in baseline or "b" not in baseline:
        return {"n": 0, "by_configuration": {}}
    meta = df[["scene_id", "configuration"]].drop_duplicates(subset="scene_id")
    merged = baseline.merge(meta, on="scene_id", how="left")
    merged = merged[merged["configuration"].notna() & (merged["configuration"] != "")]
    if merged.empty:
        return {"n": 0, "by_configuration": {}}

    merged = merged.assign(
        same_sign=(merged["a"] * merged["b"] > 0),
        resolved=(merged["a"] != 0) & (merged["b"] != 0),
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
        "by_configuration": by_config,
        "contrast": contrast,
    }


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
