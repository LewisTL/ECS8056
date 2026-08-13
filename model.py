"""
model.py: environment-aware OpenVLA-7B loading, action prediction, and run logging.

Dataset-agnostic: loads the model and turns a (PIL image, instruction) pair into a
7-DoF action vector, and records the conditions each prediction was made under.

Design choices:
  * 4-bit (NF4) quantisation by default. OpenVLA-7B needs ~16.8 GB in bfloat16,
    which does not fit a Colab T4 (16 GB). 4-bit fits in ~7 GB with no measurable
    loss in BridgeData V2 success rate (Kim et al., 2024, Table 2).
  * attn_implementation="eager" by default. FlashAttention-2 requires an
    Ampere-or-newer GPU (A100/L4); a T4 (Turing) cannot use it.
  * Precision is FIXED and explicit (default bf16), not silently derived from the
    GPU. OpenVLA takes the argmax over discretised action bins, so changing dtype
    between runs can flip a near-boundary bin and fake a directional difference,
    which is exactly the response the probe measures. The bf16 path refuses to
    fall back.
  * Seeds are fixed centrally and run conditions are logged per prediction for
    reproducibility.
  * Two readouts are available. `predict_action` returns the argmax action the
    model would execute. `predict_action_dist` additionally returns a continuous
    expected-bin value, which resolves directional differences smaller than one
    action bin (see the continuous readout section below).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from action_bins import (
    EMPTY_TOKEN_ID,
    ActionReadout,
    action_token_ids,
    bin_widths,
    readout_from_logits,
)
from prediction_log import append_row

MODEL_ID = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"  # action de-normalisation stats for BridgeData V2
DEFAULT_SEED = 42
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #
@dataclass
class GpuProfile:
    name: str
    capability: tuple[int, int]
    total_gb: float

    @property
    def supports_bf16(self) -> bool:
        # Native bf16 + FlashAttention-2 both require compute capability >= 8.0.
        return self.capability >= (8, 0)


def detect_gpu() -> GpuProfile:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU visible. In Colab: Runtime > Change runtime type > GPU. "
            "Prefer an L4 or A100 runtime; a T4 also works with 4-bit."
        )
    props = torch.cuda.get_device_properties(0)
    return GpuProfile(
        name=props.name,
        capability=(props.major, props.minor),
        total_gb=props.total_memory / (1024**3),
    )


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix all RNGs touched during inference for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_openvla(
    quantize_4bit: bool = True,
    use_flash_attn: bool = False,
    precision: str = "bf16",
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
):
    """Load the OpenVLA-7B processor and model with a fixed, explicit precision.

    Args:
        quantize_4bit: load weights in 4-bit NF4 (default; required on a T4).
        use_flash_attn: only honoured on Ampere+ GPUs with flash-attn installed.
        precision: one of {"bf16", "fp16", "fp32"}. Use ONE value for all
            evaluation runs so results stay comparable. Default "bf16" (A100/L4).
        seed: global RNG seed.
        verbose: print a short environment + load summary.

    Returns:
        (processor, vla, compute_dtype)
    """
    set_seed(seed)
    gpu = detect_gpu()

    if precision not in DTYPES:
        raise ValueError(f"precision must be one of {list(DTYPES)}, got {precision!r}")
    compute_dtype = DTYPES[precision]

    # Refuse to silently change precision between sessions.
    if precision == "bf16" and not gpu.supports_bf16:
        raise RuntimeError(
            f"{gpu.name} (sm_{gpu.capability[0]}{gpu.capability[1]}) cannot do bf16. "
            "Use an A100/L4 for evaluation runs, or pass precision='fp16' for a pilot, "
            "but fp16 results are NOT comparable to bf16."
        )

    attn = "eager"
    if use_flash_attn:
        if gpu.supports_bf16:
            attn = "flash_attention_2"
        elif verbose:
            print(f"[load_openvla] {gpu.name} cannot use FlashAttention-2; using eager.")

    if verbose:
        print(
            f"[load_openvla] GPU: {gpu.name} (sm_{gpu.capability[0]}{gpu.capability[1]}, "
            f"{gpu.total_gb:.1f} GB) | precision={precision} | attn={attn} | "
            f"4bit={quantize_4bit}"
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    load_kwargs = dict(
        attn_implementation=attn,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        # With bitsandbytes, the model is placed by device_map; do NOT call .to().
        load_kwargs["device_map"] = {"": 0}
        vla = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **load_kwargs)
    else:
        vla = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **load_kwargs).to("cuda:0")

    vla.eval()
    if verbose:
        alloc = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"[load_openvla] Loaded. GPU memory allocated: {alloc:.2f} GB")
    return processor, vla, compute_dtype


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


@torch.inference_mode()
def predict_action(
    processor,
    vla,
    image: Image.Image,
    instruction: str,
    compute_dtype: torch.dtype = torch.bfloat16,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
    do_sample: bool = False,
) -> np.ndarray:
    """Return the de-normalised 7-DoF action vector for one (image, instruction).

    Layout is [dx, dy, dz, droll, dpitch, dyaw, gripper]. For the spatial-grounding
    probe the translation deltas (dx, dy, dz) are the signal: a correct response to
    flipping "left" <-> "right" should flip the sign of the relevant axis.
    do_sample=False gives the deterministic argmax action.
    """
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image.convert("RGB")).to("cuda:0", dtype=compute_dtype)
    # predict_action appends a token to input_ids; the processor's attention_mask
    # would then be one short, causing an off-by-one in the causal mask. Drop it
    # (batch=1, no padding) so generate rebuilds a correctly sized mask.
    inputs.pop("attention_mask", None)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=do_sample)
    return np.asarray(action, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Continuous action readout
# --------------------------------------------------------------------------- #
# The decoding arithmetic lives in action_bins.py, which is free of torch so it
# can be tested without a GPU. This section holds only the inference path that
# produces the logits, and the gate that confirms the reimplemented decoding
# reproduces the executable action.


def describe_action_space(vla, unnorm_key: str = DEFAULT_UNNORM_KEY) -> dict:
    """Report the decoding constants the continuous readout depends on.

    These live in OpenVLA's remote modelling code and can move between
    revisions, so they are read from the loaded model rather than assumed.
    """
    stats = vla.get_action_stats(unnorm_key)
    n_bins = int(np.asarray(vla.bin_centers).shape[0])
    ids = action_token_ids(int(vla.vocab_size), n_bins)
    return {
        "unnorm_key": unnorm_key,
        "action_dim": int(vla.get_action_dim(unnorm_key)),
        "n_bins": n_bins,
        "vocab_size": int(vla.vocab_size),
        "bin_center_first": float(np.asarray(vla.bin_centers)[0]),
        "bin_center_last": float(np.asarray(vla.bin_centers)[-1]),
        "action_token_id_min": int(ids.min()),
        "action_token_id_max": int(ids.max()),
        "q01": list(np.asarray(stats["q01"], dtype=float)),
        "q99": list(np.asarray(stats["q99"], dtype=float)),
        "mask": list(np.asarray(stats.get("mask", np.ones(len(stats["q01"]), dtype=bool)))),
        # Width of one bin in dataset units, the resolution floor of the argmax
        # readout on each dimension.
        "bin_width": list(bin_widths(stats, n_bins)),
    }


@torch.inference_mode()
def predict_action_dist(
    processor,
    vla,
    image: Image.Image,
    instruction: str,
    compute_dtype: torch.dtype = torch.bfloat16,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
) -> ActionReadout:
    """Return the argmax action and the continuous expected-bin action.

    Decoding is greedy, matching `predict_action`, and the returned `action`
    field reproduces it exactly; `verify_readout` asserts that equality on real
    inputs. The expected value is taken over the model's distribution at each
    emission step, restricted to the action-token slice and renormalised.

    The scores captured from `generate` are post-processing logits. Greedy
    decoding applies no warpers, so they equal the raw logits unless a logits
    processor is configured; `action_mass` surfaces the case where processing
    has moved weight off the action vocabulary.
    """
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image.convert("RGB")).to("cuda:0", dtype=compute_dtype)
    # The processor's attention_mask is one short of the extended input_ids
    # below, which would misalign the causal mask. Batch is 1 and unpadded, so
    # dropping it lets generate rebuild a correctly sized mask.
    inputs.pop("attention_mask", None)
    input_ids = inputs.pop("input_ids")
    if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
        pad = torch.tensor([[EMPTY_TOKEN_ID]], dtype=input_ids.dtype,
                           device=input_ids.device)
        input_ids = torch.cat((input_ids, pad), dim=1)

    action_dim = vla.get_action_dim(unnorm_key)
    generated = vla.generate(
        input_ids,
        max_new_tokens=action_dim,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        **inputs,
    )

    return readout_from_logits(
        token_ids=generated.sequences[0, -action_dim:].cpu().numpy(),
        logits=torch.stack(generated.scores, dim=0)[:, 0, :].float().cpu().numpy(),
        bin_centers=vla.bin_centers,
        vocab_size=int(vla.vocab_size),
        action_norm_stats=vla.get_action_stats(unnorm_key),
    )


def verify_readout(
    processor,
    vla,
    samples,
    compute_dtype: torch.dtype = torch.bfloat16,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
    verbose: bool = True,
) -> dict:
    """Check that the continuous path reproduces the argmax action exactly.

    The continuous readout reimplements the decoding upstream performs inside
    `predict_action`, against constants that can move between model revisions.
    Reproducing the executable action bit for bit on real inputs is the
    condition under which the expected-bin value may be trusted.

    Args:
        samples: iterable of (image, instruction) pairs.

    Returns a summary with the number checked, the number matching, the largest
    absolute deviation, and the smallest action-token mass observed. Raises
    RuntimeError when any sample disagrees.
    """
    checked = 0
    mismatches = []
    max_dev = 0.0
    min_mass = 1.0
    for image, instruction in samples:
        baseline = predict_action(processor, vla, image, instruction, compute_dtype,
                                  unnorm_key=unnorm_key, do_sample=False)
        readout = predict_action_dist(processor, vla, image, instruction, compute_dtype,
                                      unnorm_key=unnorm_key)
        deviation = float(np.max(np.abs(baseline - readout.action)))
        max_dev = max(max_dev, deviation)
        min_mass = min(min_mass, float(np.min(readout.action_mass)))
        if deviation > 0.0:
            mismatches.append((instruction, deviation))
        checked += 1

    summary = {
        "checked": checked,
        "matched": checked - len(mismatches),
        "max_abs_deviation": max_dev,
        "min_action_mass": min_mass,
    }
    if verbose:
        print(f"[verify_readout] {summary['matched']}/{checked} exact | "
              f"max deviation {max_dev:.3e} | min action-token mass {min_mass:.4f}")
    if mismatches:
        raise RuntimeError(
            f"continuous readout disagreed with predict_action on "
            f"{len(mismatches)}/{checked} samples (max deviation {max_dev:.3e}). "
            "The decoding constants in OpenVLA's remote code have most likely "
            "changed; re-check describe_action_space before using c* columns."
        )
    return summary


# --------------------------------------------------------------------------- #
# Reproducibility: run conditions + CSV logging
# --------------------------------------------------------------------------- #
def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def run_metadata(compute_dtype: torch.dtype, seed: int = DEFAULT_SEED) -> dict:
    """Conditions every prediction should be stamped with.

    The keys are `prediction_log.RUN_METADATA_FIELDS`, in that order, which a test
    asserts so the declared schema cannot drift from what is actually written.
    """
    gpu = detect_gpu()
    return {
        "gpu_name": gpu.name,
        "gpu_capability": f"sm_{gpu.capability[0]}{gpu.capability[1]}",
        "dtype": str(compute_dtype).replace("torch.", ""),
        "seed": seed,
        "torch": torch.__version__,
        "transformers": _pkg("transformers"),
        "bitsandbytes": _pkg("bitsandbytes"),
    }




def append_prediction_log(
    csv_path: str,
    action: np.ndarray,
    instruction: str,
    meta: dict,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
    do_sample: bool = False,
    readout: ActionReadout | None = None,
    **extra,
) -> dict:
    """Append one prediction plus its run conditions to a CSV.

    Rows are written against the file's own header. A row carrying columns the
    header lacks widens the file rather than being written at its own width: an
    over-wide row is silently accepted by the writer and only fails later, when a
    reader meets a line with more fields than the header promised.

    Columns absent from a given row are written empty, so a log may mix rows from
    different call shapes. That is what allows predictions logged without a
    continuous readout, including rows migrated from an earlier schema, to sit in
    the same file as rows that have one.

    When `readout` is supplied, the continuous expected-bin action (`c0` to
    `c6`), the argmax bin indices (`b0` to `b6`), the per-dimension bin
    confidence (`p0` to `p6`) and entropy (`h0` to `h6`), and the minimum
    action-token mass are written alongside the argmax action. The `a*` columns
    keep their meaning either way, so logs from before the continuous readout
    remain comparable.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instruction": instruction,
        "unnorm_key": unnorm_key,
        "do_sample": do_sample,
        **meta,
        **{f"a{i}": float(action[i]) for i in range(len(action))},
        **(readout.to_log_fields() if readout is not None else {}),
        **extra,
    }
    return append_row(csv_path, row)


if __name__ == "__main__":
    proc, model, dtype = load_openvla(quantize_4bit=True, precision="bf16")
    meta = run_metadata(dtype)
    dummy = Image.fromarray(
        (np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8)
    )
    act = predict_action(proc, model, dummy, "pick up the object on the left", dtype)
    append_prediction_log("outputs/predictions.csv", act,
                          "pick up the object on the left", meta)
    print("Action:", np.round(act, 4))
    assert act.shape == (7,), "Expected a 7-DoF action vector"
    print("Smoke test passed; logged to outputs/predictions.csv")