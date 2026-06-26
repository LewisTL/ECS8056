"""
model.py — Environment-aware OpenVLA-7B loading, action prediction, and run logging.

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
    between runs can flip a near-boundary bin and fake a directional difference —
    the exact signal this study measures. The bf16 path refuses to fall back.
  * Seeds fixed centrally and run conditions logged per prediction (plan R5).
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

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
    """Fix all RNGs touched during inference (plan R5: reproducibility)."""
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
            "Use an A100/L4 for evaluation runs, or pass precision='fp16' for a pilot "
            "— but fp16 results are NOT comparable to bf16."
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
# Reproducibility: run conditions + CSV logging (plan R5)
# --------------------------------------------------------------------------- #
def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def run_metadata(compute_dtype: torch.dtype, seed: int = DEFAULT_SEED) -> dict:
    """Conditions every prediction should be stamped with (plan R5)."""
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
    **extra,
) -> dict:
    """Append one prediction plus its run conditions to a CSV.

    The header is written from the first row's keys, so keep the columns
    consistent: when you add probe fields later (scene_id, distractor_count,
    spatial_term), pass them via **extra in the SAME order every time.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instruction": instruction,
        "unnorm_key": unnorm_key,
        "do_sample": do_sample,
        **meta,
        **{f"a{i}": float(action[i]) for i in range(len(action))},
        **extra,
    }
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


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