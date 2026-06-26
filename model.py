"""
model.py — Environment-aware OpenVLA-7B loading and action prediction.

This module is deliberately dataset-agnostic: it loads the model and 
turns a (PIL image, instruction) pair into a 7-DoF action vector.

Design choices:

  * 4-bit (NF4) quantisation by default. OpenVLA-7B needs ~16.8 GB in bfloat16,
    which does not fit a Colab T4 (16 GB). 4-bit fits in ~7 GB with no measurable
    loss in BridgeData V2 success rate (Kim et al., 2024, Table 2).
  * attn_implementation="eager" by default. FlashAttention-2 requires an
    Ampere-or-newer GPU (A100/L4); a T4 (Turing) cannot use it. Eager runs
    everywhere; for inference-only probing the speed cost is acceptable.
  * Compute dtype follows the GPU: bfloat16 on Ampere+, float16 on Turing (T4),
    because T4 has poor native bf16 support and mixed dtypes cause runtime errors.
  * Seeds are fixed centrally for reproducibility.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"  # action de-normalisation stats for BridgeData V2
DEFAULT_SEED = 42


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
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
):
    """Load the OpenVLA-7B processor and model, adapting to the available GPU.

    Args:
        quantize_4bit: load weights in 4-bit NF4 (default; required on a T4).
        use_flash_attn: only honoured on Ampere+ GPUs with flash-attn installed.
        seed: global RNG seed.
        verbose: print a short environment + load summary.

    Returns:
        (processor, vla, compute_dtype)
    """
    set_seed(seed)
    gpu = detect_gpu()
    compute_dtype = torch.bfloat16 if gpu.supports_bf16 else torch.float16

    attn = "eager"
    if use_flash_attn:
        if gpu.supports_bf16:
            attn = "flash_attention_2"
        elif verbose:
            print(f"[load_openvla] {gpu.name} cannot use FlashAttention-2; using eager.")

    if verbose:
        print(
            f"[load_openvla] GPU: {gpu.name} (sm_{gpu.capability[0]}{gpu.capability[1]}, "
            f"{gpu.total_gb:.1f} GB) | compute_dtype={compute_dtype} | "
            f"attn={attn} | 4bit={quantize_4bit}"
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

    The vector layout is [dx, dy, dz, droll, dpitch, dyaw, gripper]. For the
    spatial-grounding probe, the translation deltas (dx, dy, dz) are the signal:
    a correct response to flipping "left" <-> "right" should flip the sign of the
    relevant axis. do_sample=False gives the deterministic argmax action.
    """
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image.convert("RGB")).to("cuda:0", dtype=compute_dtype)
    # predict_action appends a token to input_ids; the processor's attention_mask
    # would then be one short, causing an off-by-one in the causal mask. Drop it
    # (batch=1, no padding) so generate rebuilds a correctly sized mask.
    inputs.pop("attention_mask", None)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=do_sample)
    return np.asarray(action, dtype=np.float32)


if __name__ == "__main__":
    # Minimal smoke test (synthetic frame). Verifies the load + inference path
    # without needing BridgeData V2 yet. This is the core of the M2 check.
    proc, model, dtype = load_openvla(quantize_4bit=True)
    dummy = Image.fromarray(
        (np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8)
    )
    act = predict_action(proc, model, dummy, "pick up the object on the left", dtype)
    print("Action shape:", act.shape)        # expect (7,)
    print("Action vector:", np.round(act, 4))
    assert act.shape == (7,), "Expected a 7-DoF action vector"
    print("Smoke test passed.")
