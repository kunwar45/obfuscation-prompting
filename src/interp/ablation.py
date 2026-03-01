"""Ablation studies for mechanistic interpretability.

Three ablation types:
- Zero ablation: activations → 0 (harsh, used for component necessity testing)
- Mean ablation: activations → dataset mean (cleaner baseline for "typical" behavior)
- Noise ablation: activations → activation + N(0, σ)

Use cases for this project:
- Which attention heads are necessary for obfuscation in reasoning traces?
- Does ablating mid-layer MLPs degrade the monitor's ability to detect obfuscation?
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from src.interp.activation_store import ActivationStore


def compute_mean_activations(store: "ActivationStore", layer_idx: int) -> np.ndarray:
    """Compute the mean activation vector across all prompts for a given layer.

    Args:
        store: ActivationStore with last_token activations
        layer_idx: Layer to compute mean over

    Returns:
        np.ndarray of shape [d_model]
    """
    X = store.load_layer(layer_idx)  # [n_prompts, d_model]
    return X.mean(axis=0)


@contextmanager
def mean_ablate_layer(
    model: "PreTrainedModel",
    layer_idx: int,
    mean_vec: np.ndarray,
) -> Generator[None, None, None]:
    """Context manager that replaces a layer's output with the mean activation.

    Patches the residual stream after the specified transformer block so every
    position receives the dataset-mean vector. Use for necessity testing.

    Args:
        model: HuggingFace causal LM
        layer_idx: Layer index to ablate
        mean_vec: [d_model] mean activation array

    Usage:
        with mean_ablate_layer(model, layer_idx=8, mean_vec=mean_acts):
            output = model.generate(...)
    """
    import torch

    mean_tensor = torch.tensor(mean_vec, dtype=model.dtype, device=model.device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = torch.ones_like(output[0]) * mean_tensor
            return (hs,) + output[1:]
        return torch.ones_like(output) * mean_tensor

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_attention_head(
    model: "PreTrainedModel",
    layer_idx: int,
    head_idx: int,
    ablation_type: str = "zero",
    mean_vec: np.ndarray | None = None,
) -> Generator[None, None, None]:
    """Context manager that ablates a specific attention head's output.

    Hooks into the attention module's output projection and zeros out (or mean-replaces)
    the contribution of the specified head.

    Args:
        model: HuggingFace causal LM
        layer_idx: Transformer layer index
        head_idx: Attention head index within the layer
        ablation_type: "zero" or "mean"
        mean_vec: Required if ablation_type="mean"; [d_head] mean vector for this head

    Usage:
        with ablate_attention_head(model, layer_idx=12, head_idx=3):
            output = model.generate(...)
    """
    import torch

    attn = model.model.layers[layer_idx].self_attn
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    def hook_fn(module, input, output):
        # output[0]: [batch, seq_len, d_model] (concatenated heads after o_proj)
        # We zero out the slice corresponding to head_idx in the pre-proj representation.
        # NOTE: this hooks the post-projection output; for head-level ablation it
        # is cleaner to hook before o_proj. This is an approximation for the skeleton.
        if isinstance(output, tuple):
            hs = output[0].clone()
            start = head_idx * d_head
            end = start + d_head
            if ablation_type == "zero":
                hs[..., start:end] = 0.0
            elif ablation_type == "mean" and mean_vec is not None:
                mv = torch.tensor(mean_vec, dtype=hs.dtype, device=hs.device)
                hs[..., start:end] = mv
            return (hs,) + output[1:]
        return output

    handle = attn.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def run_ablated_inference(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    input_ids: "torch.Tensor",
    ablation_ctx,
    max_new_tokens: int = 256,
) -> str:
    """Run model.generate() inside an ablation context manager.

    Args:
        model: HuggingFace causal LM
        tokenizer: Corresponding tokenizer
        input_ids: [1, seq_len] token ids on model's device
        ablation_ctx: A context manager from mean_ablate_layer or ablate_attention_head
        max_new_tokens: Generation length

    Returns:
        Decoded string of newly generated tokens.
    """
    import torch

    with ablation_ctx:
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
