"""Sparse Autoencoder (SAE) feature decomposition for mechanistic interpretability.

Pre-trained SAEs available via sae-lens (EleutherAI):
- Gemma-2-2B: sae-lens has trained SAEs for many layers
- Llama-3.1-8B: available on HuggingFace
- Qwen: may need custom training

Workflow:
1. Load pre-trained SAE for a specific layer
2. Encode residual stream: sae_latents = sae.encode(hidden_states)  # sparse [n_features]
3. Find features that distinguish obfuscated vs normal reasoning
4. Feature steering: inject/suppress features during inference via forward hooks

Install dependency:
    pip install sae-lens

Note: sae-lens is NOT in requirements.txt by default since it's optional and
      model-specific. Add it when you have a target model with a trained SAE.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


def load_pretrained_sae(model_name: str, layer_idx: int, device: str = "cpu"):
    """Load a pre-trained SAE for a specific model and layer via sae-lens.

    Args:
        model_name: HuggingFace model identifier, e.g. "google/gemma-2-2b-it"
        layer_idx: Transformer layer whose residual stream the SAE was trained on
        device: PyTorch device string ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        SAE object from sae-lens with .encode() and .decode() methods.

    Example:
        sae = load_pretrained_sae("google/gemma-2-2b-it", layer_idx=12, device="cuda")
    """
    try:
        from sae_lens import SAE
    except ImportError:
        raise ImportError(
            "sae-lens is required for SAE analysis. Install with: pip install sae-lens"
        )

    # sae-lens release IDs are model-specific; update these as SAE libraries evolve.
    # See https://github.com/EleutherAI/sae-lens for available SAEs.
    release_map = {
        "google/gemma-2-2b": "gemma-scope-2b-pt-res",
        "google/gemma-2-2b-it": "gemma-scope-2b-pt-res",
        "meta-llama/Llama-3.1-8B": "llama_scope_lxr_8x",
    }
    release = release_map.get(model_name)
    if release is None:
        raise ValueError(
            f"No known SAE release for model {model_name!r}. "
            "Check https://github.com/EleutherAI/sae-lens for available SAEs."
        )

    sae_id = f"layer_{layer_idx}/width_16k/average_l0_71"  # example; adjust per release
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae


def encode_activations(sae, activations: np.ndarray) -> np.ndarray:
    """Encode a batch of residual-stream activations through the SAE.

    Args:
        sae: Loaded SAE from sae-lens
        activations: [n_prompts, d_model] float32 array

    Returns:
        [n_prompts, n_features] sparse float32 array of SAE latent activations
    """
    import torch

    x = torch.tensor(activations, dtype=torch.float32, device=sae.device)
    with torch.no_grad():
        latents = sae.encode(x)  # [n_prompts, n_features]
    return latents.cpu().numpy()


def top_activating_features(
    sae_acts: np.ndarray,
    k: int = 10,
    method: str = "mean",
) -> list[int]:
    """Find the k features with highest activation across the dataset.

    Args:
        sae_acts: [n_prompts, n_features] SAE latent array
        k: Number of top features to return
        method: "mean" (average activation) or "max" (peak activation)

    Returns:
        List of feature indices sorted by descending activation strength
    """
    if method == "mean":
        scores = sae_acts.mean(axis=0)
    elif method == "max":
        scores = sae_acts.max(axis=0)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'mean' or 'max'.")

    top_k = np.argsort(scores)[::-1][:k]
    return top_k.tolist()


@contextmanager
def steer_feature(
    model: "PreTrainedModel",
    sae,
    layer_idx: int,
    feature_idx: int,
    scale: float,
) -> Generator[None, None, None]:
    """Context manager for feature steering via forward hook.

    Injects or suppresses a SAE feature in the residual stream during generation.
    A positive scale amplifies the feature; negative scale suppresses it.

    Args:
        model: HuggingFace causal LM
        sae: Loaded SAE from sae-lens
        layer_idx: Layer at which to inject the feature
        feature_idx: SAE feature index to steer
        scale: Scalar multiplier for the feature direction (positive=amplify, negative=suppress)

    Usage:
        with steer_feature(model, sae, layer_idx=12, feature_idx=42, scale=20.0):
            output = model.generate(...)
    """
    import torch

    # SAE decoder columns are feature directions in residual-stream space
    feature_dir = sae.W_dec[feature_idx]  # [d_model]
    feature_dir = feature_dir.to(dtype=model.dtype, device=model.device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0] + scale * feature_dir
            return (hs,) + output[1:]
        return output + scale * feature_dir

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
