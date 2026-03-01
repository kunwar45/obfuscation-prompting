"""Activation patching and interpretability techniques.

Three techniques implemented here:

1. Logit Lens (nostalgebraist):
   - At each layer, decode residual stream through unembedding matrix
   - Observe how target token probability evolves layer by layer
   - Implementation: project hidden_states[layer] @ model.lm_head.weight.T, take softmax

2. Causal Tracing (ROME-style, Wang et al.):
   - Run clean prompt, cache all activations
   - Run corrupted prompt (perturbed/obfuscated), patch in clean activations at each (layer, pos)
   - Recovered performance = attribution score for that (layer, pos)

3. Attribution Patching (Nanda et al.):
   - Linear approximation: score ≈ (clean_act - corrupt_act) · grad_act
   - Much faster than causal patching, scales to large models
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer


def logit_lens(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    input_ids: "torch.Tensor",
    target_token_id: int,
) -> np.ndarray:
    """Apply logit lens at each layer to track target token probability.

    At each intermediate layer, projects the residual stream through the model's
    unembedding matrix (lm_head) to get a distribution over vocabulary. Returns
    the softmax probability of target_token_id at every layer.

    Args:
        model: HuggingFace causal LM with output_hidden_states support
        tokenizer: Corresponding tokenizer (unused currently, kept for API consistency)
        input_ids: [1, seq_len] token ids on model's device
        target_token_id: Vocabulary index of the token to track

    Returns:
        np.ndarray of shape [n_layers + 1] with probabilities per layer
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states  # tuple of [1, seq_len, d_model]

    # Access unembedding matrix (works for Gemma/Llama/Qwen — all use lm_head)
    unembed = model.lm_head.weight  # [vocab_size, d_model]

    probs = []
    for hs in hidden_states:
        last = hs[0, -1, :]  # [d_model]
        # Optional: apply model's final layer norm if available
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            last = model.model.norm(last)
        logits = last @ unembed.T  # [vocab_size]
        prob = F.softmax(logits, dim=-1)[target_token_id].item()
        probs.append(prob)

    return np.array(probs)


def causal_patch(
    model: "PreTrainedModel",
    clean_cache: dict[int, "torch.Tensor"],
    corrupted_input_ids: "torch.Tensor",
    metric_fn,
    layers: list[int] | None = None,
) -> np.ndarray:
    """Causal tracing: measure recovered performance when patching clean acts into corrupted run.

    For each (layer, position) pair, runs the corrupted prompt but patches in the clean
    cached activation. The attribution score is how much performance recovers.

    Args:
        model: HuggingFace causal LM
        clean_cache: dict mapping layer_idx -> [1, seq_len, d_model] clean activations
        corrupted_input_ids: [1, seq_len] corrupted token ids on model's device
        metric_fn: Callable([logits]) -> float, measures performance (e.g., target token prob)
        layers: Which layers to patch (default: all available in clean_cache)

    Returns:
        np.ndarray of shape [n_layers, seq_len] with attribution scores
    """
    import torch

    if layers is None:
        layers = sorted(clean_cache.keys())

    seq_len = corrupted_input_ids.shape[1]
    scores = np.zeros((len(layers), seq_len))

    model.eval()
    for li, layer_idx in enumerate(layers):
        clean_act = clean_cache[layer_idx]  # [1, seq_len, d_model]
        for pos in range(seq_len):
            hooks = []

            def make_hook(layer, position, act):
                def hook_fn(module, input, output):
                    # output may be tuple; patch the hidden state at this position
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        hs[0, position, :] = act[0, position, :]
                        return (hs,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, position, :] = act[0, position, :]
                        return out
                return hook_fn

            hook = model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx, pos, clean_act)
            )
            hooks.append(hook)

            with torch.no_grad():
                out = model(corrupted_input_ids, return_dict=True)
            scores[li, pos] = metric_fn(out.logits)

            for h in hooks:
                h.remove()

    return scores


def attribution_patch(
    model: "PreTrainedModel",
    clean_cache: dict[int, "torch.Tensor"],
    corrupted_input_ids: "torch.Tensor",
    metric_fn,
    layers: list[int] | None = None,
) -> np.ndarray:
    """Attribution patching (linear approximation of causal patching).

    Much faster than full causal tracing. Approximates the causal effect as:
        score(layer, pos) ≈ (clean_act - corrupt_act)[pos] · grad[pos]

    where grad is the gradient of the metric w.r.t. the corrupted activation.

    Args:
        model: HuggingFace causal LM
        clean_cache: dict mapping layer_idx -> [1, seq_len, d_model] clean activations
        corrupted_input_ids: [1, seq_len] corrupted token ids on model's device
        metric_fn: Callable([logits]) -> scalar tensor, differentiable metric
        layers: Which layers to attribute (default: all available in clean_cache)

    Returns:
        np.ndarray of shape [n_layers, seq_len] with attribution scores
    """
    import torch

    if layers is None:
        layers = sorted(clean_cache.keys())

    seq_len = corrupted_input_ids.shape[1]
    scores = np.zeros((len(layers), seq_len))

    model.eval()

    # Collect corrupt activations and gradients via hooks
    corrupt_acts: dict[int, torch.Tensor] = {}
    grads: dict[int, torch.Tensor] = {}
    hooks = []

    def save_act(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.detach().requires_grad_(True)
            corrupt_acts[layer_idx] = hs
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook_fn

    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(save_act(layer_idx))
        hooks.append(h)

    out = model(corrupted_input_ids, return_dict=True)
    metric = metric_fn(out.logits)
    metric.backward()

    for h in hooks:
        h.remove()

    for li, layer_idx in enumerate(layers):
        if layer_idx in corrupt_acts and corrupt_acts[layer_idx].grad is not None:
            diff = (clean_cache[layer_idx] - corrupt_acts[layer_idx]).detach().cpu().numpy()
            grad = corrupt_acts[layer_idx].grad.cpu().numpy()
            scores[li] = (diff * grad).sum(axis=-1)[0]  # dot product per position

    return scores
