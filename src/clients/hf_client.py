"""HuggingFace local model client with optional activation capture."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFClient:
    """Model-agnostic HuggingFace causal LM client.

    Supports Gemma, Qwen, Llama, and any AutoModelForCausalLM-compatible model.
    Device priority: CUDA (multi-GPU via device_map="auto") > MPS > CPU.

    Activation capture is performed as a second forward pass after generation,
    so activations are position-stable over the full prompt+response sequence.

    capture_mode:
        "none"           — no activation capture (default)
        "last_token"     — saves hidden_states[layer][0, -1, :] → [n_layers+1, d_model]
        "full_sequence"  — saves full hidden_states[layer][0, :, :] → [n_layers+1, seq_len, d_model]
        "reasoning_span" — saves positions between <reasoning>…</reasoning> tokens only
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        capture_mode: str = "none",
    ):
        self.model_name = model_name
        self.capture_mode = capture_mode
        self.device = detect_device()

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # MPS doesn't support bfloat16 reliably for all ops — downgrade to float16
        if self.device == "mps" and torch_dtype == torch.bfloat16:
            print("MPS detected: downgrading bfloat16 → float16 for compatibility")
            torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda":
            # device_map="auto" handles multi-GPU and CPU offload on CUDA
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        else:
            # MPS and CPU: load on CPU first, then move (device_map="auto" doesn't support MPS)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"Model loaded on device: {self.device} | dtype: {torch_dtype}")

        self._last_activations: Optional[dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Public interface (matches TogetherClient.chat signature)
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the loaded model.

        The `model` param is accepted for interface compatibility but ignored —
        the model is already loaded at __init__ time.
        """
        prompt = self._apply_chat_template(messages)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
            )

        # Decode only the newly generated tokens
        new_token_ids = output_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # Capture activations on the full prompt+response sequence if requested
        if self.capture_mode != "none":
            full_ids = output_ids[0:1]  # [1, full_seq_len]
            self._capture_activations(full_ids)

        return response

    # ------------------------------------------------------------------
    # Activation capture
    # ------------------------------------------------------------------

    def _capture_activations(self, input_ids: torch.Tensor) -> None:
        """Run a second forward pass and store hidden states."""
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states  # tuple of [1, seq_len, d_model]
        token_ids = input_ids[0].cpu().numpy()

        activations: dict[str, np.ndarray] = {"token_ids": token_ids}

        if self.capture_mode == "last_token":
            for i, hs in enumerate(hidden_states):
                activations[f"layer_{i}"] = hs[0, -1, :].float().cpu().numpy()

        elif self.capture_mode == "full_sequence":
            for i, hs in enumerate(hidden_states):
                activations[f"layer_{i}"] = hs[0, :, :].float().cpu().numpy()

        elif self.capture_mode == "reasoning_span":
            start, end = self._find_reasoning_span(token_ids)
            for i, hs in enumerate(hidden_states):
                activations[f"layer_{i}"] = hs[0, start:end, :].float().cpu().numpy()

        self._last_activations = activations

    def _find_reasoning_span(self, token_ids: np.ndarray) -> tuple[int, int]:
        """Find token indices for <reasoning>…</reasoning> span."""
        open_ids = self.tokenizer.encode("<reasoning>", add_special_tokens=False)
        close_ids = self.tokenizer.encode("</reasoning>", add_special_tokens=False)

        ids_list = token_ids.tolist()
        start = self._find_subsequence(ids_list, open_ids)
        end = self._find_subsequence(ids_list, close_ids)

        if start == -1:
            start = 0
        else:
            start += len(open_ids)

        if end == -1:
            end = len(ids_list)

        return start, end

    @staticmethod
    def _find_subsequence(seq: list[int], subseq: list[int]) -> int:
        """Return start index of first occurrence of subseq in seq, or -1."""
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i : i + m] == subseq:
                return i
        return -1

    def save_activations(self, prompt_id: str, out_dir: str) -> str:
        """Save last captured activations to a .npz file.

        Returns the path to the saved file, or "" if no activations are available.
        """
        if self._last_activations is None:
            return ""

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prompt_id}.npz")
        np.savez_compressed(path, **self._last_activations)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Apply the tokenizer chat template if available, else fall back to manual format."""
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role.upper()}] {content}")
        parts.append("[ASSISTANT]")
        return "\n".join(parts)

    def _get_layer_count(self) -> int:
        """Return number of transformer layers (works for Gemma/Qwen/Llama/etc.)."""
        return self.model.config.num_hidden_layers
