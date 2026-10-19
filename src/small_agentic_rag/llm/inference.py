
"""Text generation utilities for pretrained causal language models."""

from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TextGenerator:
    """Generate text using a loaded causal language model."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        max_new_tokens: int = 256,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be greater than zero.")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        do_sample: bool = False,
    ) -> str:
        """Generate a continuation for a text prompt."""
        if not prompt.strip():
            raise ValueError("The prompt cannot be empty.")

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        token_limit = max_new_tokens or self.max_new_tokens
        if token_limit < 1:
            raise ValueError("max_new_tokens must be greater than zero.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": token_limit,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

        self.model.eval()

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        # Decode only the newly generated tokens.
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return response.strip()
