
"""Configuration helpers for loading quantized language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transformers import BitsAndBytesConfig


QuantizationMethod = Literal["none", "bitsandbytes_4bit", "bitsandbytes_8bit", "awq"]


@dataclass(frozen=True)
class QuantizationConfig:
    """Describe the quantization strategy requested for a model."""

    method: QuantizationMethod = "none"
    compute_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    use_double_quant: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"

    def __post_init__(self) -> None:
        supported_methods = {
            "none",
            "bitsandbytes_4bit",
            "bitsandbytes_8bit",
            "awq",
        }

        if self.method not in supported_methods:
            raise ValueError(
                f"Unsupported quantization method: {self.method}"
            )

        if self.method == "awq":
            raise ValueError(
                "AWQ models must be loaded from a compatible pre-quantized "
                "checkpoint. Use an AWQ-compatible inference backend."
            )


def build_bitsandbytes_config(
    config: QuantizationConfig,
) -> BitsAndBytesConfig | None:
    """Build a Transformers quantization configuration.

    This config describes bitsandbytes 4-bit or 8-bit loading.
    It does not implement AWQ quantization.
    """
    if config.method == "none":
        return None

    dtype_map = {
        "float16": __import__("torch").float16,
        "bfloat16": __import__("torch").bfloat16,
        "float32": __import__("torch").float32,
    }

    if config.method == "bitsandbytes_4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map[config.compute_dtype],
            bnb_4bit_quant_type=config.quant_type,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )

    if config.method == "bitsandbytes_8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )

    raise ValueError(f"Unsupported quantization method: {config.method}")
