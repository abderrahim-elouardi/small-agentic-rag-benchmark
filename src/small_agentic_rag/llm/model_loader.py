
"""Load pretrained causal language models, with optional quantization."""

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from small_agentic_rag.llm.quantization import (
    QuantizationConfig,
    build_bitsandbytes_config,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load a Hugging Face causal language model and tokenizer."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        quantization_config: QuantizationConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.torch_dtype = torch_dtype
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.quantization_config = (
            quantization_config or QuantizationConfig()
        )

        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None

        self._validate_device()

    def _validate_device(self) -> None:
        """Validate the requested execution device."""
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but no CUDA GPU is available."
                )
        elif self.device != "cpu":
            raise ValueError(
                f"Unsupported device: {self.device}. "
                "Use 'cpu' or 'cuda'."
            )

    def _resolve_dtype(self) -> torch.dtype | str:
        """Resolve the requested numerical precision."""
        if self.torch_dtype is None:
            return (
                torch.float16
                if self.device.startswith("cuda")
                else torch.float32
            )

        if isinstance(self.torch_dtype, torch.dtype):
            return self.torch_dtype

        supported_dtypes = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "auto": "auto",
        }

        dtype = supported_dtypes.get(self.torch_dtype.lower())

        if dtype is None:
            raise ValueError(
                f"Unsupported dtype: {self.torch_dtype}. "
                f"Choose from {list(supported_dtypes)}."
            )

        if (
            dtype == torch.bfloat16
            and self.device.startswith("cuda")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "The selected GPU does not support bfloat16."
            )

        return dtype

    def load(
        self,
    ) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """Load the tokenizer and model for inference."""
        common_kwargs = {
            "trust_remote_code": self.trust_remote_code,
        }

        if self.revision is not None:
            common_kwargs["revision"] = self.revision

        logger.info("Loading tokenizer: %s", self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **common_kwargs,
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer has no padding token or EOS token."
                )

        quantization = self.quantization_config
        bnb_config = build_bitsandbytes_config(quantization)

        load_kwargs = {
            **common_kwargs,
            "dtype": self._resolve_dtype(),
            "low_cpu_mem_usage": True,
        }

        if bnb_config is not None:
            if not self.device.startswith("cuda"):
                raise RuntimeError(
                    "This bitsandbytes loading configuration requires "
                    "a CUDA GPU in this project setup. Use Google Colab "
                    "with a compatible GPU."
                )

            logger.info(
                "Loading with quantization method: %s",
                quantization.method,
            )

            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"

        logger.info("Loading model: %s", self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Quantized models are placed by the quantization backend.
        # Standard models are moved to the requested device explicitly.
        if bnb_config is None:
            self.model.to(self.device)

        self.model.eval()

        logger.info(
            "Model loaded successfully. Device=%s, dtype=%s, quantization=%s",
            self.device,
            next(self.model.parameters()).dtype,
            quantization.method,
        )

        return self.tokenizer, self.model

