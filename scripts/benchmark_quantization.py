"""Compare FP16 and 4-bit NF4 inference on a CUDA GPU."""

from __future__ import annotations

import gc
import time

import torch

from small_agentic_rag.llm.model_loader import ModelLoader
from small_agentic_rag.llm.inference import TextGenerator
from small_agentic_rag.llm.quantization import QuantizationConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain briefly what a language model is."
MAX_NEW_TOKENS = 80
WARMUP_RUNS = 1
BENCHMARK_RUNS = 3


def synchronize_gpu() -> None:
    """Wait for pending CUDA operations to finish."""
    torch.cuda.synchronize()


def benchmark_configuration(
    label: str,
    quantization: QuantizationConfig,
    dtype: str,
) -> dict[str, float | str]:
    """Measure model loading, memory usage and generation latency."""

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_load = time.perf_counter()

    loader = ModelLoader(
        model_name=MODEL_NAME,
        device="cuda",
        torch_dtype=dtype,
        quantization_config=quantization,
    )
    tokenizer, model = loader.load()

    synchronize_gpu()
    load_seconds = time.perf_counter() - start_load

    memory_after_load = (
        torch.cuda.memory_allocated() / 1024**3
    )
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    generator = TextGenerator(model=model, tokenizer=tokenizer)

    # Warm-up: exclude this generation from the measurements.
    for _ in range(WARMUP_RUNS):
        generator.generate(
            PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    latencies = []

    for _ in range(BENCHMARK_RUNS):
        synchronize_gpu()
        start = time.perf_counter()

        generator.generate(
            PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        synchronize_gpu()
        latencies.append(time.perf_counter() - start)

    average_latency = sum(latencies) / len(latencies)

    results = {
        "configuration": label,
        "load_seconds": load_seconds,
        "memory_after_load_gb": memory_after_load,
        "peak_memory_gb": peak_memory,
        "average_latency_seconds": average_latency,
    }

    # Release model resources before the next configuration.
    del generator, model, tokenizer, loader
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_NAME)

    configurations = [
        (
            "FP16",
            QuantizationConfig(method="none"),
            "float16",
        ),
        (
            "4-bit NF4",
            QuantizationConfig(
                method="bitsandbytes_4bit",
                compute_dtype="float16",
                quant_type="nf4",
                use_double_quant=True,
            ),
            "float16",
        ),
    ]

    results = []

    for label, quantization, dtype in configurations:
        print(f"\nBenchmark: {label}")
        result = benchmark_configuration(
            label=label,
            quantization=quantization,
            dtype=dtype,
        )
        results.append(result)

        for metric, value in result.items():
            print(f"{metric}: {value}")

    print("\n=== COMPARISON ===")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
