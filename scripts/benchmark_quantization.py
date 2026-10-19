"""Benchmark FP16 vs 4-bit NF4 for causal language models."""

from __future__ import annotations

import csv
import gc
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import bitsandbytes
import torch
import transformers

from small_agentic_rag.llm.model_loader import ModelLoader
from small_agentic_rag.llm.quantization import QuantizationConfig


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


PROMPT = "Explain briefly what a language model is."
MAX_NEW_TOKENS = 80
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "quantization_benchmark.csv"


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def synchronize_gpu() -> None:
    """Wait until all queued CUDA operations are complete."""
    torch.cuda.synchronize()


def gib(value: int) -> float:
    """Convert bytes to GiB."""
    return value / (1024**3)


def generate_once(model, tokenizer) -> int:
    """Generate text and return the number of new tokens."""
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return output_ids.shape[1] - inputs["input_ids"].shape[1]


# --------------------------------------------------
# Single configuration benchmark
# --------------------------------------------------

def benchmark_configuration(
    label: str,
    quantization: QuantizationConfig,
) -> dict:
    """Measure loading, memory usage and generation performance."""

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Loading time includes tokenizer and model initialization.
    load_start = time.perf_counter()

    loader = ModelLoader(
        model_name=MODEL_NAME,
        device="cuda",
        torch_dtype="float16",
        quantization_config=quantization,
    )

    tokenizer, model = loader.load()

    synchronize_gpu()
    load_seconds = time.perf_counter() - load_start

    memory_after_load = gib(torch.cuda.memory_allocated())
    peak_load_memory = gib(torch.cuda.max_memory_allocated())

    # Warm-up executions are excluded from latency statistics.
    for _ in range(WARMUP_RUNS):
        generate_once(model, tokenizer)

    synchronize_gpu()

    # Measure inference memory independently of loading.
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    generated_token_counts = []

    for _ in range(BENCHMARK_RUNS):
        synchronize_gpu()
        start = time.perf_counter()

        generated_tokens = generate_once(model, tokenizer)

        synchronize_gpu()
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        generated_token_counts.append(generated_tokens)

    peak_inference_memory = gib(
        torch.cuda.max_memory_allocated()
    )

    throughputs = [
        tokens / elapsed
        for tokens, elapsed in zip(
            generated_token_counts, latencies
        )
        if elapsed > 0
    ]

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "configuration": label,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "bitsandbytes_version": bitsandbytes.__version__,
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "load_seconds": load_seconds,
        "memory_after_load_gb": memory_after_load,
        "peak_load_memory_gb": peak_load_memory,
        "peak_inference_memory_gb": peak_inference_memory,
        "mean_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "mean_generated_tokens": statistics.mean(
            generated_token_counts
        ),
        "mean_tokens_per_second": statistics.mean(throughputs),
        "median_tokens_per_second": statistics.median(throughputs),
    }

    # Release resources before loading the next configuration.
    del model, tokenizer, loader
    gc.collect()
    torch.cuda.empty_cache()

    return result


# --------------------------------------------------
# Results export
# --------------------------------------------------

def save_results(results: list[dict]) -> None:
    """Append benchmark results to a CSV file."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_FILE.exists()
    fieldnames = list(results[0].keys())

    with RESULTS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_FILE}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires a CUDA-compatible GPU."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory
        / (1024**3), 2
    ), "GiB")
    print("Model:", MODEL_NAME)
    print("Measured runs:", BENCHMARK_RUNS)

    configurations = [
        (
            "FP16",
            QuantizationConfig(method="none"),
        ),
        (
            "4-bit NF4",
            QuantizationConfig(
                method="bitsandbytes_4bit",
                compute_dtype="float16",
                quant_type="nf4",
                use_double_quant=True,
            ),
        ),
    ]

    results = []

    for label, quantization in configurations:
        print(f"\n{'=' * 50}")
        print(f"Benchmark: {label}")
        print("=" * 50)

        result = benchmark_configuration(
            label=label,
            quantization=quantization,
        )

        results.append(result)

        for metric, value in result.items():
            print(f"{metric}: {value}")

    save_results(results)

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
