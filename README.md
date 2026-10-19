# Small Agentic RAG Benchmark

**An experimental project exploring Transformer-based language models, Mini-GPT implementation, model quantization, inference benchmarking, and Agentic Retrieval-Augmented Generation (RAG).**

## Overview

This project is a hands-on exploration of the fundamental components behind modern Large Language Models (LLMs) and Agentic RAG systems.

The objective is to understand how language models process text, learn to predict the next token, and integrate with retrieval-augmented and agent-based workflows. The project combines educational implementations of Transformer components with experiments on pretrained language models.

Development is organized into three complementary areas:

- **Mini-GPT:** Understanding and implementing the fundamental building blocks of a GPT-style decoder.
- **LLM inference and quantization:** Loading pretrained language models and comparing inference configurations, including FP16 and 4-bit NF4 quantization.
- **Agentic RAG:** Progressively developing retrieval, context augmentation, tool use, and agent orchestration.

The project emphasizes practical experimentation, measurable performance, reproducibility, and a better understanding of the trade-offs between memory consumption, inference speed, and model quality.

## Current Progress

### 1. Mini-GPT: Transformer Implementation

The Mini-GPT implementation explores the fundamental building blocks of a GPT-style language model:

- **Text preprocessing:** Reading a text corpus and converting text into token IDs.
- **Tokenization and vocabulary:** Building token representations and working with vocabulary-based and BPE tokenization.
- **Next-token dataset:** Creating input-target sequences for autoregressive language modeling.
- **Token embeddings:** Mapping token IDs to trainable vector representations.
- **Rotary Positional Embeddings (RoPE):** Incorporating positional information into attention.
- **Masked Multi-Head Self-Attention:** Allowing each token to attend only to preceding tokens and itself.
- **Grouped-Query Attention (GQA):** Exploring the sharing of key-value heads across multiple query heads.
- **Feed-Forward Network:** Applying nonlinear transformations to token representations.
- **Transformer Decoder Blocks:** Combining attention, residual connections, and normalization.
- **GPT Decoder:** Stacking decoder blocks and projecting hidden states into vocabulary logits.
- **Autoregressive language modeling:** Exploring next-token prediction and text generation.

The data-preparation pipeline and core Transformer components have been developed incrementally. Exploratory training has also been conducted in Google Colab, including experiments with a small GPT-style model.

The Mini-GPT work serves as a foundation for understanding the architectures and optimization techniques used by pretrained LLMs.

### 2. Pretrained LLM Inference

A separate inference pipeline has been implemented to load pretrained causal language models and generate text.

Current capabilities include:

- Loading pretrained models and tokenizers with Hugging Face Transformers.
- Configuring the inference device and model data type.
- Running generation with configurable parameters such as `max_new_tokens`.
- Using `torch.inference_mode()` to avoid unnecessary gradient computation.
- Supporting optional BitsAndBytes quantization configurations.
- Running inference experiments on CPU and CUDA-compatible GPUs, subject to model and library compatibility.

The first quantization experiment uses **Qwen2.5-0.5B-Instruct** as the reference model.

### 3. Quantization Benchmark: FP16 vs. 4-bit NF4

The first quantitative experiment compares standard FP16 inference against 4-bit NF4 quantization using BitsAndBytes.

The experiment was conducted in Google Colab on an NVIDIA Tesla T4 GPU.

**Experimental configuration**

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| GPU | NVIDIA Tesla T4 |
| Available GPU memory | Approximately 14.56 GiB |
| FP16 configuration | Standard FP16 model loading |
| Quantized configuration | BitsAndBytes 4-bit NF4 |
| Compute dtype for quantization | FP16 |
| Double quantization | Enabled |
| Warm-up generations | 2 per configuration |
| Measured generations | 10 per configuration |
| Maximum new tokens | 80 |
| Generation mode | Greedy decoding (`do_sample=False`) |

The same prompt and generation settings were used for both configurations.

**Measured results**

| Metric | FP16 | 4-bit NF4 |
|---|---:|---:|
| Model loading time (s) | 3.28 | 2.80 |
| GPU memory after loading (GiB) | 0.928 | 0.435 |
| Peak GPU memory during inference (GiB) | 0.940 | 0.439 |
| Mean generation latency (s) | 2.412 | 3.501 |
| Median generation latency (s) | 2.411 | 3.498 |
| Mean generation throughput (tokens/s) | 33.17 | 22.86 |
| Mean generated tokens | 80 | 80 |

These are preliminary results from one benchmark session. Loading times are single observations per configuration, while generation latency and throughput are aggregated over ten measured runs.

**Initial observations**

- **Lower memory consumption:** The 4-bit NF4 configuration reduced GPU memory allocated after loading by approximately 53.1% compared with FP16.
- **Lower inference throughput:** NF4 achieved approximately 22.86 tokens/s, compared with 33.17 tokens/s for FP16, a decrease of approximately 31.1%.
- **Higher generation latency:** Median latency increased from approximately 2.41 seconds to 3.50 seconds for the tested prompt and generation settings.
- **Comparable output length:** Both configurations generated an average of 80 tokens, allowing a more meaningful comparison of generation performance under the same token limit.

The observations illustrate an important trade-off: **4-bit quantization can reduce GPU memory requirements substantially, but it does not necessarily improve inference speed.**

The results do not establish that one configuration is universally better. Performance depends on the model, hardware, inference backend, sequence length, and workload. Model quality and quantization-induced changes in output quality have not yet been evaluated systematically.

## Planned Development

The next development steps are:

1. Improve benchmark reproducibility through repeated benchmark sessions and alternating configuration order.
2. Record individual-run measurements in addition to aggregate statistics.
3. Extend the benchmark to larger pretrained models when hardware and memory permit.
4. Evaluate quantized-model output quality using appropriate reference tasks and evaluation metrics.
5. Explore additional inference backends and quantization approaches, including AWQ-compatible checkpoints where appropriate.
6. Continue developing and validating the Mini-GPT training and generation pipeline.
7. Investigate Grouped-Query Attention (GQA) and compare its computational and memory characteristics with standard multi-head attention.
8. Develop the Agentic RAG pipeline with dense retrieval, BM25, reranking, and Python tools.
9. Integrate benchmarking and evaluation into a reproducible experimental workflow.

## Architecture

### Mini-GPT

```text
Text Corpus
    |
    v
Tokenization and Vocabulary
    |
    v
Token IDs
    |
    v
Next-Token Dataset
(Input IDs, Shifted Targets)
    |
    v
Token Embeddings
    |
    v
Transformer Decoder
    |
    +-- Causal Self-Attention + RoPE
    |
    +-- Feed-Forward Network
    |
    +-- Residual Connections and Normalization
    |
    v
Language Modeling Head
    |
    v
Vocabulary Logits
    |
    v
Next-Token Prediction
```

### Pretrained LLM Benchmark

```text
Pretrained Model
       |
       v
Model Loader
       |
       +---------------------+
       |                     |
       v                     v
      FP16              4-bit NF4
       |                     |
       v                     v
       +----------+----------+
                  |
                  v
          Text Generation
                  |
                  v
      Latency and Memory Metrics
                  |
                  v
       Throughput and Comparison
                  |
                  v
          CSV Experimental Results
```

### Planned Agentic RAG

```text
User Query
    |
    v
Agent Orchestrator
    |
    +--> Document Retrieval
    |       |
    |       +--> Dense Retrieval
    |       +--> BM25 Retrieval
    |       +--> Cross-Encoder Reranking
    |
    +--> Python Tools
    |
    v
Context Assembly
    |
    v
LLM Generation
    |
    v
Response and Evaluation
```

The Agentic RAG architecture represents the planned direction of the project; not every component is implemented yet.

## Technology Stack

- **Python** — Core implementation and experimentation.
- **PyTorch** — Neural network layers, tensor operations, and model development.
- **Hugging Face Transformers** — Pretrained models, tokenizers, and text generation.
- **BitsAndBytes** — 4-bit and 8-bit model quantization where supported.
- **CUDA** — GPU-accelerated inference and experimentation.
- **Google Colab** — GPU-based experiments and benchmarking.
- **uv** — Python dependency and environment management.
- **pytest** — Unit testing and validation.
- **pandas / CSV** — Experimental results processing and analysis.
- **Git and GitHub** — Version control and project collaboration.

Additional libraries and inference backends may be introduced as the project evolves.

## Project Structure

```text
small-agentic-rag-benchmark/
├── src/
│   └── small_agentic_rag/
│       ├── agents/
│       ├── data/
│       ├── evaluation/
│       ├── llm/
│       │   ├── model_loader.py
│       │   ├── inference.py
│       │   └── quantization.py
│       ├── model_lab/
│       │   ├── attention.py
│       │   ├── rope.py
│       │   ├── feed_forward.py
│       │   ├── transformer_block.py
│       │   ├── gpt_decoder.py
│       │   ├── gpt_language_model.py
│       │   ├── dataset.py
│       │   └── prepare_data.py
│       ├── rag/
│       └── utils/
├── scripts/
│   └── benchmark_quantization.py
├── results/
│   └── quantization_benchmark.csv
├── tests/
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

*The structure may evolve as new components and experiments are implemented.*

## Getting Started

### Prerequisites

- A Python version compatible with `pyproject.toml`.
- [uv](https://docs.astral.sh/uv/).
- Git.

A CUDA-compatible GPU is required for the current GPU quantization benchmark. Google Colab can be used if a suitable local GPU is unavailable.

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/small-agentic-rag-benchmark.git
cd small-agentic-rag-benchmark
```

Install the project dependencies:

```bash
uv sync
```

The quantization benchmark additionally requires compatible versions of PyTorch, Transformers, Accelerate, and BitsAndBytes. Verify that these dependencies are declared in the project configuration before running the benchmark.

### Run the Data Preparation Pipeline

From the project root:

```bash
uv run python -m small_agentic_rag.model_lab.prepare_data
```

The preparation script reads the corpus, encodes tokens, constructs next-token examples, and checks the tensor dimensions used by the model.

### Run Tests

```bash
uv run pytest
```

### Run the Quantization Benchmark

The benchmark requires a CUDA-enabled environment with the required dependencies installed.

From the project root:

```bash
uv run python scripts/benchmark_quantization.py
```

The script compares FP16 and 4-bit NF4 inference for the configured model and saves the aggregated results to:

```text
results/quantization_benchmark.csv
```

The first run may take longer because the pretrained model needs to be downloaded. Subsequent runs can reuse the local model cache when available.

**Important:** The benchmark results depend on hardware, software versions, model revisions, and runtime conditions. For reliable comparisons, record these details and repeat experiments rather than relying on a single session.

## Learning Objectives

This project aims to develop practical understanding of:

- Transformer decoder architectures and causal self-attention.
- Positional encoding with RoPE.
- Multi-head and Grouped-Query Attention.
- Autoregressive language modeling and next-token prediction.
- Pretrained LLM inference.
- Model quantization and GPU memory management.
- Inference latency, throughput, and memory benchmarking.
- Experimental reproducibility and evaluation methodology.
- Retrieval-Augmented Generation.
- Agentic workflows and tool orchestration.

## Project Status

**Work in progress — Mini-GPT development and LLM benchmarking.**

The project has progressed beyond initial data preparation to include Transformer experiments, pretrained-model inference, BitsAndBytes quantization, and a first FP16-versus-NF4 benchmark.

The benchmark currently provides preliminary measurements of GPU memory consumption, generation latency, and throughput. Repeated experiments, larger-model comparisons, systematic quality evaluation, and the complete Agentic RAG workflow remain future development objectives.

## Author

**Abderrahim El Ouardi**

Master's student in Data Science, interested in Machine Learning, Deep Learning, Natural Language Processing, Large Language Models, and Generative AI.


---

*Built as a learning and experimentation project to explore the foundations, optimization, and evaluation of modern language models and retrieval-augmented AI systems.*
