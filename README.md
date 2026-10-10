# Small Agentic RAG Benchmark

**An experimental project exploring Transformer-based language models, Mini-GPT implementation, and Agentic Retrieval-Augmented Generation (RAG).**

## Overview

This project is a hands-on exploration of the fundamental components behind modern Large Language Models (LLMs) and Agentic RAG systems.

The goal is to understand how language models process text, learn to predict the next token, and can eventually be integrated into retrieval-augmented and agent-based workflows.

The project is developed incrementally, starting with the implementation of a small GPT-style decoder architecture and progressively exploring more advanced language-modeling and RAG concepts.

## Current Progress

### Mini-GPT: Transformer Implementation

The current implementation focuses on the fundamental building blocks of a GPT-style language model:

- **Text preprocessing:** Reading a text corpus and converting text into token IDs.
- **Vocabulary construction:** Building a vocabulary from the corpus, including support for unknown tokens.
- **Next-token dataset:** Creating input-target sequences for autoregressive language modeling.
- **Token embeddings:** Mapping token IDs to trainable vector representations.
- **Rotary Positional Embeddings (RoPE):** Incorporating positional information into attention.
- **Masked Multi-Head Self-Attention:** Allowing each token to attend only to preceding tokens and itself.
- **Feed-Forward Network:** Applying nonlinear transformations to token representations.
- **Transformer Decoder Blocks:** Combining attention, residual connections, and layer normalization.
- **GPT Decoder:** Stacking decoder blocks and projecting hidden states into vocabulary logits.

The current data preparation pipeline has been tested with a small text corpus containing approximately 1,438 tokens and a vocabulary of 634 entries.

### Planned Development

- Implement and validate the next-token training loop.
- Experiment with training hyperparameters and loss monitoring.
- Implement autoregressive text generation.
- Evaluate model behavior on held-out text.
- Develop the Agentic RAG pipeline.
- Explore retrieval, context augmentation, and agent-based orchestration.
- Add experiments and benchmarks to compare different configurations.

## Architecture

```text
Text Corpus
    |
    v
Vocabulary & Tokenization
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
    +-- Masked Multi-Head Attention + RoPE
    |
    +-- Feed-Forward Network
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

## Technology Stack

- **Python** — Core implementation.
- **PyTorch** — Neural network layers, tensor operations, and model development.
- **uv** — Python dependency and environment management.
- **pytest** — Unit testing.
- **Git and GitHub** — Version control and project collaboration.

Additional tools and libraries may be introduced as the project evolves.

## Project Structure

```text
small-agentic-rag-benchmark/
├── src/
│   └── small_agentic_rag/
│       ├── agents/
│       ├── data/
│       ├── evaluation/
│       ├── llm/
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
├── tests/
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

*The structure may evolve as new components are implemented.*

## Getting Started

### Prerequisites

- Python version compatible with `pyproject.toml`
- [uv](https://docs.astral.sh/uv/)

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

### Run the Data Preparation Pipeline

From the project root, run:

```bash
uv run python -m small_agentic_rag.model_lab.prepare_data
```

If the package is not installed or discovered correctly, configure the project installation and package discovery rather than relying permanently on a manually set `PYTHONPATH`.

The preparation script reads the corpus, encodes tokens, constructs next-token examples, and checks the tensor dimensions used by the model.

### Run Tests

```bash
uv run pytest
```

## Learning Objectives

This project aims to develop a practical understanding of:

- Transformer decoder architectures.
- Causal self-attention and positional encoding.
- Token embeddings and vocabulary projections.
- Autoregressive language modeling.
- Model training and evaluation.
- Retrieval-Augmented Generation.
- Agentic workflows and benchmarking.

## Project Status

**Work in progress.**

The Mini-GPT architecture and data preparation pipeline are under development. Training, text generation, evaluation, and the complete Agentic RAG workflow remain part of the planned development.

## Author

**Abderrahim El Ouardi**

Master's student in Data Science, interested in Machine Learning, Deep Learning, Natural Language Processing, and Generative AI.

- GitHub: [YOUR_GITHUB_USERNAME](https://github.com/YOUR_GITHUB_USERNAME)
- LinkedIn: [Abderrahim El Ouardi](https://www.linkedin.com/in/abderrahimelouardi/)

---

*Built as a learning and experimentation project to explore the foundations of modern language models and retrieval-augmented AI systems.*