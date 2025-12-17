# LLM from Zero to RLVR with GRPO

Personal implementations and notes while working through LLM-related books. Each folder contains code written as I follow along with the material.

> **Goal:** Understand LLMs from the ground up - from tokenization to RLHF alignment.

---

## Progress Overview

| Book | Topic | Status | Progress |
|------|-------|--------|----------|
| [Book 1](book1/) | Build a Large Language Model | **Completed** | 12/12 notebooks |
| Book 2 | Build a DeepSeek Model | On Hold (MEAP) | - |
| [Book 3](book3/) | The RLHF Book | **In Progress** | Starting |
| Book 4 | Build a Reasoning Model | On Hold (MEAP) | - |

---

## Book 1: Build a Large Language Model (From Scratch)

**Author:** Sebastian Raschka | [Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch) | **Status: Completed**

The foundational journey from raw text to a working GPT model.

### What I Built

```
Text → Tokens → Embeddings → Attention → Transformer → GPT → Training → Fine-tuning
```

### Notebooks

| # | Topic | What I Learned |
|---|-------|----------------|
| 1 | [Tokenization](book1/1.Tokenization.ipynb) | BPE tokenization, embeddings, positional encoding |
| 2 | [Simple Attention](book1/2.Simple-Attention.ipynb) | Core attention mechanism without training |
| 3 | [Q, K, V Attention](book1/3.Self-Attention%20with%20trainable%20weights.ipynb) | Trainable Query, Key, Value matrices |
| 4 | [Causal Attention](book1/4.Causal-Attention.ipynb) | Masking future tokens for autoregressive models |
| 5 | [Multi-Head Attention](book1/5.Multi-Head-Attention.ipynb) | Parallel attention heads |
| 6 | [GPT-2 Implementation](book1/6.Implementing-GP2.ipynb) | Building GPT-2 architecture |
| 7 | [Transformer Block](book1/7.Trasformer.ipynb) | LayerNorm, GELU, residual connections |
| 8 | [Complete GPT](book1/8.GPT.ipynb) | Full model assembly, text generation |
| 9 | [Pre-Training Setup](book1/9.Pre-Training.ipynb) | Loss functions, data loaders |
| 10 | [Training](book1/10.Training.ipynb) | Training loop, optimization |
| 11a | [Fine-tuning (Classification)](book1/11.Fine-tuning-classification.ipynb) | Sentiment classification |
| 11b | [Fine-tuning (Instruction)](book1/11.Fine-tuning-instructional.ipynb) | Instruction following (SFT) |

**[Full Book 1 Documentation](book1/README.md)** - Detailed explanations for each notebook

### Key Takeaways

1. **Attention** allows tokens to dynamically focus on relevant context
2. **Q, K, V** separate "what I'm looking for" from "what I contain" from "what I provide"
3. **Pre-training** teaches general language; **fine-tuning** adapts to tasks
4. **SFT** transforms a next-token predictor into an instruction follower

---

## Book 2: Build a DeepSeek Model (From Scratch)

**Author:** Raj Abhijit Dandekar, et al. | [Manning](https://www.manning.com/books/build-a-deepseek-model-from-scratch) | **Status: On Hold (MEAP)**

Modern LLM architectures including:
- Mixture of Experts (MoE)
- Multi-head Latent Attention
- Efficient inference techniques

*Waiting for MEAP completion*

---

## Book 3: The RLHF Book

**Author:** Nathan Lambert | [Manning](https://www.manning.com/books/the-rlhf-book) | **Status: In Progress**

Alignment techniques to make LLMs helpful, harmless, and honest.

### Topics Covered

- **RLHF** - Reinforcement Learning from Human Feedback
- **DPO** - Direct Preference Optimization
- **GRPO** - Group Relative Policy Optimization
- Reward modeling
- Constitutional AI

### Why This Matters

Pre-training + SFT gives us a capable model, but:
- It might give harmful responses
- It might hallucinate confidently
- It might not align with user intent

RLHF/DPO teaches the model human preferences.

---

## Book 4: Build a Reasoning Model (From Scratch)

**Author:** Sebastian Raschka | [Manning](https://www.manning.com/books/build-a-reasoning-model-from-scratch) | **Status: On Hold (MEAP)**

Teaching LLMs to reason:
- Chain-of-thought prompting
- Self-consistency
- Tree of thoughts
- Reasoning fine-tuning

*Waiting for MEAP completion*

---

## Repository Structure

```
llm-from-scratch/
├── README.md           # This file
├── book1/              # Build a Large Language Model
│   ├── README.md       # Detailed book 1 documentation
│   ├── 1.Tokenization.ipynb
│   ├── 2.Simple-Attention.ipynb
│   ├── 3.Self-Attention with trainable weights.ipynb
│   ├── 4.Causal-Attention.ipynb
│   ├── 5.Multi-Head-Attention.ipynb
│   ├── 6.Implementing-GP2.ipynb
│   ├── 7.Trasformer.ipynb
│   ├── 8.GPT.ipynb
│   ├── 9.Pre-Training.ipynb
│   ├── 10.Training.ipynb
│   ├── 11.Fine-tuning-classification.ipynb
│   └── 11.Fine-tuning-instructional.ipynb
├── book2/              # DeepSeek Model (MEAP - on hold)
├── book3/              # RLHF Book (in progress)
└── book4/              # Reasoning Model (MEAP - on hold)
```

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Development Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Book 1: Foundation                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │   Tokens    │ →  │  Attention  │ →  │  Transformer │        │
│   │ Embeddings  │    │   Q, K, V   │    │    Blocks    │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│          │                                      │                │
│          └──────────────────┬───────────────────┘                │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │   GPT Model     │                          │
│                    │  (Pre-trained)  │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│   ──────────────────────────┼────────────────────────────────   │
│                             │                                    │
│   Book 1 (cont): Fine-tuning│                                    │
│                             ▼                                    │
│          ┌──────────────────┴──────────────────┐                │
│          │                                      │                │
│   ┌──────▼──────┐                      ┌───────▼───────┐        │
│   │Classification│                      │  Instruction  │        │
│   │  Fine-tune   │                      │   Fine-tune   │        │
│   └─────────────┘                      │     (SFT)     │        │
│                                        └───────┬───────┘        │
│                                                │                 │
│   ─────────────────────────────────────────────┼─────────────   │
│                                                │                 │
│   Book 3: Alignment                            ▼                 │
│                                    ┌───────────────────┐        │
│                                    │   RLHF / DPO      │        │
│                                    │   Alignment       │        │
│                                    └─────────┬─────────┘        │
│                                              │                   │
│   ──────────────────────────────────────────┼────────────────   │
│                                              │                   │
│   Book 4: Reasoning                          ▼                   │
│                                    ┌───────────────────┐        │
│                                    │    Reasoning      │        │
│                                    │   Fine-tuning     │        │
│                                    └───────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Requirements

```bash
pip install torch tiktoken transformers
```

### Recommended Order

1. **Start with Book 1** - Complete all 12 notebooks in order
2. **Move to Book 3** - Learn alignment techniques
3. **Book 2 & 4** - When MEAP versions are complete

### Running the Notebooks

All notebooks are designed to run on:
- **Google Colab** (free GPU)
- **Local machine** with CUDA GPU
- **CPU** (slower, but works for learning)

---

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - RLHF for instruction following
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization

---

## License

Personal learning project. Book content belongs to respective authors and publishers.
