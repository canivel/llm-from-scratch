# Book 1: Build a Large Language Model (From Scratch)

**Author:** Sebastian Raschka
**Status:** Completed
**Link:** [Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch)

---

## What You'll Learn

This book takes you on a journey from zero to a fully functional GPT-style language model. By the end, you'll understand every component that makes modern LLMs work.

```
Text → Tokens → Embeddings → Attention → Transformer → GPT → Training → Fine-tuning
```

---

## Notebooks Overview

| # | Notebook | Topic | Key Concepts |
|---|----------|-------|--------------|
| 1 | [Tokenization](1.Tokenization.ipynb) | Data Preparation | BPE, token embeddings, positional encoding |
| 2 | [Simple Attention](2.Simple-Attention.ipynb) | Attention Basics | Dot-product attention, softmax, context vectors |
| 3 | [Self-Attention with Trainable Weights](3.Self-Attention%20with%20trainable%20weights.ipynb) | Q, K, V | Query, Key, Value matrices, learnable attention |
| 4 | [Causal Attention](4.Causal-Attention.ipynb) | Masking | Causal masking, preventing future token leakage |
| 5 | [Multi-Head Attention](5.Multi-Head-Attention.ipynb) | Parallel Attention | Multiple attention heads, concatenation |
| 6 | [Implementing GPT-2](6.Implementing-GP2.ipynb) | Architecture | GPT-2 components, weight loading |
| 7 | [Transformer](7.Trasformer.ipynb) | Transformer Block | LayerNorm, residual connections, feed-forward |
| 8 | [GPT](8.GPT.ipynb) | Complete Model | Full GPT architecture, text generation |
| 9 | [Pre-Training](9.Pre-Training.ipynb) | Training Setup | Loss functions, data loaders, training loop |
| 10 | [Training](10.Training.ipynb) | Model Training | AdamW, learning rate, checkpointing |
| 11a | [Fine-tuning (Classification)](11.Fine-tuning-classification.ipynb) | Classification | Sentiment analysis, classification head |
| 11b | [Fine-tuning (Instruction)](11.Fine-tuning-instructional.ipynb) | Instruction Tuning | SFT, Alpaca format, instruction following |

---

## Detailed Notebook Descriptions

### 1. Tokenization and Embeddings

**Goal:** Convert raw text into numerical representations the model can process.

**What you'll learn:**
- How BPE (Byte Pair Encoding) tokenization works
- Creating input-target pairs with sliding windows
- Token embeddings: converting IDs to dense vectors
- Positional embeddings: encoding position information

**Key insight:** The same word in different positions needs different representations. Positional embeddings solve this by adding position-specific vectors to token embeddings.

```python
input_embeddings = token_embeddings + positional_embeddings
# Shape: [batch_size, seq_len, embedding_dim]
```

---

### 2. Simple Attention (Without Trainable Weights)

**Goal:** Understand the core attention mechanism using fixed embeddings.

**What you'll learn:**
- Computing attention scores with dot products
- Softmax normalization to get attention weights
- Creating context vectors as weighted sums

**Key insight:** Attention allows each token to "look at" other tokens and gather relevant information. The attention weights determine how much each token contributes.

```
Attention Score = query · key
Attention Weight = softmax(scores)
Context Vector = weights @ values
```

---

### 3. Self-Attention with Trainable Weights (Q, K, V)

**Goal:** Make attention learnable by introducing Query, Key, and Value transformations.

**What you'll learn:**
- Why we need separate Q, K, V projections
- How the model learns what to attend to
- The difference between querying and being queried

**Key insight:** The same word plays THREE different roles:
- **Query:** "What am I looking for?"
- **Key:** "What do I contain?"
- **Value:** "What do I provide?"

```python
Q = input @ W_query  # What this token is looking for
K = input @ W_key    # What this token advertises
V = input @ W_value  # What this token provides
```

---

### 4. Causal Attention (Masked Self-Attention)

**Goal:** Prevent tokens from attending to future positions (crucial for language modeling).

**What you'll learn:**
- Why autoregressive models can't see the future
- Implementing causal masks with upper triangular matrices
- Masking with `-inf` before softmax

**Key insight:** In language modeling, we predict the next token. If we could see future tokens, we'd be cheating! Causal masking ensures each position only attends to previous positions.

```
Mask:
[0, -∞, -∞, -∞]
[0,  0, -∞, -∞]
[0,  0,  0, -∞]
[0,  0,  0,  0]
```

---

### 5. Multi-Head Attention

**Goal:** Allow the model to attend to information from different representation subspaces.

**What you'll learn:**
- Splitting attention into multiple parallel heads
- Each head learns different attention patterns
- Concatenating and projecting head outputs

**Key insight:** Different heads can learn different relationships:
- Head 1: Syntactic relationships (subject-verb)
- Head 2: Semantic relationships (word meanings)
- Head 3: Positional patterns (nearby words)

```python
# Each head operates on d_model / num_heads dimensions
head_dim = 768 // 12 = 64  # For GPT-2 small
```

---

### 6-7. Implementing GPT-2 / Transformer Block

**Goal:** Build the complete transformer block with all components.

**What you'll learn:**
- Layer Normalization for training stability
- GELU activation function
- Feed-Forward Networks (4x expansion)
- Residual connections for gradient flow
- Pre-norm vs Post-norm architectures

**Architecture:**
```
Input
  │
  ├──────────────────┐
  ▼                  │ (residual)
LayerNorm → Attention → Dropout
  │                  │
  └────────(+)───────┘
  │
  ├──────────────────┐
  ▼                  │ (residual)
LayerNorm → FFN → Dropout
  │                  │
  └────────(+)───────┘
  │
Output
```

---

### 8. Complete GPT Model

**Goal:** Assemble all components into a working GPT model.

**What you'll learn:**
- Token and position embedding layers
- Stacking transformer blocks
- Output projection to vocabulary size
- Text generation (greedy decoding)

**Model Architecture:**
```
Token IDs → Token Embed + Pos Embed → [Transformer Block × 12] → LayerNorm → Output Head → Logits
```

---

### 9. Pre-Training Setup

**Goal:** Prepare everything needed to train the model.

**What you'll learn:**
- Cross-entropy loss for next-token prediction
- Perplexity as an interpretable metric
- Creating efficient data loaders
- Train/validation splits

**Key metrics:**
- **Loss:** Lower is better (measures prediction error)
- **Perplexity:** `e^loss` - "how many tokens is the model confused between?"
  - Perplexity of 1 = perfect prediction
  - Perplexity of 50,000 = random guessing

---

### 10. Training the Model

**Goal:** Actually train the GPT model on text data.

**What you'll learn:**
- AdamW optimizer with weight decay
- Learning rate scheduling
- Gradient clipping for stability
- Checkpointing and model saving
- Monitoring training progress

**Training loop:**
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
```

---

### 11a. Fine-tuning for Classification

**Goal:** Adapt the pre-trained model for sentiment classification.

**What you'll learn:**
- Adding a classification head
- Freezing vs fine-tuning layers
- Binary classification with GPT
- Transfer learning principles

**Key insight:** The pre-trained model already understands language. We just need to add a small classification layer and fine-tune on labeled data.

---

### 11b. Fine-tuning for Instruction Following (SFT)

**Goal:** Transform GPT into an instruction-following assistant.

**What you'll learn:**
- Alpaca instruction format
- Custom collate functions for variable-length sequences
- Masking padding tokens in loss calculation
- Supervised Fine-Tuning (SFT)

**Instruction Format:**
```
### Instruction:
{task description}

### Input:
{optional context}

### Response:
{model output}
```

**Before SFT:** Model just predicts next tokens, doesn't understand instructions
**After SFT:** Model follows instructions and generates appropriate responses

---

## Learning Path

```
                                    ┌─────────────────────┐
                                    │   11a. Fine-tune    │
                                    │   (Classification)  │
                                    └─────────┬───────────┘
                                              │
1. Tokenization                               │
      │                                       │
      ▼                               ┌───────┴───────┐
2. Simple Attention                   │               │
      │                               │   10. Train   │
      ▼                               │               │
3. Q, K, V Attention                  └───────┬───────┘
      │                                       │
      ▼                                       │
4. Causal Masking                     ┌───────┴───────┐
      │                               │               │
      ▼                               │ 9. Pre-Train  │
5. Multi-Head                         │    Setup      │
      │                               └───────┬───────┘
      ▼                                       │
6-7. Transformer Block ───────────────────────┤
      │                                       │
      ▼                               ┌───────┴───────┐
8. Complete GPT ──────────────────────┤               │
                                      │   11b. SFT    │
                                      │ (Instruction) │
                                      └───────────────┘
```

---

## Key Takeaways

### 1. Attention is All You Need (Almost)
The attention mechanism is the core innovation. It allows tokens to dynamically focus on relevant parts of the input, regardless of distance.

### 2. Scale Matters
GPT-2 Small (124M) → GPT-2 XL (1.5B) → GPT-3 (175B) → GPT-4 (??)
More parameters + more data + more compute = better performance.

### 3. Pre-training + Fine-tuning Paradigm
- **Pre-training:** Learn general language patterns (expensive, done once)
- **Fine-tuning:** Adapt to specific tasks (cheap, done many times)

### 4. The Training Objective is Simple
Just predict the next token! This simple objective, at scale, leads to emergent capabilities like reasoning, translation, and code generation.

---

## Requirements

```bash
pip install torch tiktoken transformers
```

**Hardware:**
- CPU: Works, but slow
- GPU: Recommended for training (10-100x faster)
- The notebooks are designed to run on free Google Colab GPUs

---

## What's Next?

After completing Book 1, you understand the foundations. The next steps are:

1. **Book 2:** Advanced architectures (MoE, Multi-head Latent Attention)
2. **Book 3:** Alignment techniques (RLHF, DPO, GRPO)
3. **Book 4:** Reasoning capabilities

Happy learning!
