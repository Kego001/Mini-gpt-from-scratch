# Mini-GPT: Transformer From Scratch

A small character-level Transformer language model implemented from scratch in PyTorch using raw tensors.

This project was built to understand how a GPT-style language model works internally rather than relying on `nn.Transformer` or a pretrained language model.

## What I Implemented

- Character-level tokenization
- Token embeddings
- Positional embeddings
- Multi-head causal self-attention
- Scaled dot-product attention
- Causal masking
- Layer normalization
- Residual connections
- Feed-forward network (FFN)
- Linear output projection
- Cross-entropy loss
- Backpropagation with Adam
- Autoregressive text generation
- Separate training and validation data

## Architecture

```text
Input Text
    ↓
Character Tokenization
    ↓
Token Embedding + Positional Embedding
    ↓
LayerNorm
    ↓
Multi-Head Causal Self-Attention
    ↓
Residual Connection
    ↓
LayerNorm
    ↓
Feed-Forward Network
    ↓
Residual Connection
    ↓
Linear Output Layer
    ↓
Next-Character Probabilities
    ↓
Autoregressive Generation
```

## Model Configuration

| Parameter | Value |
|---|---:|
| Tokenization | Character-level |
| Embedding dimension | 32 |
| Attention heads | 4 |
| Transformer blocks | 1 |
| Context window (`block_size`) | 8 |
| FFN hidden dimension | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training steps | 20,000 |

The vocabulary size is determined automatically from the input text.

## Training

The dataset is split into:

- 90% training data
- 10% validation data

For every training example, the model receives a sequence of characters and learns to predict the next character at every position.

For example:

```text
Input:  "hell"
Target: "ello"
```

Because the model is causal, each position can only attend to itself and previous positions, never future characters.

Validation loss is estimated periodically during training to compare generalization against the training loss.

## Experiments

### Tiny Shakespeare

The model was first tested on the Tiny Shakespeare dataset to verify that the Transformer could learn character-level language patterns and generate text.

### Python Code Dataset

The same architecture was also tested on a small dataset created from a Python project converted to plain text.

This experiment showed that the model could learn local patterns from programming code and reproduce tokens/patterns present in the training data, while still struggling with long-range code structure.

Because the model uses an 8-character context window, it has limited ability to capture dependencies that span longer distances, such as matching brackets or maintaining indentation across larger sections of code.

## Observations

### 1. Weight initialization matters

The initial unscaled `randn` weights produced excessively large activations and attention scores. This caused the softmax in attention to become overly saturated.

The implementation was changed to use scaled initialization so that activations and attention scores stay in a more reasonable range during training.

### 2. Train and validation loss reveal overfitting

On the smaller Python-code dataset, validation loss eventually stopped improving while training loss continued to decrease.

This indicates that the model began fitting the training data more closely than the validation data.

The result is consistent with the limitations of a small dataset combined with a model that still has enough capacity to memorize local patterns.

### 3. Short context limits structure

The model can learn local character relationships, but `block_size = 8` is extremely short for programming-language structure.

For example, if a closing bracket depends on something that happened much earlier than eight characters ago, the model cannot directly access that information.

## Limitations

This is intentionally a very small educational model.

- Single Transformer block
- 32-dimensional embeddings
- 4 attention heads
- 8-character context window
- Character-level rather than subword tokenization
- Small training datasets
- No dropout or other large-scale regularization
- Generated code is not guaranteed to be syntactically valid
- Not intended to compete with production code-generation models or LLMs

The goal of the project is understanding and experimentation rather than production-level generation quality.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place a plain-text dataset in the repository and name it:

```text
input.txt
```

The notebook reads this file automatically.

Any plain-text dataset can be used, including source code, stories, lyrics, or other text.

### 4. Run the notebook

Open:

```text
mini_gpt.ipynb
```

Run the notebook from top to bottom.

The final cell performs autoregressive generation from a starting prompt.

## Files

```text
mini-gpt-from-scratch/
│
├── mini_gpt.ipynb
├── input.txt
├── README.md
├── requirements.txt
└── .gitignore
```

## Future Improvements

Possible next steps include:

- Increasing the context window
- Adding multiple Transformer blocks
- Using subword tokenization
- Adding dropout
- Training on a larger and more diverse dataset
- Converting the raw-tensor implementation into reusable `nn.Module` classes
- Adding temperature/top-k sampling
- Visualizing attention weights
- Comparing different context lengths and numbers of attention heads

## Why I Built It

The main purpose of this project was to understand the mechanics of a GPT-style model by implementing the important components manually.
