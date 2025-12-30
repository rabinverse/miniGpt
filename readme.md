# GPT from Scratch: Decoder Only Transformer Architecture

A decoder-only transformer implementation built from scratch using PyTorch, designed to learn and imitate text through autoregressive generation. Trained on **Narayan Gopal and Shakespeare's works**, this character level language model demonstrates the power of causal self-attention and next token prediction.



---
Find experiments on [MLflow Dashboard](https://dagshub.com/rabinverse/miniGpt.mlflow/#/experiments/1/runs)
---







## Project Overview

This is a pure decoder architecture (like GPT) that learns to generate text autoregressively by predicting the next character given all previous characters. Unlike encoder-decoder models (like the original Transformer), this implementation uses only the decoder stack with causal masking to ensure each position can only attend to previous positions.

## Features

- **Full Transformer Architecture**: Pure autoregressive transformer (GPT-style)
- **Modern Training Pipeline**: Integrated MLflow experiment tracking
- **GPU Acceleration**: CUDA support for faster training
- **Dropout Regularization**: 30% dropout to prevent overfitting
- **Positional Embeddings**: Learnable position encodings
- **Residual Connections**: Skip connections with layer normalization
- **Text Generation**: Autoregressive text generation from trained model

## Architecture Details

```
Decoder-Only Transformer :
├── Input: Character sequence
├── Token Embedding Layer (vocab_size × 384)
├── Positional Embedding Layer (256 × 384)
│
├── 6× Decoder Blocks (Autoregressive)
│   ├── Masked Multi-Head Self-Attention (6 heads × 64-dim)
│   │   ├── Query, Key, Value projections
│   │   ├──  CAUSAL MASK
│   │   └── Scaled dot-product attention
│   ├── Layer Normalization (Pre-LN)
│   ├── Feed-Forward Network (384 → 1536 → 384)
│   │   └── ReLU activation
│   └── Residual Connections (around attention & FFN)
│
├── Final Layer Normalization
├── Language Model Head (384 → vocab_size)
└── Output: Next character probabilities
```

---

# Getting Started

### Prerequisites

```bash
python >= 3.8
torch >= 2.0.0
tqdm
mlflow
```

### Installation

```bash
# Clone the repository
git clone https://github.com/rabinverse/miniGpt
cd gpt-transformer

pip install torch tqdm mlflow

# Prepare your dataset
mkdir -p dataset_research_paper_docs
# Place your training text in: dataset_research_paper_docs/input_text.txt
```

### Training

```bash
python gpt.py
```

The training script will:

1. Load and tokenize your text data
2. Initialize the GPT model
3. Train for 5000 iterations with progress tracking
4. Log metrics to MLflow
5. Generate sample text and save to `dataset_research_paper_docs/generated_text.txt`

### Monitoring Training

MLflow automatically tracks your experiments. View the dashboard:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` to see training metrics, hyperparameters, and generated samples.

## Model Performance

The model is evaluated every 300 steps on both training and validation sets using cross-entropy loss. Metrics are logged to MLflow for analysis:

- Training loss progression
- Validation loss for monitoring overfitting
- Generated text samples as artifacts

## Key Components Explained

### Multi-Head Attention

The self-attention mechanism allows each token to attend to all previous tokens in the sequence, implementing causal masking for **autoregressive generation**.

```python
# Scaled dot-product attention with masking
wei = (q @ k.transpose(-2, -1)) * C**-0.5
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

### Residual Connections

Each transformer block uses residual connections around attention and feed-forward layers:

```python
x = x + self.sa(self.ln1(x))  # Attention with residual
x = x + self.ffwd(self.ln2(x))  # FFN with residual
```

### Text Generation

The model generates text autoregressively, sampling from the probability distribution at each step.

## Project Structure

```
.
├── dataset_research_paper_docs/
│   ├── input_text.txt               # Training data
│   └── generated_text.txt           # Generated samples
|
├── gpt.py                      # Main training script
├── notebooks
└── mlruns/                          # MLflow experiment logs
```

# Customization

### Training on Your Own Data

Replace `dataset_research_paper_docs/input_text.txt` with your text corpus. The model works with character-level tokenization, so any UTF-8 text will work.

### Changing Context Length

```python
block_size = 512  # Longer context for better coherence
```

## Performance Tips

- **GPU Usage**: The script automatically uses CUDA if available
- **Batch Size**: Increase for better GPU utilization (if memory allows)
- **Mixed Precision**: Consider using `torch.cuda.amp` for faster training
- **Gradient Accumulation**: For larger effective batch sizes

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## Acknowledgments

Special thanks to the PyTorch team and the broader ML community for making transformer implementations accessible and educational.

---

**Built with [⚡](https://dagshub.com/rabinverse/miniGpt.mlflow/#/experiments/1/runs) and lots of matrix multiplications**
