# MinGPT from Scratch

This repository contains implementations of language models built from scratch using PyTorch, inspired by [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT). The models are trained on a character-level corpus of Shakespeare's works and include:

- **`bigram.py`**: A baseline bigram language model.
- **`MinGPT.py`**: A minimal GPT language model implementation.
- **`input.txt`**: The training corpus containing Shakespeare's text.

## Features

- **Character-level tokenizer**: Processes text at the character level.
- **Transformer architecture**: Implements multi-head self-attention mechanisms.
- **Configurable hyperparameters**: Easy experimentation with model settings.
- **Simple training loop**: Understandable and modifiable training process.

## Getting Started

### Prerequisites

- Python 3.x
- [PyTorch](https://pytorch.org/get-started/locally/)
- CUDA toolkit (optional, for GPU acceleration)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/MinGPT-from-scratch.git
   cd MinGPT-from-scratch
   ```

2. **Install dependencies**

   ```bash
   pip install torch
   ```

   *(Additional packages may be required depending on your environment.)*

### Data Preparation

The `input.txt` file contains the corpus of Shakespeare's works used for training. To use a different dataset, replace `input.txt` with your own text file.

## Usage

### Running the Bigram Model

The bigram model serves as a simple baseline.

```bash
python bigram.py
```

This script will train the bigram model and generate sample text output.

### Running the MinGPT Model

The `MinGPT.py` script implements a minimal GPT model.

```bash
python MinGPT.py
```

This script will train the GPT model and generate sample text after training.

### Hyperparameters

You can adjust hyperparameters directly in the `MinGPT.py` script:

```python
# Hyperparameters
batch_size = 32      # Number of sequences processed in parallel
seq_len = 128        # Maximum context length for predictions
max_iters = 5000     # Total training iterations
eval_interval = 500  # Interval for evaluation
learning_rate = 3e-4 # Learning rate for the optimizer
d_model = 192        # Model dimension
num_head = 6         # Number of attention heads
n_layer = 3          # Number of Transformer blocks
dropout = 0.2        # Dropout rate
```

### Generating Text

After training, the model will generate text based on the learned patterns:

```python
# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.inference(context, max_len=500)[0].tolist()))
```

## Code Overview

### Tokenization

- Uses a character-level tokenizer.
- Encodes text into integers and decodes integers back to text.

### Model Architecture

- **Embedding Layers**: Token and positional embeddings.
- **Transformer Blocks**: Stacked layers with multi-head self-attention and feed-forward networks.
- **Layer Normalization**: Applied before each sub-layer (Pre-LN).
- **Residual Connections**: Added around sub-layers for stable training.

### Training Loop

- Uses mini-batch stochastic gradient descent with the AdamW optimizer.
- Evaluates training and validation loss at regular intervals.
- Generates sample text after training.

## Sample Output

Here's an example of text generated by the MinGPT model after training:

```
Enter KING HENRY VI:

What says Lord Montague? Will you not go?
I am his father and his mother; therefore
I cannot stay him further.

KING EDWARD IV:
O, to see what thou wilt do!
```

*(Note: The output may vary based on training and randomness.)*

## Acknowledgments

- **Andrej Karpathy**: For his [minGPT](https://github.com/karpathy/minGPT) repository, which inspired this project.
- **Shakespeare**: The dataset consists of his collected works.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

Feel free to contribute to this project by opening issues or submitting pull requests.