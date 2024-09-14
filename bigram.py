import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
seq_len = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # The vocabulary
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Let's now split up the data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long) 
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data= data[:n]
val_data = data[n:]

# Data loading
def get_batch(split:str):
    '''
    split: 'train' or 'text' to specify the train_data or val_data
    '''
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # randomly generate the start idx for the batch_size  of sentence 
    start_idx = torch.randint(len(data) - seq_len, (batch_size,)) # (batch_size)
    x = torch.stack([data[idx: idx + seq_len] for idx in start_idx])
    y = torch.stack([data[idx+1: idx+1 + seq_len] for idx in start_idx])
    x, y = x.to(device), y.to(device)
    return x, y 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model using an embedding matrix as a transition probability table.
    This model predicts the next word based on the current word's index, simulating a bigram model.
    The model is designed for educational and conceptual understanding of basic NLP operations
    involving embeddings and softmax-based probability distributions for text generation.

    Attributes:
        vocab_size (int): Size of the vocabulary. The model assumes that input tokens are 
                          represented as integer indices within this vocabulary.
        transition_matrix (nn.Embedding): Embedding layer to simulate transition probabilities
                                          between consecutive words in the language model.

    Methods:
        forward(x, label=None): Processes input tensor `x` through the model and calculates the
                                loss if `label` is provided. Returns logits and optionally the loss.
        inference(x, max_len): Generates text starting from input `x` up to a length of `max_len`.
                               Uses greedy sampling based on the predicted probabilities at each step.
    """
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # This embedding layer is used to directly output logits for the next word.
        # It acts like a probability table for the transition from one word to the next.
        self.transition_matrix = nn.Embedding(vocab_size, vocab_size) 

    def forward(self, x, label = None):
        # x, label are both (batch_size, seq_len)
        logits = self.transition_matrix(x) # (batch_size, seq_len, vocab_size)

        if label is None:
            loss = None

        else:
            logits = logits.view(-1, self.vocab_size) # (batch_size * seq_len, vocab_size)
            label = label.view(-1) # (batch_size * seq_len)
            loss = F.cross_entropy(logits, label)
        return logits, loss
    
    def inference(self, x, max_len:int):
        initial_x = x  # (batch_size, seq_len)
        while True:
            if initial_x.size(1) == max_len:
                break
            logits, _ = self(initial_x) # (batch_size, seq_len, vocab_size)
            # Focus on the last position in second dim, select the last item in seq_len dim
            logits = logits[:, -1, :] # (batch_size, vocab_size), this will reduce the dim
            probs = F.softmax(logits, dim=-1) # (batch_size, vocab_size), the sum of the value in the last dim is 1
            next_words = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            initial_x = torch.cat([initial_x, next_words], dim=1) # (batc_size, 1 + initial_x.size(1)), concat on the second dim 
        return initial_x # (batch_size, max_len)

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.inference(context, max_len=500)[0].tolist()))