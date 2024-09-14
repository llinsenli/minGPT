import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
seq_len = 128 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
d_model = 192
num_head = 6
n_layer = 3
assert d_model % num_head == 0, "The d_model is divisible by num_head"
dropout = 0.2
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

class Head(nn.Module):
    '''
    The single head self-attention
    '''
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Compute the attention score
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T) 
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # (B, T, T) 
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # Randomly prevent some of the nodes from communicating
        # Perform the weighted aggregation of the value
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    '''
    Multiple heads of self-attention in parallel
    '''
    def __init__(self, num_head, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(d_model, d_model) # projection layer going back to the residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (batch_size, seq_len, num_head * head_size = d_model)
        out = self.proj(out)
        out  = self.dropout(out)
        return out

class FeedForward(nn.Module):
    '''
    The fully connected network
    '''
    def __init__(self, d_model) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model), # projection layer going back to the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    '''
    Transformer Block, include the multi-head self-sttention and feed forward layer with residual connection
    '''
    def __init__(self, d_model, num_head) -> None:
        super().__init__()
        head_size = d_model // num_head
        self.sa = MultiHeadAttention(num_head, head_size)
        self.ffwd = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x) # Apply the layernorm before the transformation
        x = x + self.sa(x) # Residual connection
        x = self.ln2(x) # Apply the layernorm before the transformation
        x = x + self.ffwd(x) # Residual connection
        return x

class MinGPTModel(nn.Module):
    """
    MinGPT model, include token embedding, position embedding, decoder blocks, language model head
    """
    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, d_model) # Mapping each token idx to a d_model vector
        self.position_embedding_table = nn.Embedding(seq_len, d_model) # mapping each position in seq_len to a d_model vector
        self.blocks = nn.Sequential(
            *[Block(d_model, num_head) for _ in range(n_layer)],
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        

    def forward(self, idx, label = None):

        batch_size, seq_len = idx.shape # x, label are both (batch_size, seq_len)
        tok_emb = self.token_embedding_table(idx) # (batch_size, seq_len, d_model)
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=device)) # (seq_len, d_model)
        x = tok_emb + pos_emb # (batch_size, seq_len, d_model)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)

        if label is None:
            loss = None

        else:
            logits = logits.view(-1, self.vocab_size) # (batch_size * seq_len, vocab_size)
            label = label.view(-1) # (batch_size * seq_len)
            loss = F.cross_entropy(logits, label)
        return logits, loss
    
    def inference(self, x, max_len:int):
        initial_x = x  # (batch_size, 1)
        while True:
            if initial_x.size(1) == max_len:
                break
            cond_x = initial_x[:, -seq_len:] # crop idx to the last seq_len tokens, (batch_size, seq_len)
            logits, _ = self(cond_x) # (batch_size, seq_len, vocab_size)
            # Focus on the last position in second dim, select the last item in seq_len dim
            logits = logits[:, -1, :] # (batch_size, vocab_size), this will reduce the dim
            probs = F.softmax(logits, dim=-1) # (batch_size, vocab_size), the sum of the value in the last dim is 1
            next_words = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            initial_x = torch.cat([initial_x, next_words], dim=1) # (batc_size, 1 + initial_x.size(1)), concat on the second dim 
        return initial_x # (batch_size, max_len)

model = MinGPTModel()
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