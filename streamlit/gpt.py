import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time
from datetime import datetime


# Hyperparameters
block_size = 256  # context length of input
batch_size = 64  # no of input sequence to process in parallel
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.mps.is_available() else "cpu"
eval_iters = 200
n_embd = 384
head_size = 32
n_layer = 6
n_head = 6  # 384/6=64 -every head is of 64 dim
dropout = 0.3  # 30%dropout
max_new_tokens = 1000


# ------------
torch.manual_seed(42)

with open("./dataset_research_paper_docs/input_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ------------


chars = sorted(list(set(text)))
vocab_size = len(chars)



# ------------


# ------------text to int and reverse
strtoint = {ch: i for i, ch in enumerate(chars)}
inttostr = {i: ch for i, ch in enumerate(chars)}

encode_txt = lambda s: [strtoint[c] for c in s]
# returns list of integer for input string given

decode_txt = lambda l: "".join(inttostr[i] for i in l)
# returns string from given integers


# ------------
# encode whole text
data = torch.tensor(encode_txt(text), dtype=torch.long)

# split to train test
n = int(0.9 * len(data))


# first 90% in the train and rest 10% in the val


train_data = data[:n]
val_data = data[n:]

# ------------


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# xb, yb = get_batch("train")
# xb is the input to the transformer


# ------------
#   def forward(self, idx, targets=None):
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # masking
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores -- affinities
        wei = (q @ k.transpose(-2, -1)) * C**-0.5  # (B,T,C)@(B,C,T). -->(B,T,T)
        # masking
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # type: ignore
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        #
        v = self.value(x)  # (B,T,C)
        # perform weighted aggregation of the values calculating affinity
        out = wei @ v
        return out


# ------------
class MultiHeadAttention(nn.Module):
    "multiple head of self attention in parallel"

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concat in channel dim
        out = self.proj(out)
        out = self.dropout(out)
        #   linear projection of torch.cat([h(x) for h in self.heads], dim=-1) layer
        return out


# ------------
class FeedForwardNetwork(nn.Module):
    # a simple linear layer followed by a non linearity
    # from papaer--The dimensionality of input and output is ( d_{model} = 512 ), and the inner-layer has dimensionality ( d_{ff} = 2048 ).
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForwardNetwork(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # without residual connection
        # x = self.sa(x)
        # x = self.ffwd(x)

        # add residual
        # apply layer norm before sending to self attention and feed forward
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ------------
# ------------
# ------------


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # 65*65
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd)
        # self.sa_head = MultiHeadAttention(4, n_embd // 4)
        # 4heads of 8-dim self-attention
        # self.ffwd = FeedForwardNetwork(n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # language modelling head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        # tok_emb becomes (B,T,C)
        # ---add pos_embedding to the token embedding
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        # pos_emb returns # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)
        # x = self.sa_head(x)  # apply one head of self_attention. (B,T,C)
        # x = self.ffwd(x)  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,C) this C is vocab size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  #  stretching the vec
            targets = targets.view(B * T)  # (32)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # takes (B,T) and generate fn generates (b,T+1,T+2,... )ie generate new token in time dim ie(contextlength dim)
        # idx is (B,T) array of indices
        for _ in range(max_new_tokens):
            #   get new prediction
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # returns(batch, time, embedding_dim) ie(B,T,C)->(1,1)->(1,1,65)
            # next focus only on the last time step
            logits = logits[:, -1, :]
            # logits = logits[:, -1, :] becomes (B,C) <-last element in the time dim
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # (B,1)ie(1,1)selects any one token from the probability values from C dim
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
            #  next = idx=[31,32,+...]
        return idx


model = BigramLanguageModel()
m = model.to(device)
#   def forward(self, idx, targets=None):
# ------------

# optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)


# ------------

# training loop


# for iter in tqdm(range(max_iters), desc="Training"):

#     # every once in a while evaluate loss  on train and val sets
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(
#             f"step {iter}: training loss {losses['train']:.4f},val loss {losses['val']:.4f}"
#         )
#         mlflow.log_metric("train_loss", losses["train"].item(), step=iter)
#         mlflow.log_metric("val_loss", losses["val"].item(), step=iter)

#     # -----
#     # sample a batch of data
#     xb, yb = get_batch("train")

#     #   def forward(self, idx, targets=None):
#     # evaluate the loss
#     logits, loss = m(xb, yb)

#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()


# ------------
# generated_text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode_txt(
    m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
)


# ------------
sample_path = "./dataset_research_paper_docs/generated_text.txt"
with open(sample_path, "w", encoding="utf-8") as f:
    f.write(generated_text)


# ------------

# ------------
