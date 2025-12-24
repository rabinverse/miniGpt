import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time
import mlflow
from datetime import datetime

# Hyperparameters
block_size = 8  # context length of input
batch_size = 32  # no of input sequence to process in parallel
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
# device="mps" if torch.mps.is_available() else "cpu"
eval_iters = 200
max_new_tokens = 600
n_embd = 32


start = time.time()
# ------------
torch.manual_seed(1337)

with open("./dataset_research_paper_docs/input_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ------------


chars = sorted(list(set(text)))
vocab_size = len(chars)


# ------------ mlflow
mlflow.set_experiment("gpt_train")
run_name = (
    f"with_posemb_bigram_bs{batch_size}_emb{n_embd}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
mlflow.start_run(run_name=run_name)
mlflow.log_params(
    {
        "block_size": block_size,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "learning_rate": learning_rate,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "device": device,
    }
)
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


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # 65*65
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # language modelling head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and target are both (B,T)tensor of integer B-batch ,T-time/block_size/context length, C-channel. (here b=4,T=8,C=vocabsize ie 65)

        tok_emb = self.token_embedding_table(idx)
        # tok_emb becomes (B,T,C)ie(4,8,n_embd) c is n_embd
        # ---add posembedding to the token embedding
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        # pos_emb returns # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)

        logits = self.lm_head(x)  # (B,T,C) this C is vocab size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (32*65) stretching the vec
            targets = targets.view(B * T)  # (32)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # takes (B,T) and generate work is to generate (b,T+1,T+2)ie generate new token in time dim ie(contextlength dim)
        # idx is (B,T) array of indices in the current context(1,1)
        for _ in range(max_new_tokens):
            #   get new prediction
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # returns(batch, time, embedding_dim) ie(B,T,C)->(1,1)->(1,1,65)
            # focus only on the last time step
            logits = logits[
                :, -1, :
            ]  # becomes (B,C) <-last element in the time dim,,,just one time dim so selects that whole tensor(1,1)->(1,1,65)->(1,65)
            # applying softmax to get probabilities form logits
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # (B,1)ie(1,1)selects any one token from the probability values from 65 of them
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
            # eg next = idx=[31,32]
        return idx


model = BigramLanguageModel()
m = model.to(device)
# logits, loss = m(xb, yb)
# print("logits", logits.shape, "\n loss= ", loss)


# --------
# generate
# idx = torch.zeros((1, 1), dtype=torch.long)
# 0 index in vocab represents \n
# PyTorch expects a batch dimension in tensors, so even a single sequence must be shaped as (B, T) rather than just (T).
# print("idx begin------")
# print("idx=", idx)
# print("idxshape", idx.shape)
# ret_idx = m.generate(idx, max_new_tokens=100)[0].tolist()
# print("ret_idx=", ret_idx)
# print("len=", len(ret_idx))
# print("----\n generated_text -> ", decode_txt(ret_idx))
# print(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist())


# ------------

optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)


# ------------

# training loop


for iter in tqdm(range(max_iters), desc="Training"):

    # every once in a while evaluate loss  on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: training loss {losses['train']:.4f},val loss {losses['val']:.4f}"
        )
        mlflow.log_metric("train_loss", losses["train"].item(), step=iter)
        mlflow.log_metric("val_loss", losses["val"].item(), step=iter)

    # -----
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode_txt(
    m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
)
print(generated_text)

# ------------
sample_path = "./dataset_research_paper_docs/generated_text.txt"
with open(sample_path, "w", encoding="utf-8") as f:
    f.write(generated_text)

# Track with MLflow
mlflow.log_artifact(sample_path)

# ------------


# ------------


# ------------


# ------------


# ------------

mlflow.end_run()
# ------------

# step 29700: training loss 2.4542,val loss 2.4862. time 20.746973037719727
# step 29700: training loss 2.4542,val loss 2.4862 tm 67.80790400505066 gp

# step 2700: training loss 2.8731,val loss 2.8799
# step 2700: training loss 2.4959,val loss 2.5076

print("*" * 20, "The end", time.time() - start)
