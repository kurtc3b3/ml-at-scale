"""
A causal (decoder) Transformer: a tiny GPT that generates text.

torch5.py built an ENCODER Transformer -- every position could see every other
position, good for classifying a whole sequence. A GPT is a DECODER: it is
trained to predict the NEXT token given only the tokens BEFORE it, then we let
it generate by feeding its own predictions back in.

The one crucial change from torch5.py is the CAUSAL MASK: position i is only
allowed to attend to positions <= i. Without it the model could "cheat" by
peeking at the answer (the next character) during training.

This is a character-level model: it learns to spell and structure text one
character at a time. Same core mechanism as real LLMs, just tiny.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Training on device: {device}\n")

# --- Tiny corpus --------------------------------------------------------
# A small repetitive text so a tiny model can clearly learn its structure.
TEXT = (
    "to be or not to be that is the question. "
    "whether tis nobler in the mind to suffer. "
    "the slings and arrows of outrageous fortune. "
) * 40

# Character-level vocabulary: map each unique char <-> an integer id.
chars = sorted(set(TEXT))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

print(f"Corpus length: {len(TEXT)} chars | vocabulary: {vocab_size} unique chars")

BLOCK = 32   # context length: how many previous chars the model conditions on
data = torch.tensor(encode(TEXT), dtype=torch.long)

# Build (context, next-char-for-each-position) training pairs.
# y is x shifted left by one, so the model learns "given chars so far, predict
# the following char" at EVERY position simultaneously.
xs, ys = [], []
for i in range(len(data) - BLOCK):
    xs.append(data[i : i + BLOCK])
    ys.append(data[i + 1 : i + BLOCK + 1])
xs, ys = torch.stack(xs), torch.stack(ys)
loader = DataLoader(TensorDataset(xs, ys), batch_size=64, shuffle=True)


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, block=BLOCK):
        super().__init__()
        self.block = block
        self.tok_embed = nn.Embedding(vocab_size, d_model)      # char -> vector
        self.pos_embed = nn.Embedding(block, d_model)           # position -> vector
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        # We use an encoder layer but feed it a causal mask -> behaves as a decoder.
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)              # -> score per next char

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)           # (B, T, d_model)
        # Causal mask: True where attention is DISALLOWED (future positions).
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        x = self.blocks(x, mask=mask)
        return self.head(self.ln(x))                            # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, start, n_new):
        self.eval()
        idx = torch.tensor([encode(start)], device=device)
        for _ in range(n_new):
            idx_cond = idx[:, -self.block :]                    # last BLOCK chars
            logits = self(idx_cond)[:, -1, :]                  # scores for next char
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)       # sample a char
            idx = torch.cat([idx, nxt], dim=1)
        return decode(idx[0].tolist())


model = MiniGPT(vocab_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"MiniGPT parameters: {n_params:,}\n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

EPOCHS = 8
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                                     # (B, T, vocab)
        # Flatten to (B*T, vocab) vs (B*T,) so every position contributes.
        loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    avg = total / len(loader)
    sample = model.generate("to be", n_new=60)
    print(f"epoch {epoch + 1}/{EPOCHS} | loss {avg:.3f} | sample: {sample!r}")

print("\nStarting from 'to be', the model generates one character at a time,")
print("each conditioned only on what came before (the causal mask). This is")
print("exactly how a real GPT works -- just scaled to billions of parameters,")
print("word-piece tokens, and vast training text.")
