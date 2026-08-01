"""
The Transformer: self-attention, the architecture behind modern LLMs.

RNN/LSTM/GRU (torch4.py) read a sequence step by step, so information from
early steps must survive a long chain to reach the end. A Transformer instead
lets every position look DIRECTLY at every other position in one shot, via
"self-attention." No recurrence, fully parallel, and no long-range fade --
this is why it replaced RNNs and powers GPT, Claude, etc.

THE TASK: a genuine long-range memory test that a plain RNN struggles with.
Each input is a sequence of random numbers; the label is the sign of the
FIRST element. The network must carry that one fact across the whole sequence.
Self-attention reaches position 0 from the output in a single hop, so it
handles this cleanly.

Two ideas make it work on sequences:
  1. Positional encoding -- attention alone is order-blind, so we add a signal
     that tells the model WHERE each element sits in the sequence.
  2. Multi-head attention -- several attention "views" run in parallel, each
     free to focus on different relationships.
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

SEQ_LEN = 40
N_TRAIN, N_TEST = 4000, 1000
gen = torch.Generator().manual_seed(0)


def make_data(n):
    x = torch.rand(n, SEQ_LEN, 1, generator=gen) * 2 - 1   # (n, SEQ_LEN, 1)
    y = (x[:, 0, 0] > 0).long()                            # label = sign of first element
    return x, y


x_train, y_train = make_data(N_TRAIN)
x_test, y_test = make_data(N_TEST)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=256)


class PositionalEncoding(nn.Module):
    """Add a fixed sinusoidal 'where am I' signal to each position."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        # Project each scalar input up to the model's working width d_model.
        self.embed = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model)

        # A stack of standard Transformer encoder layers. Each contains
        # multi-head self-attention + a small feed-forward network.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,                 # number of parallel attention heads
            dim_feedforward=64,
            batch_first=True,            # input shape (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)                # (batch, seq_len, d_model)
        x = self.pos(x)                  # inject position info
        x = self.encoder(x)             # self-attention: every pos sees every pos
        x = x.mean(dim=1)                # pool across the sequence
        return self.head(x)


model = TransformerClassifier().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Transformer parameters: {n_params:,}\n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    return 100.0 * correct / N_TEST


EPOCHS = 12
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1:2d}/{EPOCHS} | test accuracy {evaluate():.2f}%")

print("\nThis is the same 'remember the first element across 40 steps' task")
print("that stumped the plain RNN/LSTM in early runs. Self-attention reaches")
print("position 0 in a single hop, so long-range memory is not the bottleneck.")
