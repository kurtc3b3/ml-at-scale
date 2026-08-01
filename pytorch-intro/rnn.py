"""
Recurrent networks in PyTorch: RNN vs LSTM vs GRU, side by side.

Recurrent nets process a SEQUENCE one step at a time, carrying a hidden
"memory" state forward. The three variants differ only in how they manage
that memory:

  - RNN  : the plain version. One hidden state, updated each step. Simple,
           but its memory can fade over long sequences (vanishing gradients).
  - LSTM : adds a separate cell state + input/forget/output gates that learn
           what to keep vs discard. Strong long-term memory, most parameters.
  - GRU  : a streamlined LSTM (fewer gates, no separate cell state). Nearly
           as capable, a bit lighter/faster.

THE TASK: next-value prediction on sine waves. Each sample is a window of
`SEQ_LEN` points from a sine wave with random frequency and phase; the target
is the very next point. This is a clean regression task that ALL THREE learn
well, so it's a fair side-by-side of the architectures.

A note on comparisons: on a task this size the three land close together, and
which "wins" shifts with learning rate, epochs, and seed. That is itself the
lesson -- architecture choice matters far less than training dynamics until
sequences get long and hard. So we just report each model's final error.
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

SEQ_LEN = 30
N_TRAIN, N_TEST = 4000, 1000

# Fixed generator so every model sees identical data -> fair comparison.
gen = torch.Generator().manual_seed(0)


def make_data(n):
    # Random frequency and phase per sample.
    freq = torch.rand(n, 1, generator=gen) * 0.4 + 0.1      # 0.1 .. 0.5
    phase = torch.rand(n, 1, generator=gen) * 2 * math.pi
    # Time steps 0..SEQ_LEN (last one is the target).
    t = torch.arange(SEQ_LEN + 1).float().unsqueeze(0)      # (1, SEQ_LEN+1)
    wave = torch.sin(freq * t + phase)                      # (n, SEQ_LEN+1)
    x = wave[:, :SEQ_LEN].unsqueeze(-1)                     # (n, SEQ_LEN, 1)
    y = wave[:, SEQ_LEN].unsqueeze(-1)                      # (n, 1) next value
    return x, y


x_train, y_train = make_data(N_TRAIN)
x_test, y_test = make_data(N_TEST)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=256)


# --- One model, swappable recurrent core --------------------------------
class SequencePredictor(nn.Module):
    def __init__(self, cell_type, input_size=1, hidden_size=32):
        super().__init__()
        # nn.RNN / nn.LSTM / nn.GRU share almost the same constructor.
        # batch_first=True -> input shape is (batch, seq_len, features).
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)   # hidden state -> one number

    def forward(self, x):
        # RNN/GRU return (output, h_n); LSTM returns (output, (h_n, c_n)).
        # `output` is every timestep's hidden state: (batch, seq_len, hidden).
        output, _ = self.rnn(x)
        last_step = output[:, -1, :]     # summary after the whole sequence
        return self.head(last_step)


def test_loss(model, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            total += loss_fn(model(xb), yb).item() * len(xb)
    return total / N_TEST


def train(cell_type, epochs=20):
    model = SequencePredictor(cell_type).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {cell_type}  ({n_params:,} params) ===")

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:2d}/{epochs} | test MSE {test_loss(model, loss_fn):.5f}")
    print()
    return test_loss(model, loss_fn)


final = {}
for cell in ("RNN", "LSTM", "GRU"):
    final[cell] = train(cell)

print("Final test MSE (lower = better at predicting the next point):")
for cell, mse in final.items():
    print(f"  {cell:5} {mse:.5f}")
print("\nAll three learn this task. They swap in with a single word change")
print("(nn.RNN / nn.LSTM / nn.GRU) -- that's the practical takeaway.")
