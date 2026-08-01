"""
PyTorch starter on Apple Silicon (M4 Pro) using the MPS GPU.

Trains a small neural network to learn a simple nonlinear function.
The point is to show the full training loop AND confirm the model runs
on the "mps" device (your integrated GPU) rather than the CPU.
"""

import torch
import torch.nn as nn

# Pick the best available device. On your Mac this is "mps" (the GPU).
# CUDA users would see "cuda"; everyone else falls back to "cpu".
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Training on device: {device}")

# --- Data ---------------------------------------------------------------
# A toy regression problem: learn y = sin(x). 1000 points in [-2pi, 2pi].
# .to(device) moves the tensors onto the GPU's (unified) memory.
x = torch.linspace(-6.28, 6.28, 1000).unsqueeze(1).to(device)  # shape (1000, 1)
y = torch.sin(x)

# --- Model --------------------------------------------------------------
# A tiny multilayer perceptron: 1 -> 64 -> 64 -> 1 with ReLU activations.
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to(device)  # move the model's weights onto the GPU too

# Confirm the weights actually live on the GPU.
print(f"Model parameters live on: {next(model.parameters()).device}")

# --- Training setup -----------------------------------------------------
loss_fn = nn.MSELoss()                                  # mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training loop ------------------------------------------------------
# This is the canonical PyTorch loop you'll reuse everywhere:
#   forward -> compute loss -> zero grads -> backward -> step
for epoch in range(2000):
    pred = model(x)                 # forward pass
    loss = loss_fn(pred, y)         # how wrong are we?

    optimizer.zero_grad()           # clear old gradients
    loss.backward()                 # backprop: compute new gradients
    optimizer.step()                # nudge weights to reduce the loss

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch + 1:4d} | loss {loss.item():.6f}")

# --- Check the result ---------------------------------------------------
# Predict sin at a few points and compare to the true value.
model.eval()
with torch.no_grad():
    test = torch.tensor([[0.0], [1.5708], [3.1416]]).to(device)  # 0, pi/2, pi
    out = model(test).cpu().squeeze()   # move back to CPU to print
    truth = torch.sin(test).cpu().squeeze()

print("\n        x      predicted    true(sin x)")
for xi, pi_, ti in zip([0.0, 1.5708, 3.1416], out.tolist(), truth.tolist()):
    print(f"  {xi:7.4f}   {pi_:9.4f}   {ti:9.4f}")
