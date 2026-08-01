"""
MNIST digit classification on the Apple Silicon GPU (MPS).

Step up from torch1.py:
  - a real dataset (28x28 handwritten digit images, 10 classes)
  - mini-batches via DataLoader instead of training on everything at once
  - classification loss (CrossEntropyLoss) instead of regression
  - a train/test split so we measure accuracy on unseen data

The dataset (~10 MB) downloads once to ./data on the first run.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Training on device: {device}")

# --- Data ---------------------------------------------------------------
# ToTensor() turns each image into a (1, 28, 28) float tensor in [0, 1].
transform = transforms.ToTensor()

train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

# DataLoader hands out data in shuffled mini-batches of 128 images.
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

print(f"Train images: {len(train_set)} | Test images: {len(test_set)}")

# --- Model --------------------------------------------------------------
# Flatten the 28x28 image to 784 inputs, then a small MLP -> 10 class scores.
model = nn.Sequential(
    nn.Flatten(),            # (batch, 1, 28, 28) -> (batch, 784)
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),      # 10 outputs = one score per digit 0-9
).to(device)

loss_fn = nn.CrossEntropyLoss()   # standard loss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate():
    """Return accuracy (%) on the held-out test set."""
    model.eval()
    correct = 0
    with torch.no_grad():                      # no gradients needed for eval
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)   # pick highest-scoring class
            correct += (preds == labels).sum().item()
    return 100.0 * correct / len(test_set)


# --- Training loop ------------------------------------------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:        # one mini-batch at a time
        images, labels = images.to(device), labels.to(device)

        preds = model(images)                  # forward
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()                  # the same 3-step update
        loss.backward()                        # as torch1.py, now batched
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    acc = evaluate()
    print(f"epoch {epoch + 1}/{EPOCHS} | train loss {avg_loss:.4f} | test accuracy {acc:.2f}%")

print("\nDone. A simple MLP typically reaches ~97% on MNIST in a few epochs.")
