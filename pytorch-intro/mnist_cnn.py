"""
MNIST with a Convolutional Neural Network (CNN) on the Apple Silicon GPU.

Step up from torch2.py, which flattened each image into 784 numbers and lost
all spatial structure. A CNN instead slides small learnable filters across the
image, so it detects local patterns (edges, curves, strokes) wherever they
appear. This is THE core building block of computer vision.

Same dataset and training loop as torch2.py -- only the model changes, and
accuracy jumps from ~96.5% to ~99%.
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

# --- Data (identical to torch2.py) --------------------------------------
transform = transforms.ToTensor()
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)


# --- Model: a small CNN -------------------------------------------------
# Written as a class (the standard PyTorch style) so we can describe the
# two stages: convolutional feature extraction, then a classifier head.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d(in_channels, out_channels, kernel_size): learnable filters.
        # MaxPool2d(2) halves the height/width, keeping the strongest signals.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 -> 32 feature maps, 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 -> 64 maps, 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 64 x 7 x 7
        )
        # Classifier head: flatten the feature maps, then map to 10 classes.
        self.classifier = nn.Sequential(
            nn.Flatten(),                # 64 * 7 * 7 = 3136 features
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),            # randomly zero 25% of units -> less overfitting
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)

# Show the model's size -- CNNs get high accuracy with relatively few weights.
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / len(test_set)


# --- Training loop (identical shape to torch2.py) -----------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    acc = evaluate()
    print(f"epoch {epoch + 1}/{EPOCHS} | train loss {avg_loss:.4f} | test accuracy {acc:.2f}%")

# --- Save the trained model so it can be reused -------------------------
torch.save(model.state_dict(), "mnist_cnn.pt")
print("\nSaved trained weights to mnist_cnn.pt")
print("A CNN typically reaches ~99% on MNIST -- notably better than the MLP.")
