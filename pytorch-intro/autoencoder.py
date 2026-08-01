"""
Autoencoder: unsupervised learning by compression.

Everything so far (torch2-torch5) was SUPERVISED -- we had labels and the
network learned input -> label. An autoencoder needs NO labels. It learns to:

    input  --encoder-->  tiny "latent" code  --decoder-->  reconstruction

and is trained to make the reconstruction match the ORIGINAL input. To do that
through a narrow bottleneck, the network is forced to discover the essential
structure of the data (here: what makes a digit look like a digit).

Uses on MNIST: dimensionality reduction, denoising, anomaly detection, and the
conceptual ancestor of generative models (VAEs, diffusion latents).

We squeeze each 784-pixel image down to just LATENT_DIM numbers and back.
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

LATENT_DIM = 16   # the bottleneck: 784 pixels -> 16 numbers -> 784 pixels

transform = transforms.ToTensor()
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder: shrink 784 -> 128 -> latent_dim.
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Decoder: grow latent_dim -> 128 -> 784, back to a valid image.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),            # pixels are in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)          # compress to the latent code
        out = self.decoder(z)        # reconstruct
        return out.view(-1, 1, 28, 28)

    def encode(self, x):
        return self.encoder(x)


model = Autoencoder().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Autoencoder parameters: {n_params:,}")
print(f"Compressing each image: 784 pixels -> {LATENT_DIM} numbers "
      f"({784 / LATENT_DIM:.0f}x smaller)\n")

# The target IS the input -> MSE between reconstruction and original.
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def test_recon_error():
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, _ in test_loader:            # labels ignored -- unsupervised!
            images = images.to(device)
            total += loss_fn(model(images), images).item() * len(images)
    return total / len(test_set)


EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    for images, _ in train_loader:               # note: we throw the labels away
        images = images.to(device)
        recon = model(images)
        loss = loss_fn(recon, images)            # compare to the input itself
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1}/{EPOCHS} | test reconstruction MSE {test_recon_error():.5f}")

# --- Show what the bottleneck learned -----------------------------------
# Encode 10 test images and report the average per-pixel reconstruction error.
model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)
    codes = model.encode(images[:10])                     # the 16-number codes
    recon = model(images[:10])
    per_pixel = (recon - images[:10]).abs().mean().item()

print(f"\nExample: 10 images each compressed to {LATENT_DIM} numbers.")
print(f"Average per-pixel reconstruction error: {per_pixel:.4f} (pixels are 0-1).")
print("The 16-number codes are a learned, compressed summary of each digit --")
print("found with NO labels, purely by forcing the data through the bottleneck.")
