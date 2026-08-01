import torch
import torch.nn as nn

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Training on device: {device}")

x = torch.linspace(-6.28, 6.28, 1000).unsqueeze(1).to(device)
y = torch.sin(x)

model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to(device)

print(f"Model parameters live on: {next(model.parameters()).device}")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch + 1:4d} | loss {loss.item():.6f}")
    
model.eval()
with torch.no_grad():
    test = torch.tensor([[0.0], [1.5708], [3.1416]]).to(device)
    out = model(test).cpu().squeeze()
    truth = torch.sin(test).cpu().squeeze()

print("\n        x      predicted    true(sin x)")
for xi, pi_, ti in zip([0.0, 1.5708, 3.1416], out.tolist(), truth.tolist()):
    print(f"  {xi:7.4f}   {pi_:9.4f}   {ti:9.4f}")
