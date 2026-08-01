"""
Capstone: tuning a PyTorch model with Ray Tune + Optuna.

This unites both threads of everything so far:
  - ray4.py tuned a scikit-learn model with Ray Tune + Optuna.
  - torch1-7 built PyTorch neural nets.
Here Optuna chooses a NEURAL NET's hyperparameters (learning rate, width,
depth) and Ray Tune trains many candidate networks in parallel across your
cores, then reports the best.

THE TASK (kept small on purpose): fit y = sin(x) with an MLP, like torch1.py.
Each trial is a full little training run; we tune the architecture around it.

A DELIBERATE DEVICE CHOICE:
  Hyperparameter search parallelizes across TRIALS, not within one model. Ray
  runs each trial in its own process; many small nets on separate CPU cores
  finishes faster (and avoids several processes contending over the single
  MPS GPU) than serializing everything onto one GPU. So each trial uses CPU
  here -- for HPO of small models that is usually the right call. You'd switch
  to GPU per trial only when a single model is big enough to need it.
"""

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch


def objective(config):
    # Imports live INSIDE the trial so each Ray worker process has them.
    import torch
    import torch.nn as nn

    device = "cpu"  # see the module docstring for why CPU per trial here

    # --- Data: y = sin(x), deterministic split into train / validation ---
    x = torch.linspace(-6.28, 6.28, 1000).unsqueeze(1)
    y = torch.sin(x)
    x_train, y_train = x[0::2].to(device), y[0::2].to(device)   # even indices
    x_val, y_val = x[1::2].to(device), y[1::2].to(device)       # odd indices

    # --- Build an MLP from the sampled hyperparameters -------------------
    hidden = config["hidden"]
    layers = []
    in_dim = 1
    for _ in range(config["num_layers"]):
        layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
        in_dim = hidden
    layers += [nn.Linear(in_dim, 1)]
    model = nn.Sequential(*layers).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # --- Train, then report validation error back to Tune/Optuna ---------
    for _ in range(400):
        loss = loss_fn(model(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_mse = loss_fn(model(x_val), y_val).item()

    # Optuna uses this metric to steer which hyperparameters to try next.
    tune.report({"val_mse": val_mse})


ray.init()

# The search SPACE (ranges), not a grid. Optuna samples intelligently within it.
param_space = {
    "lr": tune.loguniform(1e-4, 1e-1),          # log scale: 0.0001 .. 0.1
    "hidden": tune.choice([16, 32, 64, 128]),   # width of each hidden layer
    "num_layers": tune.choice([1, 2, 3]),       # depth of the network
}

tuner = tune.Tuner(
    tune.with_resources(objective, resources={"cpu": 1}),   # 1 core/trial -> many run at once
    tune_config=tune.TuneConfig(
        search_alg=OptunaSearch(),   # the Optuna brain picks hyperparameters
        metric="val_mse",
        mode="min",                  # lower validation MSE is better
        num_samples=24,              # 24 candidate networks, tuned in parallel
    ),
    param_space=param_space,
)

results = tuner.fit()

best = results.get_best_result(metric="val_mse", mode="min")
print("\nBest neural-net hyperparameters found by Optuna:")
for k, v in best.config.items():
    print(f"  {k:11} = {v}")
print(f"  val_mse     = {best.metrics['val_mse']:.6f}")
print("\nRay trained 24 candidate networks across your cores; Optuna decided")
print("which architectures to try. Swap in any torch1-7 model + its search")
print("space and the exact same harness tunes it.")

ray.shutdown()
