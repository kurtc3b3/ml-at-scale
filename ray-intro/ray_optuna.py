"""
Smarter ML tuning with Ray Tune + Optuna.

Compare with ray3.py: that brute-forced a fixed 25-point grid. Here Optuna's
sampler (TPE / Bayesian) chooses *which* hyperparameters to try next, while
Ray Tune runs many trials in parallel across your cores. Fewer trials, and
they still run concurrently -> smart AND fast.
"""

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ray.init()

# Load the dataset once and share it via Ray's object store (zero-copy reads).
data = load_breast_cancer()
X_ref = ray.put(data.data)
y_ref = ray.put(data.target)


def objective(config):
    """One trial: train + cross-validate with the sampled hyperparameters."""
    X = ray.get(X_ref)
    y = ray.get(y_ref)

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        random_state=0,
    )
    score = cross_val_score(model, X, y, cv=5).mean()

    # Report the metric back to Tune so Optuna can steer the next trials.
    tune.report({"accuracy": score})


# The search *space* (ranges), not a fixed grid. Optuna samples within these.
param_space = {
    "n_estimators": tune.randint(10, 400),
    "max_depth": tune.randint(2, 32),
    "min_samples_split": tune.randint(2, 20),
}

tuner = tune.Tuner(
    # Give each trial 1 CPU so multiple run in parallel across your cores.
    tune.with_resources(objective, resources={"cpu": 1}),
    tune_config=tune.TuneConfig(
        search_alg=OptunaSearch(),   # <- the Optuna brain picks parameters
        metric="accuracy",
        mode="max",
        num_samples=30,              # 30 smart trials (vs 25 brute-force in ray3)
    ),
    param_space=param_space,
)

results = tuner.fit()

best = results.get_best_result(metric="accuracy", mode="max")
print("\nBest configuration found by Optuna:")
for k, v in best.config.items():
    print(f"  {k:18} = {v}")
print(f"  accuracy           = {best.metrics['accuracy']:.4f}")

ray.shutdown()
