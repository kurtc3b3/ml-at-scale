"""
Simple ML with Ray: parallel hyperparameter search.

We train many RandomForest models with different hyperparameters at the
same time, one per CPU core, then pick the best. This is the classic case
where Ray helps: each trial is independent, so they run truly in parallel.
"""

import time

import ray
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Start a local Ray cluster (uses all your cores).
ray.init()

# Load a small built-in dataset once, then put it in Ray's shared-memory
# object store. Every worker reads it zero-copy instead of re-pickling it.
data = load_breast_cancer()
X_ref = ray.put(data.data)
y_ref = ray.put(data.target)


@ray.remote  # each call runs in its own worker process, on its own core
def train_and_eval(n_estimators, max_depth):
    # Ray automatically resolves the object refs back into the real arrays.
    X = ray.get(X_ref)
    y = ray.get(y_ref)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=0,
    )
    # 5-fold cross-validation accuracy.
    score = cross_val_score(model, X, y, cv=5).mean()
    return {"n_estimators": n_estimators, "max_depth": max_depth, "accuracy": score}


# The grid of hyperparameters to search.
grid = [
    (n, d)
    for n in (10, 50, 100, 200, 400)
    for d in (2, 4, 8, 16, None)
]

print(f"Launching {len(grid)} training jobs across your cores...\n")
start = time.time()

# Fire off every trial at once — returns immediately with futures.
futures = [train_and_eval.remote(n, d) for n, d in grid]

# Block until all trials finish, gathering their results.
results = ray.get(futures)

elapsed = time.time() - start

# Report the best configuration.
best = max(results, key=lambda r: r["accuracy"])
print(f"Ran {len(results)} trials in {elapsed:.1f}s\n")
print("Best configuration:")
print(f"  n_estimators = {best['n_estimators']}")
print(f"  max_depth    = {best['max_depth']}")
print(f"  accuracy     = {best['accuracy']:.4f}")

ray.shutdown()
