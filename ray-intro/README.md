```sh
conda create -n mas_ray python=3.12 -y
conda activate mas_ray
pip install -r requirements.txt
```

Use Python 3.12 or 3.13. Do **not** install the full `anaconda` metapackage — it conflicts with the pinned deps.

`numpy>=2.1` is required for Python 3.13 wheels; `numpy==2.0.2` has no prebuilt wheel for 3.13 and will fail to compile from source.

Run scripts by filename, e.g. `python ray1.py`. Do not name scripts `ray.py` — that shadows the `import ray` package.
