# SEN2SR comparison (exploratory)

Isolated sandbox to compare our own `DSen2Net20m` (see `../../src/s2sr/`, `../../notebooks/04_train.ipynb`) against [ESAOpenSR/SEN2SR](https://github.com/ESAOpenSR/SEN2SR), an ESA-funded diffusion/Mamba-based Sentinel-2 super-resolution package. Not part of the main pipeline — nothing here is imported by `src/s2sr/` or the numbered notebooks, and no results from here feed back into the trained model or its outputs unless we explicitly decide to act on them.

**Why a separate folder, and a separate environment**: SEN2SR's dependencies (`mamba-ssm`, a specific CUDA/torch build) are heavier and more failure-prone to install than this project's own stack, and there's no reason to risk destabilizing the working `conda/pytorch` environment for an exploratory comparison. Set up a **new, separate conda env** for this (see below) — don't install these into the environment notebooks 1–6 depend on.

## Why this comparison

DSen2-style models (pixel-loss CNNs, including ours) tend toward conservative, somewhat smooth predictions. Diffusion-based approaches like SEN2SR can look sharper, but risk hallucinating plausible-but-inaccurate detail — a real concern for a geoAI pipeline that might do quantitative analysis downstream, not just visualization. SEN2SR's authors specifically target this (their "trustworthy" framing, pixel-level uncertainty maps), which is why it's the one worth actually testing rather than just reading about.

## Setup (on RAILS, in a fresh conda env — do this before running the notebook)

```bash
conda create -n sen2sr_compare python=3.11
conda activate sen2sr_compare
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install sen2sr mlstac
pip install pystac-client planetary-computer rasterio geopandas
```

This is the **lightweight path** (`SEN2SRLite`) — no `mamba-ssm` build step, which needs a fussier from-source compile. Start here; only chase the full Mamba-architecture install (see SEN2SR's README) if the lite version turns out not to expose the `Reference_RSWIR_x2` variant we actually want.

Deliberately skipping SEN2SR's suggested `cubo` package for data loading — reusing our own already-working `s2sr.stac`/`s2sr.boundaries` code (same pattern as every other notebook, `sys.path.insert(...)`) to fetch the identical scene instead, so there's only one unfamiliar API to debug (SEN2SR's model) instead of two.

## Status

Not yet run. `compare.ipynb` is a first attempt built from SEN2SR's documented API (I don't have network access to verify it against the live package/model download) — expect some real debugging on the first run, same as everything else in this project. Report back whatever actually happens.
