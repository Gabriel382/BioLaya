# BioLaya on Google Colab

BioLaya is Colab-first but not Colab-only.

## Canonical notebooks

- Sprint 1: `notebooks/01_sprint1_colab.ipynb`
- Sprint 2: `notebooks/02_sprint2_colab.ipynb`

The `.ipynb` files live in the ZIP and GitHub repository. The intended workflow is to download/copy the relevant notebook into **Google Drive**, open it with **Google Colab**, and execute it there.

The notebook installs BioLaya directly from GitHub; a clone is not required:

```python
%pip install -q --upgrade "biolaya[train,colab] @ git+https://github.com/Gabriel382/BioLaya.git"
```

The `colab` extra aligns `fsspec==2025.3.0` and `gcsfs==2025.3.0` to avoid the dependency conflict observed in the standard Colab environment.

## Device policy

Use `DEVICE="cuda"` for real Colab training. CUDA mode is strict: if Colab did not actually assign a GPU, BioLaya aborts instead of silently training on CPU.

`DEVICE="auto"` is useful for portable notebooks/tests; it selects CUDA if PyTorch sees it and CPU otherwise.

## Storage policy

Train from `/content` for fast local VM I/O. Persist only expensive/recovery artifacts to Drive:

```text
MyDrive/BioLaya/
├── checkpoints/
├── logs/
├── results/
├── final_models/
└── manifests/
```

Sprint 2 automatically backs up Trainer checkpoints and restores the newest one with `--resume auto`.
