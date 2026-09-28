# BioLaya v0.2.0 — cumulative Sprints 1 + 2

This release contains the complete Sprint-1 benchmark foundation and Sprint-2 biomedical domain-adaptation stack.

## Canonical Colab notebooks

- `notebooks/01_sprint1_colab.ipynb`
- `notebooks/02_sprint2_colab.ipynb`

Download/copy the notebook into Google Drive and open it with Colab. Each notebook installs the current project directly from `https://github.com/Gabriel382/BioLaya`.

## Sprint 2 quick start

Local structural test:

```powershell
python -m pip install -e ".[train,dev]"
python scripts\sprint2_smoke.py --device cpu --preset smoke
python scripts\plan_dapt.py --preset 10m
pytest -q
```

Colab training uses `02_sprint2_colab.ipynb`, Google Drive checkpoints and `--resume auto`.

Presets: `smoke`, `1m`, `10m`, `100m`.
