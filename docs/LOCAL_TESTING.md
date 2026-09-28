# Local testing before Google Colab

BioLaya supports the same device contract locally and in Colab:

- `cpu` — force CPU; works with no GPU.
- `cuda` — require an NVIDIA CUDA GPU; fails early if unavailable.
- `auto` — use CUDA when visible to PyTorch, otherwise CPU.

## Windows / PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"
```

## Level 1 — structural CPU smoke (no model download)

```powershell
python scripts\check_env.py --device cpu
python scripts\local_smoke.py --device cpu --skip-model
pytest -q
```

This validates package imports, schemas, CPU selection, and Drive-style checkpoint backup/restore using a temporary local directory.

## Level 2 — one real Laya inference on CPU

Requires internet the first time so Laya can be downloaded:

```powershell
python scripts\local_smoke.py --device cpu
```

This may be slow, but it runs only one synthetic typed-decision example.

## Level 3 — CUDA smoke (optional)

```powershell
python scripts\check_env.py --device cuda
python scripts\local_smoke.py --device cuda
```

If CUDA is unavailable, `--device cuda` fails immediately instead of silently falling back to CPU.

## Small real-data evaluation locally

```powershell
python scripts\download_datasets.py bionli nli4ct
python scripts\evaluate.py `
  --dataset-file data\processed\bionli\test.jsonl `
  --device cpu `
  --max-examples 5 `
  --output-dir results\local_cpu_smoke
```

Replace `cpu` by `cuda` on a CUDA machine.
