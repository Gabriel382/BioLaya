# Local testing

Every BioLaya sprint must be testable before moving to Colab.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"
```

## CPU-only structural tests

```powershell
python scripts\check_env.py --device cpu
python scripts\local_smoke.py --device cpu --skip-model
python scripts\sprint2_smoke.py --device cpu --preset smoke
python scripts\plan_dapt.py --preset 10m
pytest -q
```

These do not require a CUDA GPU and Sprint-2 structural tests do not download ModernBERT.

## Network/corpus test

```powershell
python scripts\inspect_corpus.py --preset smoke
```

This verifies that the current PubMed and PMC streaming sources can actually yield usable documents before a large model is loaded.

## Real CPU tests

Sprint 1 one-example Laya inference:

```powershell
python scripts\local_smoke.py --device cpu
```

Sprint 2 real ModernBERT MLM smoke (large download and slow on CPU):

```powershell
python scripts\train_dapt.py --preset smoke --device cpu --resume none
```

## CUDA tests

```powershell
python scripts\check_env.py --device cuda
python scripts\local_smoke.py --device cuda
python scripts\sprint2_smoke.py --device cuda --preset smoke
python scripts\train_dapt.py --preset smoke --device cuda --resume none
```

Explicit CUDA mode fails immediately when CUDA is unavailable.
