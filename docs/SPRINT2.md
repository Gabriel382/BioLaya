# Sprint 2 — Biomedical domain adaptation (BioModernBERT)

Sprint 2 adapts the **ModernBERT-large backbone family used by Laya** to biomedical language using masked-language-model (MLM) continued pretraining on PubMed abstracts and PMC full text.

The output is intentionally an intermediate backbone, **BioModernBERT**, not yet the final BioLaya decision model. Sprint 3 will compare:

1. Laya -> biomedical typed-decision fine-tuning (No-DAPT baseline), and
2. BioModernBERT -> Laya decision training -> biomedical typed decisions (DAPT route).

That keeps biomedical DAPT as a measurable scientific variable.

## Corpus

Default mixture:

- 80% PubMed abstracts: `slinusc/PubMedAbstractsSubset`
- 20% PMC full text: `aochongoliverli/pmc_openaccess_split`

PMC rows are filtered to non-retracted documents and the conservative reusable license set `CC0`, `CC BY`, and `CC BY-SA`. PubMed/PMC documents use PMID when available for deterministic train/validation partitioning, so the same article cannot land in train as an abstract and validation as a full text.

The corpus is streamed; the full datasets are not downloaded before training.

## Presets

| preset | token budget | purpose |
|---|---:|---|
| `smoke` | 100K | end-to-end debugging |
| `1m` | 1M | short real training / Colab throughput benchmark |
| `10m` | 10M | substantial development BioModernBERT |
| `100m` | 100M | paper-candidate DAPT run |

All use sequence length 512 except the tiny `smoke` preset (128).

## Local CPU validation

This does **not** download ModernBERT:

```powershell
python -m pip install -e ".[train,dev]"
python scripts\check_env.py --device cpu
python scripts\sprint2_smoke.py --device cpu --preset smoke
python scripts\plan_dapt.py --preset 10m
pytest -q
```

Network/corpus test, still without loading the model:

```powershell
python scripts\inspect_corpus.py --preset smoke
```

A genuine CPU training smoke is also supported, but it downloads ModernBERT-large and is intentionally slow:

```powershell
python scripts\train_dapt.py --preset smoke --device cpu --resume none
```

## Local CUDA validation

```powershell
python scripts\check_env.py --device cuda
python scripts\sprint2_smoke.py --device cuda --preset smoke
python scripts\train_dapt.py --preset smoke --device cuda --resume none
```

`--device cuda` is strict and aborts if PyTorch cannot see CUDA. `--device auto` chooses CUDA when available and CPU otherwise.

## Google Colab + Drive

Canonical notebook: **`notebooks/02_sprint2_colab.ipynb`**.

Download/copy that notebook into Google Drive and open it with Colab. It does not depend on a repository clone. It installs the current GitHub project directly:

```python
%pip install -q --upgrade "biolaya[train,colab] @ git+https://github.com/Gabriel382/BioLaya.git"
```

The notebook trains from `/content` for speed and stores persistent recovery state under:

```text
/content/drive/MyDrive/BioLaya/checkpoints/sprint2/<preset>/
```

Every Trainer save is copied to Drive. `--resume auto` restores the newest Drive checkpoint to local `/content` and resumes from it. The final BioModernBERT export is also copied to Drive.

For the first Colab run use `PRESET="1m"` or `PRESET="10m"` to benchmark throughput. Switch to `100m` for the publication-scale candidate once the environment is stable.

## Output

Local:

```text
outputs/sprint2/biomodernbert_<budget>/final/
```

Drive:

```text
MyDrive/BioLaya/checkpoints/sprint2/<preset>/final/
```

The final directory includes the ModernBERT MLM checkpoint, tokenizer and `training_manifest.json`.
