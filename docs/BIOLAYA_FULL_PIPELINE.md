# BioLaya: reproducible extension of the original repository

## What is included

Two **different** architectures and evaluation families, both based on ModernBERT:

- **BioLaya Standard:** `AutoModelForSequenceClassification`, intended as a conventional NLI/relation classifier comparable to the original BioJev classifier only under identical datasets/splits/metrics. Does not run as native Ollama SystemOne.
- **BioLaya-SystemOne:** Laya 421M `encoder + typed-decision head`, trained with upstream `laya-train` and the same canonical examples BioJev SystemOne used. **Not Qwen3.5; not completion-only letter supervision.** Supports choice, noul, score as native Laya types.

The existing BioLaya Sprint 1/2 implementation is retained, including optional biomedical MLM DAPT. To initialize SystemOne's encoder from biomedical DAPT use `scripts/attach_biomedical_encoder.py`, which refuses incompatible weights/tokenizers. Treat that experiment as a **separate** DAPT branch and label it accordingly.

## 0. Dependencies

`python -m pip install -e '.[train,systemone,dev]'`

Use CUDA torch matched to the machine. Laya is pinned to 0.4.15, and training flags follow that release. Confirm `laya-train --help` on the environment; upstream flags can change. The old BioLaya 0.3.20 dependency provided inference but was missing this training CLI.

## 1. Reuse identical BioJev SystemOne examples

Copy/locate `data/sprint8/systemone_4b/train.jsonl` and `dev.jsonl` from the **existing** BioJev repository. You can share the files by path, no duplication needed except transformed outputs.

```bash
python scripts/prepare_comparable_data.py \
 --biojev-train ../BioJev/data/sprint8/systemone_4b/train.jsonl \
 --biojev-dev ../BioJev/data/sprint8/systemone_4b/dev.jsonl
python scripts/smoke_comparison.py
```

Expected from user-supplied BioJev history: 147,048 train / 6,768 dev and 11 development sources. **Verify the generated manifest**: the script never asserts these without reading the real files. It rejects identical examples leaking between splits. This step also writes upstream Laya-compatible `laya_train.jsonl` and `laya_dev.jsonl` with `expected` hard labels.

**Note:** same examples does not guarantee same truncation, prompts, or calibration. Laya's context budgets (512/1024) differ from BioJev (2048 plus adaptive extended contexts). For NLI4CT, record lengths and truncated/failed cases before any comparison.

## 2. Train BioLaya SystemOne

```bash
python scripts/train_systemone.py --dry-run
python scripts/train_systemone.py --config configs/systemone/biolaya.yaml
```

Output is a real Laya checkpoint containing `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`. It can be loaded with `laya.load(...)` and is compatible with the native Laya typed-decision API. Its calibration is fitted on a separate training-derived slice by upstream Laya. **Do not claim clinical calibration.**

### Optional biomedical-DAPT branch

```bash
python scripts/attach_biomedical_encoder.py \
  --laya-checkpoint PATH_TO_DOWNLOADED_LAYA_CHECKPOINT \
  --dapt-dir outputs/sprint2/biomodernbert_100m/final \
  --out outputs/biolaya/systemone/dapt_seed
```

The `--laya-checkpoint` must point to the full Laya checkpoint, not the MLM folder. If it passes strict shape/vocabulary checks, adjust `model.base` in a new training config to point to the DAPT seed and retrain. If it fails, investigate model/tokenizer compatibility; never use `strict=False` silently.

## 3. Evaluate on same 11 sources (BioJev frozen dev set)

```bash
python scripts/eval_comparable.py --backend laya \
  --models outputs/biolaya/systemone/final \
  --output results/comparison/biolaya_systemone --limit 55
python scripts/eval_comparable.py --backend laya \
  --models outputs/biolaya/systemone/final \
  --output results/comparison/biolaya_systemone_full
```

Outputs: `predictions.jsonl`, `summary.csv`, `run.json`, including per-source accuracy, macro-F1, error count, top-1 ECE, binary correctness Brier and genuine multiclass Brier/NLL when full class distributions are returned, median and p95 latency. Do not confuse the confidence field with calibrated accuracy probability.

## 4. Train standard BioLaya NLI classifiers

```bash
python scripts/prepare_standard.py --source bionli --out data/standard/bionli
python scripts/train_standard.py --config configs/standard/bionli.yaml
python scripts/eval_standard.py --checkpoint outputs/biolaya/standard/bionli/final \
 --data data/standard/bionli/dev.jsonl --output results/standard/bionli_dev

python scripts/prepare_standard.py --source nli4ct --out data/standard/nli4ct
python scripts/train_standard.py --config configs/standard/nli4ct.yaml
```

For cross-domain / RE benchmark tasks, build per-task classifiers using `scripts/prepare_standard.py --source YOUR_SOURCE --out ...` then a corresponding YAML with that dataset's class vocabulary, checked against actual data. BioJev Sprint 4's **ChemProt, DDI2013, BioRED** test splits are not identical to Sprint 8's 11-source decision development suite. These tasks require the original BioJev frozen evaluation dataset files and must not be faked from the 11-source subset.

### Important NLI label caveat

BioJev may encode label integers `0=contradiction, 1=entailment, 2=neutral`; the canonical textual labels are preserved instead. For the BioNLI/NLI4CT datasets, use label lists as explicitly configured, never rely on unverified integer conventions. Do not compare macro-F1 to accuracy.

## 4b. Five original BioJev biomedical test benchmarks (no baseline reruns)

Use exact original BioJev processed train/dev/test splits. The existing BioLaya Sprint1 prep can produce local versions, but its split construction may differ and **must not** be silently considered equivalent to BioJev.

```bash
python scripts/prepare_five_benchmarks.py \
 --root ../BioJev/data/processed --require-test
# Inspect data/standard/five/audit.json; STOP if a task is BLOCKED/MISSING_SPLIT.
for task in bionli nli4ct chemprot ddi2013 biored; do
 python scripts/train_standard.py --config "configs/standard/five/${task}.yaml"
done
python scripts/eval_five_benchmarks.py --split test
python scripts/import_existing_biojev_results.py --biojev-root ../BioJev
```

The generated audit exposes missing/invalid dataset files; some BioJev processed files may need their original preprocessor. Do not silently fabricate relation negatives or infer an unavailable test split. Compare with the saved BioJev Sprint4 frozen/supervised summaries only after checking identical example IDs and label space.

### General-NLI then biomedical-NLI stage (optional)

```bash
python scripts/prepare_multinli.py
python scripts/train_standard.py --config configs/standard/general_mnli.yaml
python scripts/train_standard.py --config configs/standard/biomedical_multinli.yaml
```

This runs a deliberate two-stage experiment. The biomedical stage uses a three-label head inherited from MNLI, even though BioNLI/NLI4CT may use only two gold classes; score those binary tasks separately and document the label protocol. To test DAPT, start general stage from the DAPT ModernBERT checkpoint and keep all other settings fixed.

## 5. Native Ollama (not a Python bridge)

Laya's **true** native Ollama support requires **Ollama >= 0.40.0**. You must convert the **entire trained Laya decision checkpoint** into a Laya-aware GGUF with the correct encoder *and* typed decision head. BioJev's Qwen3.5 GGUF converter and metadata `qwen35.decision.type=tev1` are **incompatible** with this architecture.

Use a Laya-compatible converter from a verified current llama.cpp/Ollama release and inspect the resulting GGUF. This project provides a transparent gate and registration commands but **does not bundle or pretend to implement the upstream converter**:

```bash
python scripts/export_native_ollama.py \
 --gguf /path/to/biolaya-laya-decision.gguf \
 --ollama-tag biolaya-systemone:421m --create
python scripts/test_native_ollama.py --model biolaya-systemone:421m
python scripts/compare_native_parity.py --samples 30
python scripts/eval_comparable.py --backend ollama \
 --models biolaya-systemone:421m \
 --output results/comparison/biolaya_ollama_full
```

The script checks that a GGUF is present, but only the end-to-end test and numerical parity against upstream `laya.load` confirm a working export. `ollama create` can reject unsupported GGUF architecture. **Do not publish unless create, typed decision E2E, and parity all pass.**

The *default* `num_ctx=1024` is configured at export; optional larger context profiles must be separate tags and should be compared carefully due to training context length.

## 6. Existing BioJev results: DO NOT rerun baselines

Export/read pre-existing BioJev Sprint4 and Sprint8 summary CSVs and join them without changing data:

```bash
python scripts/compare_existing.py \
 --biolaya results/comparison/biolaya_systemone_full/summary.csv \
 --biojev ../BioJev/results/sprint8/native_ollama_eval_full_repaired/summary.csv
```

Comparison carries `NOT_VERIFIED` protocol status. Only use `--confirmed-same-protocol` **after** checking dataset IDs, sample counts, labels, split, prompts, truncation and metrics. This is especially important for Standard BioLaya (Sprint4 test) vs SystemOne (Sprint8 dev).

## 7. Release checks

```bash
python scripts/smoke_comparison.py
python -m pytest -q
python scripts/model_card.py \
  --metrics results/comparison/biolaya_systemone_full/summary.csv \
  --model-name BioLaya-SystemOne
```

Do not publish claimed scores until actual BioLaya training/evaluation complete. Do not include unpublished dataset texts, trained checkpoints, HF access tokens, or other licensed external artifacts in GitHub commits.

## 8. Publishing after model training and validation

```bash
# Add/review README.md to the FINAL checkpoint directory before any upload.
python scripts/model_card.py --metrics results/comparison/biolaya_systemone_full/summary.csv \
 --out outputs/biolaya/systemone/final/README.md
python scripts/publish_huggingface.py --checkpoint outputs/biolaya/systemone/final \
 --repo-id Gabriel382/BioLaya-SystemOne
# Append --confirm only when ready and attribution/license/model card are correct.

# Ollama requires native Laya GGUF and parity checks already passed.
python scripts/publish_ollama.py --local biolaya-systemone:421m \
 --remote Gabriel382/biolaya-systemone:421m
# Append --confirm only after native E2E and parity PASS.
```

Avoid publishing raw biomedical texts, training data, proprietary weights or access tokens.
