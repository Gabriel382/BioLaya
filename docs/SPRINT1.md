# Sprint 1 — reproducible benchmark foundation

## Goal

Represent the same biomedical benchmark family used by BioJev as Laya-native typed decisions so model-family comparisons reuse the same source tasks and metrics.

## Datasets

- BioNLI — NLI as a two-option `choice` decision (entailment / contradiction).
- NLI4CT — clinical-trial NLI as a two-option `choice` decision.
- ChemProt — relation type over annotated entity pairs as `choice`.
- DDI2013 — relation type over annotated entity pairs as `choice`.
- BioRED — relation type over annotated entity pairs as `choice`.

Relation datasets intentionally start from gold annotated relation pairs; relation detection / NO_RELATION negative generation is a later controlled ablation rather than silently changing the task.

## Commands

```bash
python scripts/download_datasets.py --all
python scripts/evaluate.py --dataset-file data/processed/bionli/test.jsonl --max-examples 50
```

Outputs include predictions, confidence and classification/calibration metrics. Sprint 5 will extend calibration analysis substantially.
