# Split contamination audit: BioJev Sprint 8 → BioLaya

**Status:** This is a code fix and reproducibility tool, **not** a claim that the
original BioJev results are clean. The source BioJev Sprint 8 dataset triggered
`TRAIN/DEV LEAKAGE: 468 overlapping content fingerprints` on one server run.
That is the number of **distinct normalized inputs**, not necessarily the
number of affected training rows or development rows. Audit counts must be read
from the source files on the server.

## Why it happens

The old `build()` threw an error whenever it discovered an identical
`(source, state, question, sorted candidate key/description pairs)` in train
and dev. It correctly **ignored the gold label and example ID** when making
a content fingerprint. Option shuffling does not eliminate overlap.
BioJev's own Sprint 8 loader can enumerate training and development examples
with the same source-local IDs, so **matching IDs alone cannot establish
leakage**. Conversely, different IDs do not remove an actual content overlap.

## 1. Audit without producing data

```bash
python scripts/prepare_comparable_data.py \
  --biojev-train ../BioJev/data/sprint8/systemone_4b/train.jsonl \
  --biojev-dev ../BioJev/data/sprint8/systemone_4b/dev.jsonl \
  --audit-only
```

The command creates these files in `data/comparable/sprint8/`:
- `overlap_audit.json` — counts and per-source summary, original SHA256 hashes
- `overlap_examples.csv` — one row per overlapping unique content fingerprint,
  with source, train/dev counts, example IDs and possible contradictory labels
- `removed_train_rows.csv` — which **training rows** would be removed by the
  explicit sanitization; audit-only does not actually remove them

It does not write the converted train/dev examples.

## 2. Produce a clean BioLaya training split, preserving BioJev dev

```bash
python scripts/prepare_comparable_data.py \
  --biojev-train ../BioJev/data/sprint8/systemone_4b/train.jsonl \
  --biojev-dev ../BioJev/data/sprint8/systemone_4b/dev.jsonl \
  --overlap-policy drop-train
```

`drop-train` is an **explicit opt-in**. The default is still fail-closed. The
new conversion removes **all contaminated training rows**, while retaining
all development rows, original row order, answer keys, and questions. Counts,
SHA256 checksums and provenance go in `manifest.json`; overlap must be 0.
It also writes both canonical `train.jsonl`/`dev.jsonl` and Laya-native
`laya_train.jsonl`/`laya_dev.jsonl`.

## 3. Interpretation and publication policy

- The cleaned BioLaya train set is **not identical** to what previously trained
  BioJev-SystemOne. Keeping the development split fixed allows per-item
  comparisons but **does not** equate the training data or eliminate BioJev's
  original contamination.
- The old BioJev validation results (92.82% overall 4B, 92.66% 9B) may be
  optimistically biased on overlapping examples. Do not make a head-to-head
  SOTA claim from those development scores until the overlaps are accounted
  for, or all models are measured on a genuinely untouched test set.
- Do not assume the 468 fingerprints correspond to 468 dev rows. Use
  `overlapping_train_rows` and `overlapping_dev_rows` from the audit.
- Comparison against original BioJev scores for **non-overlapping dev examples**
  is a useful supplementary analysis; for primary scientific results prefer
  the same held-out data and properly leakage-free baselines.
- Removing overlap is not a substitute for checking near-duplicates,
  paraphrases, source-specific split leakage and calibration/test reuse.

## Environment

No new dependencies are needed. The patch works with the existing project
`pyproject.toml` and preserves previously prepared BioLaya models and other
benchmark workflows.
