# BioLaya five-benchmark repair

The original `prepare_five_benchmarks.py` assumed that every source dataset had train/dev/test and was already a typed `state/instructions/criteria/gold` decision. That is not the original BioJev Sprint 4 protocol.

Repaired conventions:

* **BioNLI:** read original `train`, reconstruct *exactly* the BioJev frozen test membership from `../BioJev/results/sprint4/frozen_4b/biojev-4b__bionli__test__seed42/predictions.jsonl` IDs. Only if all IDs match; otherwise fail. Create an internal development split from the remaining training pool.
* **NLI4CT:** use official `dev` as *evaluation*, and create a separate internal validation partition only from `train`. Do not invent official test.
* **ChemProt/BioRED:** adapt decision records and common NLI/relation-classification input formats; preserve original test and supplied validation.
* **DDI2013:** preserve the original test. Build an internal development split only from train.
* **Unrecognised schemas or integer labels:** fail closed; no guess at label-name mapping.

Run:

```bash
python scripts/audit_biojev_processed.py --root ../BioJev/data/processed
python scripts/prepare_five_benchmarks.py --root ../BioJev/data/processed --require-test
cat data/standard/five/audit.json
```

Every successful task writes train/dev/evaluation files and an audit manifest. Original test is additionally written as test.jsonl, except for NLI4CT, whose evaluation is official dev. Only start training when all tasks report READY. Keep previous BioJev baseline scores, but comparisons must use identical evaluation IDs (and disclose different fine-tuning and sequence limits).

After training all five classifiers:

```bash
python scripts/eval_five_benchmarks.py --split evaluation
```

**Important:** `train_standard.py` constructs a standard sequence-classification head, which is not the native Laya SystemOne decision head. These experiments should be reported as distinct model variants.
