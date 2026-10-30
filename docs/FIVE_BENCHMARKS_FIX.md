# BioLaya v0.3.6: preserving BioJev biomedical benchmark protocols

This patch fixes two demonstrated limitations of v0.3.4. **The code has not been validated on the private BioJev data directory; verify the resulting audit before starting a long GPU run.**

## BioNLI held-out evaluation: no train/dev/test duplication

- The exact **BioJev Sprint 4 BioNLI frozen-test ID membership** is preserved (3,599 expected) from the existing predictions file; gold labels come from the underlying BioNLI data. It is not a new random test set.
- Before deriving an internal validation set, other training rows with identical (source, text, question) to a frozen evaluation item are removed, even when they have another ID. In the reported v0.3.4 run, this is needed because the internal development set overlapped evaluation by 89 content fingerprints. The exact count of removed training rows must be measured on your server.
- The internal validation split is now performed **by identical input groups**, preventing duplicate examples from crossing train and validation. Evaluation is never used for model selection.

## ChemProt / DDI2013 / BioRED entity pairs

- The original BioLaya BigBio relation loader represents a target through `metadata.entity1`, `metadata.entity2`, `metadata.arg1_id`, `metadata.arg2_id`. The previous v0.3.4 converter ignored `metadata`, conflating separate gold entity pairs that share the same article and surface names.
- v0.3.5 reads this target metadata, appends the source annotated **pair IDs** to the question (when present), retains pair annotation metadata for audit, and checks that contradictory gold labels no longer refer to an identical input. The IDs **do not by themselves locate token spans** in context; results must be interpreted accordingly.
- For relation corpora, any internal validation split is grouped by **whole document context** (all pairs from the same document together). The cross-split contamination filter is document-level, not pair-level.
- If the same annotated input still has different gold labels, the script intentionally **blocks**. That may be a genuine multi-label annotation or missing mention offsets. A single-label classifier is inappropriate until the task is correctly specified. It does not invent entity spans, drop gold classes, or relabel examples.

## Commands

```bash
cd ~/git/postdoc/BioLaya
unzip -o BioLaya_Five_Benchmarks_Fix_v0.3.5.zip -d .
python scripts/audit_biojev_processed.py --root ../BioJev/data/processed
python scripts/prepare_five_benchmarks.py --root ../BioJev/data/processed --require-test
cat data/standard/five/audit.json
```

The audit prints input and metadata **field names**, counts, and conflict counts, not biomedical document contents. If any task is still BLOCKED, inspect the diagnostic message and schema audit before attempting training; prior output from other runs may remain and should not be treated as valid merely because files exist.

**Comparison warning:** BioLaya supervised per-task fine-tuning is not the same experimental regime as the *frozen* BioJev 4B classifier, even when they use identical evaluation examples. Compare only matched task/sample/label protocols and explicitly disclose supervision differences.

## v0.3.6 — Correct BioJev processed relation rows

The original BioJev processed JSONL **has** target-entity columns:
`context`, `subject`, `object`, `subject_type`, and `object_type`. The
BioLaya v0.3.5 converter accidentally ignored `subject` and `object`,
because it only considered the legacy BioLaya BigBio format's `entity1`,
`entity2`, `arg1_id`, and `arg2_id` metadata. As a result it generated
identical input strings from different annotated pairs in the same document.

The new adapter explicitly builds a pair-conditioned question from
`subject`, `object` and optional entity types. It retains source labels,
ID membership, all official evaluation instances, and document-level
train/internal-dev grouping. It does **not** use the relation-row ID, gold
label or candidate label as a feature.

**Remaining caveat:** When *identical* document, target subject/object and
optional types legitimately bear multiple labels, a **single-label**
softmax cannot learn both labels from indistinguishable input. The adapter
continues to fail closed for these conflicts. A multi-label architecture or
original mention-span annotations would then be necessary; avoid silently
removing or relabeling these records.

The source files are not accessible from the user's server in this environment.
The patch is validated on representative synthetic rows only; the server
audit decides whether all tasks really become READY.
