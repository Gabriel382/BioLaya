# Upstream Laya contract

BioLaya pins `laya==0.3.20`.

The project depends on, rather than vendors, upstream Laya. This keeps upstream runtime/training fixes separate from BioLaya's biomedical data, experiments and releases.

Important assumptions used by BioLaya:

- English base checkpoint: `convaiinnovations/laya`.
- Non-autoregressive typed decisions using `choice`, `score`, and `noul`.
- English architecture: ModernBERT-large plus Laya decision head, 421M parameters.
- Domain fine-tuning should preserve a held-out calibration set; calibration must not be fitted on the evaluation test split.
- Upstream fine-tuning supports encoder and decision-head activation checkpointing; future BioLaya training code will expose these explicitly.

Whenever the pinned Laya version changes, rerun Sprint-1 baselines and record the version in result manifests.
