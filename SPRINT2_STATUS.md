# BioLaya Sprint 2 status

Delivered in v0.2.0:

- Installable cumulative Sprint 1 + Sprint 2 package.
- ModernBERT-large masked-LM biomedical domain adaptation.
- Streaming PubMed + PMC corpus (80/20 default).
- Deterministic PMID-based train/validation partitioning.
- Conservative PMC license filtering and retraction filtering.
- Exact token-budget packed training streams.
- 100K, 1M, 10M and 100M presets.
- CPU, CUDA and auto device modes.
- Colab-specific dependency compatibility extra.
- Google Drive checkpoint backup on every Trainer save.
- `--resume auto` from the latest Drive checkpoint.
- Final model + manifest persistence to Drive.
- Canonical `02_sprint2_colab.ipynb` that installs from GitHub.
- Local CPU structural smoke tests before Colab.
- Optional real CPU/GPU training smoke.

Sprint-2 output is **BioModernBERT**, an intermediate biomedical backbone. Sprint 3 will build the actual BioLaya decision model and compare DAPT vs no-DAPT routes.
