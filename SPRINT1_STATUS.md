# BioLaya Sprint 1 status

Delivered in v0.1.0:

- Installable Python package (`pip install -e .`).
- Upstream Laya pinned to `laya==0.3.20`.
- Google Colab setup notebook and executable `.py` companion.
- Google Drive persistent layout and checkpoint backup/restore primitives.
- Environment/GPU report CLI.
- BioNLI, NLI4CT, ChemProt, DDI2013 and BioRED preparation adapters.
- Laya-native typed-decision schema.
- Base-Laya evaluation wrapper and metrics (accuracy, macro/weighted F1, ECE when confidence is available).
- Sprint 1 Colab walkthrough and Drive result copy.
- Seven-sprint roadmap shared conceptually with BioJev.

Not yet included by design:

- Sprint 2 biomedical encoder/domain adaptation training.
- Sprint 3 Laya-native RLCD biomedical fine-tuning.
- Training-time Drive callback / `resume=auto` integration (the storage primitives are already present and will be wired into the first training sprint).
- Hugging Face model push, because there is no BioLaya trained checkpoint yet.
