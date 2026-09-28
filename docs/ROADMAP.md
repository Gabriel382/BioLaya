# BioLaya roadmap

BioLaya follows the same seven scientific sprints as BioJev while using Laya-native decision primitives.

1. **Benchmark foundation** — same biomedical datasets/splits/metrics as BioJev, represented as Laya typed decisions.
2. **Biomedical domain adaptation** — compare Laya without DAPT against a biomedical encoder-adapted branch (masked-language/domain continual pretraining appropriate to ModernBERT, not causal LM DAPT).
3. **Core BioLaya** — biomedical typed-decision fine-tuning using Laya's native decision head/RLCD recipe; fit calibration only on held-out calibration data.
4. **Generalization** — BioNLI, NLI4CT, ChemProt, DDI2013, BioRED, cross-dataset and multi-task transfer.
5. **Calibration & reliability** — ECE, Brier, NLL/risk-coverage/selective accuracy plus Laya-native temperature fitting.
6. **Ablations** — DAPT/no-DAPT, dataset mixtures, decision primitives, order/amount of specialization, calibration choices.
7. **Release/paper** — reproducible configs/results, GitHub release, Hugging Face model card/checkpoint, comparison with BioJev/OpenJev/Laya.
