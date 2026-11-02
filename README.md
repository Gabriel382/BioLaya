# BioLaya

**BioLaya** is a biomedical specialization of **Laya** focused on structured decision making for biomedical and clinical NLP.
It was developed to give **Laya** the same kind of rigorous experimental treatment given to **BioJev**, including:

- a **task-specific biomedical evaluation track** on 5 benchmarks;
- a **SystemOne track** for structured multi-choice decision tasks;
- **comparisons against strong baselines**;
- reproducible configs, scripts, and export flows for **Hugging Face** and **Ollama**.

**Project lead:** Gabriel Henrique Alencar Medeiros  
**Supervision:** Lina F. Soualmia

---

## What is in this repository?

BioLaya has **two complementary model families**:

### 1) BioLaya Standard
Task-specific supervised biomedical classifiers:

- **BioNLI**
- **NLI4CT**
- **ChemProt**
- **DDI2013**
- **BioRED**

These models were trained and evaluated as dedicated classifiers for each benchmark.

### 2) BioLaya SystemOne
A **Laya-style structured decision model** trained for typed decision tasks, including biomedical and general decision benchmarks.

This is the branch intended for:

- structured inference;
- interoperability with **Laya/SystemOne-style APIs**;
- export and validation for **Ollama** deployment.

---

## Main results

### Standard biomedical benchmarks

| Task | Accuracy | Macro-F1 | ECE |
| --- | ---: | ---: | ---: |
| BioNLI | **94.11%** | **93.55%** | 0.0394 |
| NLI4CT | 50.00% | 38.00% | 0.0242 |
| ChemProt | 39.22% | 17.95% | 0.1039 |
| DDI2013 | 56.28% | 45.08% | 0.1202 |
| BioRED | **76.50%** | **64.77%** | 0.1707 |

### BioLaya SystemOne

- **11-task evaluation**
- **6,768 evaluation examples**
- **89.44% aggregate accuracy**
- fast structured decision inference with native SystemOne formatting

---

## Comparison figures

The repository now includes ready-to-use charts under [`results/images/`](results/images).

### BioLaya vs supervised biomedical baselines

![Supervised accuracy comparison](results/images/01_supervised_accuracy.png)

![Supervised macro-F1 comparison](results/images/02_supervised_macro_f1.png)

### BioLaya vs frozen decision-model baselines

![BioLaya vs frozen models](results/images/03_biolaya_vs_frozen_accuracy.png)

### BioLaya SystemOne on 11 tasks

![BioLaya SystemOne](results/images/05_biolaya_systemone_11_tasks.png)

### Delta to best supervised baseline

![Delta to best baseline](results/images/06_delta_to_best_supervised_accuracy.png)

A compiled PDF version is also provided in [`results/reports/BioLaya_Comparison_Charts.pdf`](results/reports/BioLaya_Comparison_Charts.pdf).

---

## Repository structure

```text
BioLaya/
├── configs/                    # training and evaluation configs
├── docs/                       # notes, sprint docs, protocol docs
├── notebooks/                  # Colab / walkthrough notebooks
├── scripts/                    # training, preparation, export, evaluation
├── src/biolaya/                # source package
├── results/
│   ├── images/                 # publication / README-ready figures
│   ├── reports/                # compiled PDF reports
│   └── tables/                 # CSVs used for the charts
└── README.md
```

---

## Quick start

Install locally:

```bash
python -m pip install -e .
```

For training / experiments:

```bash
python -m pip install -e ".[train]"
```

For SystemOne / Laya-specific workflows:

```bash
python -m pip install -e ".[systemone]"
```

---

## Reproducing the core workflows

### 1) Prepare comparable SystemOne data

```bash
python scripts/prepare_comparable_data.py \
  --biojev-train ../BioJev/data/sprint8/systemone_4b/train.jsonl \
  --biojev-dev ../BioJev/data/sprint8/systemone_4b/dev.jsonl \
  --overlap-policy drop-train
```

### 2) Train BioLaya SystemOne

Dry run:

```bash
python scripts/train_systemone.py --dry-run
```

Actual training:

```bash
python scripts/train_systemone.py
```

### 3) Prepare the five biomedical benchmarks

```bash
python scripts/prepare_five_benchmarks.py \
  --root ../BioJev/data/processed \
  --require-test
```

### 4) Evaluate / inspect results

Use the generated summaries and the chart tables inside [`results/tables/`](results/tables).

---

## Model publishing

### Hugging Face
This repository is designed so final released checkpoints can be exported and published manually to Hugging Face.

Recommended repository name:

- **`Gabriel382/biolaya`**

### Ollama
BioLaya SystemOne was also prepared for **Ollama-style deployment**, after conversion / validation in the dedicated export pipeline.

The README for the Ollama release should document:

- model purpose;
- SystemOne request format;
- benchmark scope;
- limitations;
- expected `/v1/systemone` usage.

---

## Notes on comparisons

Please keep the following in mind when reporting results:

- **BioLaya Standard** is a **task-specific supervised setup**.
- **BioJev 4B / OpenJev / Qwen frozen** are **not** directly identical training regimes.
- Some comparisons are therefore **descriptive**, not strict apples-to-apples claims.
- NLI4CT and benchmark split policies should always be stated clearly.

The included figures are meant to be **professional, publication-ready summaries**, but claims in papers or posts should still mention the exact evaluation setting.

---

## External links

- **GitHub:** `https://github.com/Gabriel382/BioLaya`
- **Hugging Face:** `https://huggingface.co/Gabriel382/biolaya`
- **Reference project:** `https://github.com/Gabriel382/BioJev`

---

## Citation

If you use BioLaya, please cite the repository / report associated with the release.
A formal citation block can be added once the final technical report or paper entry is fixed.

---

## Disclaimer

**For research use only.** BioLaya is not a medical device and must not be used for autonomous clinical decision making.
