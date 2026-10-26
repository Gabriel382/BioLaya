# BioLaya 0.3.3 — Laya CLI compatibility fix

## Why

The previous BioLaya repository pinned `laya==0.3.20`. On the target Linux Python 3.12 environment, that version installed correctly but **did not create the `laya-train` command**. Reinstalling with `pip install --force-reinstall --no-deps` could not fix an absent console script.

BioLaya 0.3.3 pins `laya==0.4.15`, in which Laya documents `laya-train`, and improves missing-CLI diagnostics. The upstream reference is https://nandhakishorm.github.io/laya/finetune/.

## Install patch in the current repository

```bash
cd ~/git/postdoc/BioLaya
unzip -o BioLaya_LayaCLI_Hotfix_v0.3.3.zip -d .
python -m pip install -e ".[train,systemone,dev]"
python -m pip show laya
command -v laya-train
laya-train --help
python scripts/train_systemone.py --dry-run
```

If `laya-train` remains unavailable, output `python -m pip show laya; python -c 'import importlib.metadata as m; print([(e.name,e.value) for e in m.distribution("laya").entry_points])'` and investigate the package index/wheel actually installed. The existence of the command cannot be guaranteed from a local static test.

No prepared data, baselines, checkpoints or result files are changed. Previous `laya_train.jsonl` with 145,998 training / 6,768 dev is retained.

**Important:** after changing an upstream runtime dependency, do not assume numerical parity with previous baselines. Keep per-run version manifests.
