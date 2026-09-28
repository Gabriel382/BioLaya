# Google Colab + Google Drive

BioLaya is Colab-first. The Colab VM is disposable compute; Google Drive is persistent recovery storage.

## Canonical notebook per sprint

Every BioLaya sprint ZIP/release includes a self-contained notebook:

`notebooks/NN_sprintN_colab.ipynb`

The same notebook is committed to GitHub. Download/copy it to Google Drive, open it with Google Colab, and run it there.

It installs BioLaya directly from GitHub:

```python
%pip install -q --upgrade "biolaya[train] @ git+https://github.com/Gabriel382/BioLaya.git"
```

No clone is required for normal Colab use.

## Device policy

All model/training entry points use `--device auto|cpu|cuda`:

- `auto`: CUDA if PyTorch can see it, otherwise CPU.
- `cpu`: force CPU for local correctness tests.
- `cuda`: require CUDA and fail early if no GPU is available.

For actual Colab training use `DEVICE="cuda"` after selecting a GPU runtime.

## Storage policy

Use `/content` for live datasets, caches and training. Use Drive for checkpoints, logs, results, final models and manifests.

Default Drive root: `/content/drive/MyDrive/BioLaya`

Do not use Drive as the Hugging Face cache or as the live training directory.
