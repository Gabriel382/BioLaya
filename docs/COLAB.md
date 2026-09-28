# Google Colab + Drive

BioLaya is Colab-first. The Colab VM is treated as disposable compute; Google Drive is persistent recovery storage.

## Storage policy

Train/read caches under `/content`, not directly inside Drive. Drive is used for checkpoints, logs, results, final models and manifests.

Default Drive root:

`/content/drive/MyDrive/BioLaya`

Layout:

- `checkpoints/`
- `logs/`
- `results/`
- `final_models/`
- `manifests/`

Future training scripts use `biolaya.cloud.drive.backup_checkpoint()` and `restore_latest_checkpoint()` so a Colab session can resume after disconnects.

## New Colab session

1. Select a GPU runtime.
2. Mount Drive.
3. Clone the BioLaya GitHub repository into `/content/BioLaya`.
4. `pip install -e ".[train]"`.
5. Run `python scripts/check_env.py` and confirm `cuda_available: true`.
6. Initialize `/content/drive/MyDrive/BioLaya`.
7. Restore the latest checkpoint if the current sprint has one.

Do not use Drive as the Hugging Face cache or as the live training directory; repeated small-file I/O is slower and less reliable than `/content`.
