#!/usr/bin/env python3
"""Publish an existing, validated BioLaya checkpoint. No default upload."""
import argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--checkpoint',required=True)
 p.add_argument('--repo-id',required=True,help='e.g. Gabriel382/BioLaya-SystemOne')
 p.add_argument('--private',action='store_true')
 p.add_argument('--confirm',action='store_true',help='Actually upload checkpoint; omit to preview')
 a=p.parse_args()
 source=Path(a.checkpoint)
 if not source.exists():raise SystemExit(f'Missing checkpoint: {source}')
 if not (source/'README.md').is_file():
    raise SystemExit(f'Missing model card: {source}/README.md; generate/review it before publishing')
 files=[x.name for x in source.iterdir()]
 print('Source:',source.resolve(),'\nDestination:',a.repo_id,'\nFiles:',files)
 if not a.confirm:
    print('Dry-run only. Add --confirm to publish.');return
 from huggingface_hub import HfApi
 api=HfApi();api.create_repo(repo_id=a.repo_id,private=a.private,exist_ok=True,repo_type='model')
 result=api.upload_folder(folder_path=str(source),repo_id=a.repo_id,repo_type='model',
    ignore_patterns=['checkpoint_latest/**','*.pt','*.bin','*.log','**/trainer_state.json'])
 print('Published to',result)
if __name__=='__main__':main()
