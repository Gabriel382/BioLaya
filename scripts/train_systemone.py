#!/usr/bin/env python3
"""Laya-native training with upstream laya-train; NOT causal-letter SFT."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import yaml

def build_command(config, dry_run=False):
    t=config['training']
    exe=shutil.which('laya-train')
    if exe is None: raise RuntimeError('laya-train missing: pip install -e ".[systemone]" (laya==0.3.20)')
    cmd=[exe,'--data',str(config['data']['train']),'--base',str(config['model']['base']),
         '--out',str(config['model']['output']),'--loss',str(t.get('loss','soft-ce')),
         '--epochs',str(t.get('epochs',3)),'--micro-batch',str(t.get('micro_batch',4)),
         '--grad-accum',str(t.get('grad_accum',16)),'--encoder-lr',str(t.get('encoder_lr',2.5e-5)),
         '--head-lr',str(t.get('head_lr',1e-4)),'--max-len',str(t.get('max_len',1024)),
         '--head-max-len',str(t.get('head_max_len',256)),
         '--seed',str(config.get('seed',42)),'--device',str(t.get('device','cuda'))]
    if t.get('shuffle_options',True): cmd.append('--shuffle-options')
    if t.get('freeze_encoder',False): cmd.append('--freeze-encoder')
    if dry_run: cmd.append('--dry-run')
    if config.get('data',{}).get('dev'): cmd.extend(['--eval-data',str(config['data']['dev'])])
    return cmd

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/systemone/biolaya.yaml')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--print-command',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    for key in ('train','dev'):
        path=cfg.get('data',{}).get(key)
        if path and not Path(path).exists(): raise SystemExit(f'Missing dataset: {path}; run prepare_comparable_data.py')
    cmd=build_command(cfg,dry_run=a.dry_run)
    print(' '.join(cmd),flush=True)
    if not a.print_command:
        subprocess.run(cmd,check=True)
        if not a.dry_run:
            dest=Path(cfg['model']['output']);dest.mkdir(parents=True,exist_ok=True)
            (dest/'biolaya_run_config.json').write_text(json.dumps(cfg,indent=2))
if __name__=='__main__': main()
