#!/usr/bin/env python3
"""Evaluate BioLaya Standard on five original tasks. Baselines are not rerun."""
import argparse,subprocess
from pathlib import Path

DATASETS=('bionli','nli4ct','chemprot','ddi2013','biored')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',choices=['dev','test'],default='test')
    p.add_argument('--root',default='data/standard/five')
    p.add_argument('--ckpts',default='outputs/biolaya/standard/five')
    p.add_argument('--results',default='results/standard/five')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    for task in DATASETS:
        path=Path(a.root)/task/(a.split+'.jsonl')
        ck=Path(a.ckpts)/task/'final'
        if not path.exists():raise SystemExit(f'Missing {path}; do not substitute dev for test')
        if not ck.exists():raise SystemExit(f'Missing trained checkpoint {ck}')
        cmd=['python','scripts/eval_standard.py','--checkpoint',str(ck),'--data',str(path),
             '--output',str(Path(a.results)/task),'--max-length','1024' if task=='nli4ct' else '512']
        if a.limit:cmd.extend(['--limit',str(a.limit)])
        print(' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True)
if __name__=='__main__':main()
