#!/usr/bin/env python3
"""Evaluate on frozen per-task evaluation files, not arbitrary split names."""
import argparse
import subprocess
import sys
from pathlib import Path

DATASETS=('bionli','nli4ct','chemprot','ddi2013','biored')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',choices=['evaluation','dev','test'],default='evaluation',
                   help='evaluation is official test, except NLI4CT which is official dev')
    p.add_argument('--root',default='data/standard/five')
    p.add_argument('--ckpts',default='outputs/biolaya/standard/five')
    p.add_argument('--results',default='results/standard/five')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    for name in DATASETS:
        path=Path(a.root)/name/(a.split+'.jsonl')
        ck=Path(a.ckpts)/name/'final'
        if not path.exists():raise SystemExit(f'Missing {path}; cannot substitute a different split')
        if not ck.exists():raise SystemExit(f'Missing trained checkpoint {ck}')
        cmd=[sys.executable,'scripts/eval_standard.py','--checkpoint',str(ck),
             '--data',str(path),'--output',str(Path(a.results)/name),
             '--max-length','1024' if name=='nli4ct' else '512']
        if a.limit:cmd.extend(['--limit',str(a.limit)])
        print(' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True)


if __name__=='__main__':main()
