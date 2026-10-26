#!/usr/bin/env python3
"""Prepare general MNLI and biomedical BioNLI+NLI4CT from canonical BioJev split.

No test data is used in either training stage.
"""
from pathlib import Path
import argparse,json
from biolaya.comparison.data import records,dump
from biolaya.comparison.standard import encode_standard

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',default='data/comparable/sprint8/train.jsonl')
 p.add_argument('--dev',default='data/comparable/sprint8/dev.jsonl')
 p.add_argument('--out',default='data/standard');a=p.parse_args()
 for name,srcs in [('mnli',{'mnli'}),('biomedical_multinli',{'bionli','nli4ct'})]:
    out=Path(a.out)/name
    tr=[encode_standard(r) for r in records(a.train) if r['source'] in srcs]
    dv=[encode_standard(r) for r in records(a.dev) if r['source'] in srcs]
    if not tr or not dv:raise ValueError(f'{name} has no training/dev rows')
    dump(out/'train.jsonl',tr);dump(out/'dev.jsonl',dv)
    (out/'manifest.json').write_text(json.dumps({'sources':sorted(srcs),'train':len(tr),'dev':len(dv)},indent=2))
    print(name,len(tr),len(dv))
if __name__=='__main__':main()
