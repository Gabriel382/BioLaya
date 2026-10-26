#!/usr/bin/env python3
"""Reuse *exact* BioJev Sprint8 train/dev examples, with provenance hashes."""
import argparse,json
from biolaya.comparison.data import build

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--biojev-train',default='../BioJev/data/sprint8/systemone_4b/train.jsonl')
    p.add_argument('--biojev-dev',default='../BioJev/data/sprint8/systemone_4b/dev.jsonl')
    p.add_argument('--out',default='data/comparable/sprint8')
    p.add_argument('--max-train',type=int,default=0,help='For smoke experiments only')
    p.add_argument('--max-dev',type=int,default=0,help='For smoke experiments only')
    p.add_argument('--overlap-policy', choices=('fail','drop-train'), default='fail',
                   help='Explicitly filter contaminated TRAIN rows while preserving the full DEV split')
    p.add_argument('--audit-only', action='store_true',
                   help='Write overlap_audit.json and overlap_examples.csv; do not write training data')
    a=p.parse_args()
    print(json.dumps(build(a.biojev_train,a.biojev_dev,a.out,max_train=a.max_train,max_dev=a.max_dev,
                            overlap_policy=a.overlap_policy,audit_only=a.audit_only),indent=2))
if __name__=='__main__':main()
