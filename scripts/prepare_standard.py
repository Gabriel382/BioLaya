#!/usr/bin/env python3
import argparse,json
from biolaya.comparison.standard import prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',default='data/comparable/sprint8/train.jsonl')
    p.add_argument('--dev',default='data/comparable/sprint8/dev.jsonl')
    p.add_argument('--source',default='bionli')
    p.add_argument('--out',default='data/standard/bionli')
    a=p.parse_args();print(json.dumps(prepare(a.train,a.dev,a.out,a.source),indent=2))
if __name__=='__main__':main()
