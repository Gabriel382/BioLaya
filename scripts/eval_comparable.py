#!/usr/bin/env python3
import argparse,json
from biolaya.comparison.evaluate import evaluate_file

def main():
    p=argparse.ArgumentParser(description='Same canonical BioJev Sprint8 development examples; BioLaya-only runs')
    p.add_argument('--data',default='data/comparable/sprint8/dev.jsonl')
    p.add_argument('--backend',choices=['laya','ollama'],default='laya')
    p.add_argument('--models',nargs='+',default=None)
    p.add_argument('--endpoint',default='http://127.0.0.1:11434')
    p.add_argument('--output',default='results/comparison/biolaya_systemone')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    names=a.models or (['outputs/biolaya/systemone/final'] if a.backend=='laya' else ['biolaya-systemone:421m'])
    rows=evaluate_file(a.data,names,backend=a.backend,endpoint=a.endpoint,output=a.output,limit=a.limit,skip_existing=a.resume,timeout=a.timeout)
    print(json.dumps([r for r in rows if r['source']=='ALL'],indent=2))
if __name__=='__main__':main()
