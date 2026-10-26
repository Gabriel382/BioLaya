#!/usr/bin/env python3
"""Numerical parity between native Laya PyTorch inference and Ollama SystemOne.

Fail if top-1 predictions disagree; report distribution deviations by sample.
Never declare native export valid from a GGUF magic header alone.
"""
import argparse,json,math
from biolaya.comparison.data import records,to_laya,to_ollama
from biolaya.comparison.evaluate import post

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='outputs/biolaya/systemone/final')
 p.add_argument('--ollama-model',default='biolaya-systemone:421m')
 p.add_argument('--data',default='data/comparable/sprint8/dev.jsonl')
 p.add_argument('--samples',type=int,default=30)
 p.add_argument('--endpoint',default='http://127.0.0.1:11434')
 p.add_argument('--prob-tol',type=float,default=0.05)
 a=p.parse_args()
 import laya
 agent=laya.load(a.checkpoint)
 failures=[];seen=0
 for row in list(records(a.data))[:a.samples]:
    left=agent.predict(to_laya(row)['state'],to_laya(row)['questions'])['answers']['decision']
    right=post(to_ollama(row,a.ollama_model),a.endpoint)['answers']['decision']
    seen+=1
    if left.get('choice')!=right.get('choice'):
        failures.append({'id':row['id'],'type':'prediction','torch':left.get('choice'),'ollama':right.get('choice')})
    lp=left.get('probabilities',{});rp=right.get('probabilities',{})
    if isinstance(lp,dict) and isinstance(rp,dict):
        for key in sorted(set(lp)&set(rp)):
            if abs(float(lp[key])-float(rp[key]))>a.prob_tol:
                failures.append({'id':row['id'],'type':'probability','label':key,
                   'torch':lp[key],'ollama':rp[key]})
 print(json.dumps({'samples':seen,'failures':len(failures),'details':failures[:30]},indent=2))
 if failures:raise SystemExit('Native Ollama export has not passed parity checks')
 print('BIOLAYA OLLAMA PARITY: PASS')
if __name__=='__main__':main()
