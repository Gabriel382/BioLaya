#!/usr/bin/env python3
"""Join previously produced BioJev and new BioLaya metrics, without reruns.

Never asserts a fair head-to-head unless the evaluation manifest matches.
"""
import argparse,csv,json
from pathlib import Path

def load_csv(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--biolaya',required=True,help='BioLaya summary.csv')
    p.add_argument('--biojev',required=True,help='Existing BioJev summary.csv')
    p.add_argument('--output',default='results/comparison/existing_vs_biolaya.csv')
    p.add_argument('--confirmed-same-protocol',action='store_true',help='ONLY after split/task/metric checks')
    a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for origin,path in [('BioLaya',a.biolaya),('BioJev (pre-existing)',a.biojev)]:
        for r in load_csv(path):
            rows.append({'experiment':origin,'model':r.get('model',''),'source':r.get('source',r.get('dataset','')),
                         'n':r.get('n',''),'accuracy':r.get('accuracy',''),
                         'f1_macro':r.get('f1_macro',''),
                         'protocol_status':'USER_CONFIRMED' if a.confirmed_same_protocol else 'NOT_VERIFIED',
                         'warning':'NO COMPARABLE CLAIM until same IDs/splits/tasks/metrics verified'})
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Written {out}; protocol status: ',rows[0]['protocol_status'])
if __name__=='__main__':main()
