#!/usr/bin/env python3
"""Prepare all five original biomedical benchmarks from local labelled split files.

DO NOT reconstruct train/test splits or substitute the 11-source Sprint8 dev
subset for the 5-task Sprint4 held-out protocol.
"""
import argparse,json
from pathlib import Path
from biolaya.comparison.standard import prepare
from biolaya.comparison.data import records
import yaml

DATASETS=('bionli','nli4ct','chemprot','ddi2013','biored')

def locate(root,name,split):
    d=Path(root)/name
    for suffix in ('.jsonl','.json'):
        p=d/(split+suffix)
        if p.exists():return p
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,help='Original BioJev data/processed or BioLaya labelled splits')
    p.add_argument('--out',default='data/standard/five')
    p.add_argument('--configs-out',default='configs/standard/five')
    p.add_argument('--require-test',action='store_true')
    a=p.parse_args()
    summary={}
    for name in DATASETS:
        tr=locate(a.root,name,'train')
        dev=locate(a.root,name,'dev') or locate(a.root,name,'validation')
        test=locate(a.root,name,'test')
        if not tr or not dev or (a.require_test and not test):
            summary[name]={'status':'MISSING_SPLIT','train':str(tr),'dev':str(dev),'test':str(test)}
            continue
        dest=Path(a.out)/name
        try:
            result=prepare(tr,dev,dest,source=None)
            # read source is BioLaya 'dataset': all source rows should be one task.
            sources={r['source'] for r in records(dest/'train.jsonl')}
            if sources!={name}: raise ValueError(f'{name}: inconsistent source values {sources}')
            if test:
                from biolaya.comparison.standard import encode_standard
                from biolaya.comparison.data import dump
                xs=[encode_standard(x) for x in records(test)]
                if any(x['label'] not in result['labels'] for x in xs):
                    raise ValueError(f'{name}: unseen test labels; do not discard classes')
                dump(dest/'test.jsonl',xs)
            cfg={'seed':42,'model':{'base':'answerdotai/ModernBERT-large',
                 'output':f'outputs/biolaya/standard/five/{name}'},'labels':result['labels'],
                 'data':{'train':str(dest/'train.jsonl'),'dev':str(dest/'dev.jsonl')},
                 'training':{'epochs':3,'max_length':1024 if name=='nli4ct' else 512,
                 'batch_size':2 if name=='nli4ct' else 4,'eval_batch_size':8,
                 'grad_accum':16,'learning_rate':2e-5,'bf16':True,'gradient_checkpointing':True}}
            out=Path(a.configs_out)/(name+'.yaml');out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(yaml.safe_dump(cfg,sort_keys=False))
            summary[name]={'status':'READY','train':result['train'],'dev':result['dev'],
                'test':len(list(records(dest/'test.jsonl'))) if test else None,
                'labels':result['labels'],'config':str(out)}
        except Exception as e:
            summary[name]={'status':'BLOCKED','reason':str(e),
                 'note':'Preserve exact BioJev split and ontology; no automatic re-labelling'}
    print(json.dumps(summary,indent=2))
    report=Path(a.out)/'audit.json';report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(summary,indent=2))
    if any(r['status']!='READY' for r in summary.values()):
        raise SystemExit('Not all five tasks ready; see audit.json for missing splits or incompatible formatting.')
if __name__=='__main__':main()
