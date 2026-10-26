"""NLI/RE standard BioLaya ModernBERT sequence-classification baseline.

Important: these are *not* Laya's native typed decision head. They produce
standalone classifier weights and should be benchmarked against the original
BioJev classifier with the same dataset labels and splits.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from .data import normalise,records,dump

NLI_LABELS=['contradiction','entailment','neutral']

def encode_standard(row: dict) -> dict:
    r=normalise(row)
    # Label definition for variable-size, per-dataset classifier heads.
    # Preserve criterion and question, instead of discarded hypothesis text.
    text=json.dumps(r['state'],ensure_ascii=False) if not isinstance(r['state'],str) else r['state']
    return {'id':r['id'],'source':r['source'],'text':text,'question':r['question'],
            'label':r['answer_key'],'options':[o['key'] for o in r['options']]}

def prepare(source_train,source_dev,out,source=None):
    tr=[encode_standard(x) for x in records(source_train)]
    dv=[encode_standard(x) for x in records(source_dev)]
    if source:
        tr=[x for x in tr if x['source']==source]
        dv=[x for x in dv if x['source']==source]
    if not tr or not dv: raise ValueError('Empty source train/dev. Check source identifier and paths.')
    labels=sorted({x['label'] for x in tr})
    if set(x['label'] for x in dv)-set(labels):raise ValueError('Dev has labels absent from train')
    if len(labels)<2:raise ValueError('At least two labels required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dump(out/'train.jsonl',tr);dump(out/'dev.jsonl',dv)
    manifest={'train':len(tr),'dev':len(dv),'labels':labels,'source':source,
              'warning':'Use same held-out test for actual baseline comparisons; dev used for model selection.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
