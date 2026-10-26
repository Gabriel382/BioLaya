"""Dataset bridge. No automatic downloads and no invented train/test splits.

Primary input is the existing BioJev Sprint8 *canonical* JSONL, preserving
source, ID, option descriptions, correct answers and split assignments.
All conversions fail closed on inconsistent or unsupported records.
"""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
from typing import Iterator


def records(path: str | Path) -> Iterator[dict]:
    path=Path(path)
    if path.suffix.lower()=='.json':
        obj=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(obj,list):
            for row in obj: yield row
            return
        raise ValueError(f'{path}: expected JSON array for .json files')
    with path.open(encoding="utf-8") as f:
        for lineno,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f"{path}:{lineno}: {e}") from e


def dump(path: str | Path, rows: list[dict]) -> None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')


def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def normalise(row: dict) -> dict:
    """Accept BioJev Sprint8 canonical records, or original BioLaya DecisionExample.

    The validated target is the same for ordinary Laya and native Ollama.
    """
    if {'state','question','options','answer_key'} <= row.keys():
        opts=[]
        for o in row['options']:
            if isinstance(o,dict):
                k=str(o['key']); d=str(o.get('description') or k)
            else: raise ValueError('expected canonical options list of key/description dicts')
            opts.append({'key':k,'description':d})
        answer=str(row['answer_key'])
        instruction=str(row['question'])
        state=row['state']
    elif {'state','instructions','criteria','gold'} <= row.keys():
        if row.get('type','choice') != 'choice': raise ValueError('only choice-labelled records are trainable here')
        if not isinstance(row['criteria'],dict): raise ValueError('criteria must be a dict')
        opts=[{'key':str(k),'description':str(d)} for k,d in row['criteria'].items()]
        answer=str(row['gold']); instruction=str(row['instructions']); state=row['state']
    else:
        raise ValueError('row is neither canonical BioJev nor BioLaya decision format')
    if not isinstance(state,(str,dict,list)) or not instruction.strip(): raise ValueError('state/instructions invalid')
    keys=[o['key'] for o in opts]
    if not 2<=len(keys)<=26 or len(keys)!=len(set(keys)): raise ValueError('options require 2..26 unique keys')
    if answer not in keys: raise ValueError(f'gold key {answer!r} missing from {keys}')
    return {'id':str(row.get('id','')),'source':str(row.get('source',row.get('dataset','unknown'))),
            'state':state,'question':instruction,'options':opts,'answer_key':answer,
            'provenance':row.get('provenance',{})}


def to_laya(row: dict) -> dict:
    r=normalise(row)
    return {'state':r['state'],
            'questions': {'decision': {'type':'choice','instructions':r['question'],
                                       'criteria':{o['key']:o['description'] for o in r['options']}}},
            'expected':{'decision':r['answer_key']}}


def to_ollama(row: dict, model: str) -> dict:
    d=to_laya(row); return {'model':model,'state':d['state'],'questions':d['questions']}


def fingerprint(r: dict) -> str:
    r=normalise(r)
    obj={'source':r['source'],'state':r['state'],'question':r['question'],
         'options':sorted(r['options'],key=lambda o:o['key'])}
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def build(train: str | Path, dev: str | Path, dest: str | Path, *, max_train=0,max_dev=0) -> dict:
    # No repartitioning, permutation or source rebalancing: exact BioJev rows/splits.
    tr=[normalise(x) for x in records(train)]
    dv=[normalise(x) for x in records(dev)]
    train_keys={fingerprint(x) for x in tr}; dev_keys={fingerprint(x) for x in dv}
    overlap=train_keys.intersection(dev_keys)
    if overlap: raise ValueError(f'TRAIN/DEV LEAKAGE: {len(overlap)} overlapping content fingerprints')
    if max_train: tr=tr[:int(max_train)]
    if max_dev: dv=dv[:int(max_dev)]
    dest=Path(dest)
    dump(dest/'train.jsonl',tr); dump(dest/'dev.jsonl',dv)
    dump(dest/'laya_train.jsonl',[to_laya(x) for x in tr])
    dump(dest/'laya_dev.jsonl',[to_laya(x) for x in dv])
    src_train={s:sum(r['source']==s for r in tr) for s in sorted({r['source'] for r in tr})}
    src_dev={s:sum(r['source']==s for r in dv) for s in sorted({r['source'] for r in dv})}
    manifest={
      'origin':'BioJev Sprint8 canonical train/dev files (user provided)',
      'protocol':'same rows and choice alternatives; no baseline reruns',
      'train_rows':len(tr),'dev_rows':len(dv),'train_sources':src_train,'dev_sources':src_dev,
      'input_sha256':{'train':sha256(train),'dev':sha256(dev)},
      'output_sha256':{f:sha256(dest/f) for f in ('train.jsonl','dev.jsonl','laya_train.jsonl','laya_dev.jsonl')},
      'fingerprint_overlap':0,
      'notice':'Development results are not an independent test; option length and architecture may differ.',
    }
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
