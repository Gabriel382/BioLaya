#!/usr/bin/env python3
"""Prepare five BioJev biomedical benchmark tasks for BioLaya Standard.

Never fabricate an evaluation split. Preserve existing official test/dev data.
BioNLI single-split evaluation membership is reconstructed using the IDs
saved in BioJev Sprint4 predictions, not a fresh random 10% split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import yaml

DATASETS = ("bionli", "nli4ct", "chemprot", "ddi2013", "biored")
ID_SOURCE_PATTERN = re.compile(r"^(?:bionli)[-_:](?:train[-_:])?(\d+)$", re.I)


def read_rows(path):
    if not path or not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, list): raise ValueError(f"Expected list in {path}")
        yield from data
        return
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict): raise ValueError('Expected object')
                    yield row
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"{path}:{i}: {exc}") from exc


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def filepath(root, ds, name):
    for suffix in ('.jsonl', '.json'):
        p = Path(root) / ds / (name + suffix)
        if p.is_file(): return p
    return None


def first(row, *keys):
    for key in keys:
        x = row.get(key)
        if x is not None and x != '' and x != []:
            return x
    return None


def as_text(value):
    if isinstance(value, str): return value
    if isinstance(value, (dict, list)): return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _pair_annotations(row):
    """Represent BioJev RE target pair from subject/object or BioLaya metadata.

    BioJev processed ChemProt/DDI2013/BioRED has root fields `subject` and
    `object` with optional entity types, and metadata.document_id. The older
    BioLaya choice rows store `entity1`, `entity2`, `arg1_id`, `arg2_id` in
    metadata instead. Do not use a unique relation-row ID or the gold label
    as a model feature: that would make evaluation meaningless.
    """
    md=row.get('metadata') if isinstance(row.get('metadata'),dict) else {}

    def pick(*names):
        for src in (md,row):
            for name in names:
                x=src.get(name)
                if x is not None and x!='' and x!=[] and x!={}:
                    return x
        return None

    def entity_text(x):
        if x is None: return None
        if isinstance(x,dict):
            # Prefer human-readable mention, preserve supplied offsets/IDs in
            # the pair identifier if present (never derive from gold label).
            for k in ('text','name','mention','surface','value','label'):
                if x.get(k) not in (None,'',[],{}):
                    return as_text(x[k])
            return as_text(x)
        return as_text(x)

    a1=pick('arg1_id','head_id','subject_id','entity1_id')
    a2=pick('arg2_id','tail_id','object_id','entity2_id')
    e1=pick('entity1','subject','head_entity','head_text','e1','head')
    e2=pick('entity2','object','tail_entity','tail_text','e2','tail')
    t1=pick('subject_type','entity1_type','head_type')
    t2=pick('object_type','entity2_type','tail_type')
    e1=entity_text(e1)
    e2=entity_text(e2)
    if e1 is not None and not e1.strip():e1=None
    if e2 is not None and not e2.strip():e2=None
    if not (e1 and e2) and not (a1 and a2):
        return None,None
    pieces=[]
    if e1 and e2:
        # These two features are part of the actual task input in BioJev,
        # even when there are no mention IDs in the processed JSONL.
        pieces.append(f'Target subject: {e1}. Target object: {e2}.')
    if t1 or t2:
        pieces.append(f'Subject type: {entity_text(t1) or "unspecified"}; '
                      f'object type: {entity_text(t2) or "unspecified"}.')
    if a1 and a2:
        pieces.append(f'Target mention IDs: {as_text(a1)} -> {as_text(a2)}.')
    signature={'entity1':e1,'entity2':e2,'arg1_id':as_text(a1) if a1 is not None else None,
               'arg2_id':as_text(a2) if a2 is not None else None,
               'subject_type':entity_text(t1),'object_type':entity_text(t2)}
    return '\n'.join(pieces),signature


def normalize_record(row, source, index):
    """Accept BioLaya choice rows and common BioJev NLI/RE labelled rows.

    Fail closed on numeric-only labels or records with unrecognised schema.
    Keeping labels verbatim prevents accidental relabeling of CPR/DDI/RE classes.
    """
    if not isinstance(row, dict): raise ValueError('expected JSON object')
    rid = first(row, 'id', 'example_id', 'uid', 'pair_id')
    rid = str(rid) if rid is not None else ''

    if {'state', 'instructions', 'criteria', 'gold'} <= set(row):
        if row.get('type', 'choice') != 'choice': raise ValueError('record must be a choice task')
        criteria = row['criteria']
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError('choice criteria must contain at least 2 keyed alternatives')
        label = str(row['gold'])
        if label not in {str(x) for x in criteria}:
            raise ValueError(f'gold {label!r} not in criteria')
        text = as_text(row['state'])
        question = str(row['instructions'])
    elif {'state','question','options','answer_key'} <= set(row):
        opts = row['options']
        if not isinstance(opts,list): raise ValueError('expected options list')
        keys={str(o['key']) for o in opts}
        label=str(row['answer_key'])
        if label not in keys: raise ValueError(f'answer_key {label!r} missing in options')
        text=as_text(row['state'])
        question=str(row['question'])
    else:
        label = (first(row,'label_name','relation_type','relation','gold_label','label','gold','answer')
                 if source in ('chemprot','ddi2013','biored') else
                 first(row,'gold_label','label_name','label','gold','answer'))
        if isinstance(label, dict): label = first(label,'type','label','name')
        if label is None: raise ValueError(f'no label field; available keys={sorted(row)}')
        if isinstance(label, bool) or isinstance(label, (int,float)) or str(label).strip().isdigit():
            raise ValueError(f'numeric class {label!r} needs an explicit verified label-name mapping')
        label=str(label)
        premise=first(row,'premise','sentence1','text_a','context','state')
        hypothesis=first(row,'hypothesis','sentence2','text_b','statement','Statement')
        if premise is not None and hypothesis is not None:
            text=as_text(premise)
            question=f'Hypothesis: {as_text(hypothesis)}'
        else:
            text=first(row,'text','input','sentence','marked_text','context','state','premise')
            if text is None:
                raise ValueError(f'no raw input text found; available keys={sorted(row)}')
            text=as_text(text)
            question=as_text(first(row,'question','instructions','prompt') or '')
    pair_description, pair_signature = (None, None)
    if source in ('chemprot', 'ddi2013', 'biored'):
        pair_description, pair_signature = _pair_annotations(row)
        # Do not accidentally add the same target metadata twice when a row
        # already includes entity *names* in the natural-language question.
        # Always include explicit pair IDs when available; those distinguish
        # identical surface forms referring to distinct annotated entities.
        if pair_description:
            # If a legacy BioLaya choice row already names both entities,
            # don't repeat their names, but retain annotation IDs if present.
            old_choice = ('instructions' in row or 'question' in row) and bool(str(question).strip())
            if old_choice:
                a1=pair_signature.get('arg1_id')
                a2=pair_signature.get('arg2_id')
                if a1 and a2:
                    question=str(question).rstrip()+f'\nTarget pair IDs: entity1={a1}, entity2={a2}.'
            else:
                question=(str(question).rstrip()+'\n'+pair_description).strip()
    if not str(text).strip() or not str(label).strip(): raise ValueError('empty text or label')
    if not rid:
        rid=f'__missing_id__:{source}:{index}'
    result = {'id':rid, 'source':source, 'text':str(text), 'question':str(question), 'label':str(label)}
    if source in ('chemprot','ddi2013','biored'):
        result['pair_annotations_present']=bool(pair_signature)
        result['pair_ids_present']=bool(pair_signature and pair_signature.get('arg1_id') and pair_signature.get('arg2_id'))
        if pair_signature:
            result['pair_annotation']=pair_signature
    return result


def canonical_id(value):
    m = ID_SOURCE_PATTERN.fullmatch(str(value))
    return f'bionli:{int(m.group(1))}' if m else str(value)


def test_ids_from_biojev(biojev_root):
    path=(Path(biojev_root)/'results/sprint4/frozen_4b/'
          'biojev-4b__bionli__test__seed42/predictions.jsonl')
    if not path.is_file():
        raise FileNotFoundError(f'BioNLI frozen test membership not found: {path}')
    rows=list(read_rows(path))
    if not rows:raise ValueError(f'Empty BioJev test predictions {path}')
    ids={canonical_id(first(r,'id','example_id')) for r in rows}
    if None in ids or 'None' in ids:raise ValueError('BioJev test predictions contain missing IDs')
    if len(ids)!=len(rows): raise ValueError(f'Duplicate BioJev test IDs: {len(rows)} records vs {len(ids)} unique')
    return ids, path, rows


def stratified_holdout(rows, *, seed=42, fraction=.1, group_by_document=False):
    """Content-group stratification: identical input pairs never straddle splits.

    Contradictory labels on identical inputs are retained in the same group
    instead of silently dropping data or splitting conflicting examples.
    """
    by_fingerprint=defaultdict(list)
    for row in rows:
        key = document_fingerprint(row) if group_by_document else fingerprint(row)
        by_fingerprint[key].append(row)
    grouped=defaultdict(list)
    for key in sorted(by_fingerprint):
        items=by_fingerprint[key]
        counts=Counter(r['label'] for r in items)
        primary=sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))[0][0]
        grouped[primary].append(items)
    rng=random.Random(seed)
    train,dev=[],[]
    for label in sorted(grouped):
        groups=grouped[label][:]
        rng.shuffle(groups)
        # group count rather than row count avoids boundary-crossing copies.
        n=max(1, int(round(len(groups)*fraction))) if len(groups)>1 else 0
        n=min(n,len(groups)-1)
        for items in groups[:n]:dev.extend(items)
        for items in groups[n:]:train.extend(items)
    # A rare relation class may occur only in a document whose primary label
    # is common. The stratifier can assign that document to internal dev;
    # restore training-label coverage by moving whole document groups back to
    # train, never by leaking examples from official evaluation.
    labels_needed={r['label'] for r in rows}
    labels_in_train={r['label'] for r in train}
    if labels_needed-labels_in_train and dev:
        by_group=defaultdict(list)
        group_id=(document_fingerprint if group_by_document else fingerprint)
        for r in dev:by_group[group_id(r)].append(r)
        to_move=set()
        for missing in sorted(labels_needed-labels_in_train):
            options=[(len(items),k) for k,items in by_group.items()
                     if k not in to_move and any(r['label']==missing for r in items)]
            if not options:continue
            chosen=min(options)[1]
            to_move.add(chosen)
            labels_in_train.update(r['label'] for r in by_group[chosen])
        if to_move:
            train.extend(r for k in sorted(to_move) for r in by_group[k])
            dev=[r for r in dev if group_id(r) not in to_move]
    rng.shuffle(train);rng.shuffle(dev)
    return train,dev


def fingerprint(row):
    data={'text':row['text'],'question':row['question'],'source':row['source']}
    return hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def document_fingerprint(row):
    """Hash source + complete document context, without pair-specific question."""
    return hashlib.sha256(json.dumps({'text':row['text'], 'source':row['source']},
        ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def remove_cross_split_duplicates(train, dev, evaluation, *, by_document=False):
    # Preserve dev and evaluation, remove exact text duplicates from training.
    fp=document_fingerprint if by_document else fingerprint
    held={fp(x) for x in dev+evaluation}
    filtered=[x for x in train if fp(x) not in held]
    overlap_eval=len({fp(x) for x in dev}&{fp(x) for x in evaluation})
    if overlap_eval:
        # Do not silently score on material seen by model-selection validation.
        raise ValueError(f'{overlap_eval} dev/evaluation overlapping fingerprints; revise the split policy')
    return filtered, len(train)-len(filtered)


def collect(root, name, split):
    path=filepath(root,name,split)
    if path is None: return None, None
    records=[]
    for i,row in enumerate(read_rows(path)):
        try:records.append(normalize_record(row,name,i))
        except Exception as exc:raise ValueError(f'{path}:{i+1}: {exc}') from exc
    if not records:raise ValueError(f'empty {name} {split} file: {path}')
    return records, path


def create_task(name, root, biojev_root, out_dir, cfg_dir, base, seed, fraction):
    train, trainpath = collect(root,name,'train')
    if train is None:raise FileNotFoundError(f'missing {name} train')
    dev, devpath = collect(root,name,'dev')
    if dev is None: dev,devpath=collect(root,name,'validation')
    test, testpath=collect(root,name,'test')
    orig_count=len(train)
    sources={'train':str(trainpath),'dev':str(devpath) if devpath else None,
             'test':str(testpath) if testpath else None}
    derived_dev=False
    removed_pre_dev=0
    eval_original_split='test'
    if name=='bionli':
        # A provided test file cannot be assumed to be the BioJev 3,599 subset.
        biojev_ids, predpath, prediction_rows = test_ids_from_biojev(biojev_root)
        from_train=[r for r in train if canonical_id(r['id']) in biojev_ids]
        found={canonical_id(r['id']) for r in from_train}
        if found!=biojev_ids:
            raise ValueError(f'BioNLI ID mismatch: matched {len(found)}/{len(biojev_ids)} '
              f'BioJev test IDs. Missing examples: {sorted(biojev_ids-found)[:8]}; '
              'do not invent new test samples. Check source ID schema/order.')
        test=from_train
        train=[r for r in train if canonical_id(r['id']) not in biojev_ids]
        # Non-identical IDs can still contain duplicate premise/hypothesis
        # pairs. Remove all evaluation-content copies *before* splitting an
        # internal dev: never allow a test example into model selection.
        eval_fingerprints={fingerprint(row) for row in test}
        before=len(train)
        train=[row for row in train if fingerprint(row) not in eval_fingerprints]
        removed_pre_dev=before-len(train)
        sources['test']=str(predpath)+' [ID membership only]'
        dev=None # derived from remaining train, not from the held-out test
    elif name=='nli4ct':
        # Per the original Sprint4 frozen protocol, evaluation is NLI4CT/dev, NOT test.
        if dev is None:raise FileNotFoundError('NLI4CT official dev/validation split is missing')
        test=dev
        sources['evaluation_source']=str(devpath)
        eval_original_split='dev'
        dev=None
    elif test is None:
        raise FileNotFoundError(f'{name}: test missing; cannot reproduce frozen benchmark')

    if dev is None:
        train,dev=stratified_holdout(train,seed=seed,fraction=fraction,
                                    group_by_document=name in ('chemprot','ddi2013','biored'))
        derived_dev=True
    train,removed=remove_cross_split_duplicates(train,dev,test,
                         by_document=name in ('chemprot','ddi2013','biored'))
    if not train or not dev or not test:
        raise ValueError(f'{name}: empty train/dev/evaluation after split handling')
    if name in ('chemprot','ddi2013','biored'):
        # An entity-pair task cannot be learned if all pairs in a document
        # collapse to the identical unmarked text while having different labels.
        for split_name, values in [('train',train),('dev',dev),('evaluation',test)]:
            known={}
            conflicts=[]
            for row in values:
                key=(row['text'],row['question'])
                if key in known and known[key]['label'] != row['label']:
                    conflicts.append((known[key],row))
                else:
                    known[key]=row
            if conflicts:
                sample=[{'ids':[p['id'] for p in pair],
                         'labels':[p['label'] for p in pair],
                         'pair_metadata_present':[p.get('pair_annotations_present',False) for p in pair],
                         'pair_ids_present':[p.get('pair_ids_present',False) for p in pair]}
                        for pair in conflicts[:3]]
                raise ValueError(
                    f'{name}/{split_name}: {len(conflicts)} contradictory labels on indistinguishable '
                    f'model inputs; samples={json.dumps(sample, ensure_ascii=False)}. '
                    'If subject/object and source mention IDs cannot disambiguate, this could be a genuinely multi-label relation '
                    'and needs a multi-label training head; do not rename or drop gold classes.')
    labels=sorted({r['label'] for r in train})
    unseen={r['label'] for r in dev+test}-set(labels)
    if unseen:raise ValueError(f'{name}: evaluation has classes missing in training: {sorted(unseen)}')
    # Protect against spurious IDs in train/dev/eval, keeping ID namespaces intact.
    if name=='bionli' and len(test)!=len(biojev_ids):
        raise ValueError(f'{name}: unexpected held-out test size {len(test)}')
    taskdir=Path(out_dir)/name
    write_rows(taskdir/'train.jsonl',train)
    write_rows(taskdir/'dev.jsonl',dev)
    write_rows(taskdir/'evaluation.jsonl',test)
    if eval_original_split=='test':write_rows(taskdir/'test.jsonl',test)
    cfg={
      'seed':seed,
      'model':{'base':base,'output':f'outputs/biolaya/standard/five/{name}'},
      'labels':labels,
      'data':{'train':str(taskdir/'train.jsonl'),'dev':str(taskdir/'dev.jsonl')},
      'training':{'epochs':3,'max_length':1024 if name=='nli4ct' else 512,
                  'batch_size':2 if name=='nli4ct' else 4, 'eval_batch_size':8,
                  'grad_accum':16,'learning_rate':2e-5,'bf16':True,
                  'gradient_checkpointing':True}}
    configpath=Path(cfg_dir)/(name+'.yaml')
    configpath.parent.mkdir(parents=True, exist_ok=True)
    configpath.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    audit={'status':'READY','source_files':sources,
           'original_training_rows':orig_count,'train':len(train),'dev':len(dev),
           'evaluation':len(test),'evaluation_original_split':eval_original_split,
           'internal_dev_derived_from_training':derived_dev,
           'train_removed_by_exact_content_overlap':removed,
           'additional_training_removed_for_evaluation_overlap_before_holdout':removed_pre_dev,
           'split_overlap_granularity': 'document' if name in ('chemprot','ddi2013','biored') else 'input',
           'pair_annotations_train':sum(bool(r.get('pair_annotations_present')) for r in train),
           'pair_ids_train':sum(bool(r.get('pair_ids_present')) for r in train),
           'pair_source':'root subject/object or explicit entity metadata' if name in ('chemprot','ddi2013','biored') else None,
           'labels':labels,'config':str(configpath),
           'note':'Evaluation labels and instances preserved. Compare with the exact same BioJev dataset IDs and length policy.'}
    (taskdir/'manifest.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--biojev-root',default='../BioJev')
    p.add_argument('--out',default='data/standard/five')
    p.add_argument('--configs-out',default='configs/standard/five')
    p.add_argument('--base',default='answerdotai/ModernBERT-large')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--internal-dev-fraction',type=float,default=.1)
    p.add_argument('--require-test',action='store_true',help='Legacy option: enforce real evaluation split; NLI4CT uses official dev')
    a=p.parse_args()
    if not (0<a.internal_dev_fraction<.5):p.error('--internal-dev-fraction must be between 0 and .5')
    audit={}
    for name in DATASETS:
        try:
            audit[name]=create_task(name,a.root,a.biojev_root,a.out,a.configs_out,
                                    a.base,a.seed,a.internal_dev_fraction)
        except Exception as exc:
            audit[name]={'status':'BLOCKED','reason':str(exc),
                         'note':'Task failed closed. No automatic replacement of missing evaluation data.'}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2,ensure_ascii=False))
    if any(x['status']!='READY' for x in audit.values()):
        raise SystemExit('One or more tasks blocked. Review audit.json; nothing was invented.')


if __name__=='__main__':main()
