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


from collections import Counter, defaultdict
import csv


def _audit_overlap(train_rows: list[dict], dev_rows: list[dict]) -> tuple[dict, dict, list[dict], list[dict]]:
    """Fingerprint exact normalized input/alternatives, ignoring option order.

    Do not use row IDs for split deduplication: source-local IDs can be reused
    across independent splits, notably BioJev NLI loaders enumerate each split.
    """
    train_by_fp = defaultdict(list)
    dev_by_fp = defaultdict(list)
    for i, row in enumerate(train_rows, 1):
        train_by_fp[fingerprint(row)].append((i, row))
    for i, row in enumerate(dev_rows, 1):
        dev_by_fp[fingerprint(row)].append((i, row))

    shared = set(train_by_fp) & set(dev_by_fp)
    rows_to_remove = {i for fp in shared for i, _ in train_by_fp[fp]}
    train_counts = Counter()
    dev_counts = Counter()
    examples = []
    removed = []
    conflicts = 0
    for fp in sorted(shared):
        tr = train_by_fp[fp]
        dv = dev_by_fp[fp]
        source = str(tr[0][1]['source'])
        train_counts[source] += len(tr)
        dev_counts[source] += len(dv)
        train_labels = sorted({r['answer_key'] for _, r in tr})
        dev_labels = sorted({r['answer_key'] for _, r in dv})
        conflict = len(set(train_labels) | set(dev_labels)) > 1
        conflicts += int(conflict)
        examples.append({
            'source': source,
            'fingerprint': fp,
            'train_row_count': len(tr),
            'dev_row_count': len(dv),
            'train_ids': '|'.join(sorted({r['id'] for _, r in tr})[:8]),
            'dev_ids': '|'.join(sorted({r['id'] for _, r in dv})[:8]),
            'train_gold': '|'.join(train_labels),
            'dev_gold': '|'.join(dev_labels),
            'conflicting_gold_labels': conflict,
        })
        for lineno, row in tr:
            removed.append({
                'train_line': lineno,
                'source': source,
                'id': row['id'],
                'fingerprint': fp,
                'gold_key': row['answer_key'],
            })

    audit = {
        'overlapping_content_fingerprints': len(shared),
        'overlapping_train_rows': len(rows_to_remove),
        'overlapping_dev_rows': sum(len(dev_by_fp[k]) for k in shared),
        'conflicting_gold_fingerprints': conflicts,
        'overlap_by_source': {
            s: {'train_rows': train_counts[s], 'dev_rows': dev_counts[s],
                'fingerprints': sum(e['source'] == s for e in examples)}
            for s in sorted(train_counts)
        },
        'identity_definition': 'SHA256(source, state, question, options sorted by key), excluding ID and gold label',
        'action': 'Preserve the existing development split; remove contaminated training rows only when explicitly requested.',
    }
    return audit, {'train_indices': rows_to_remove}, examples, removed


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_audit_files(dest: Path, audit: dict, examples: list[dict], removed: list[dict]):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'overlap_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    _write_csv(dest / 'overlap_examples.csv', examples, [
        'source', 'fingerprint', 'train_row_count', 'dev_row_count', 'train_ids',
        'dev_ids', 'train_gold', 'dev_gold', 'conflicting_gold_labels'])
    _write_csv(dest / 'removed_train_rows.csv', removed, [
        'train_line', 'source', 'id', 'fingerprint', 'gold_key'])


def build(train: str | Path, dev: str | Path, dest: str | Path, *,
          max_train=0, max_dev=0, overlap_policy='fail', audit_only=False) -> dict:
    """Convert source splits with a reproducible contamination audit.

    Default remains fail-closed. `drop-train` deliberately removes only training
    records whose normalized input + candidates also appear on dev, preserving
    all evaluation examples, ordering and labels. This is NOT equivalent to the
    original BioJev training protocol; do not claim a fair head-to-head with the
    original BioJev dev numbers without re-evaluating uncontaminated baselines.
    """
    if overlap_policy not in ('fail', 'drop-train'):
        raise ValueError("overlap_policy must be 'fail' or 'drop-train'")
    tr = [normalise(x) for x in records(train)]
    dv = [normalise(x) for x in records(dev)]
    if not tr or not dv:
        raise ValueError('Training and development inputs must both be non-empty')
    audit, index, examples, removed = _audit_overlap(tr, dv)
    dest = Path(dest)
    audit.update({
        'origin_train_rows': len(tr), 'origin_dev_rows': len(dv),
        'input_sha256': {'train': sha256(train), 'dev': sha256(dev)},
        'requested_overlap_policy': overlap_policy,
        'audit_only': bool(audit_only),
    })
    _write_audit_files(dest, audit, examples, removed)
    overlap = audit['overlapping_content_fingerprints']
    if audit_only:
        return audit
    if overlap and overlap_policy == 'fail':
        raise ValueError(
            f'TRAIN/DEV LEAKAGE: {overlap} overlapping content fingerprints, '
            f'{audit["overlapping_train_rows"]} affected training rows and '
            f'{audit["overlapping_dev_rows"]} affected development rows. '
            f'Inspect {dest / "overlap_audit.json"} and '
            'run with --overlap-policy drop-train to preserve dev and remove '
            'matching training rows (different protocol from original BioJev).'
        )
    removed_indices = index['train_indices'] if overlap_policy == 'drop-train' else set()
    clean_tr = [r for i, r in enumerate(tr, 1) if i not in removed_indices]
    if not clean_tr:
        raise ValueError('Overlap removal would leave an empty training set. No datasets written.')
    if max_train: clean_tr = clean_tr[:int(max_train)]
    if max_dev: dv = dv[:int(max_dev)]
    if not clean_tr or not dv:
        raise ValueError('max-train/max-dev selected an empty dataset')
    # Strict postcondition: no cross-split repeated content even after optional limits.
    if {fingerprint(r) for r in clean_tr} & {fingerprint(r) for r in dv}:
        raise RuntimeError('Unexpected overlap after sanitization')
    dump(dest / 'train.jsonl', clean_tr)
    dump(dest / 'dev.jsonl', dv)
    dump(dest / 'laya_train.jsonl', [to_laya(x) for x in clean_tr])
    dump(dest / 'laya_dev.jsonl', [to_laya(x) for x in dv])
    src_train = dict(sorted(Counter(r['source'] for r in clean_tr).items()))
    src_dev = dict(sorted(Counter(r['source'] for r in dv).items()))
    manifest = {
        'origin': 'BioJev Sprint8 canonical train/dev files (user provided)',
        'protocol': 'development examples preserved; training examples with identical dev content removed on explicit request',
        'overlap_policy': overlap_policy,
        'train_rows': len(clean_tr), 'dev_rows': len(dv),
        'original_train_rows': len(tr), 'original_dev_rows': audit['origin_dev_rows'],
        'removed_train_rows': len(removed_indices),
        'overlapping_content_fingerprints': overlap,
        'overlap_by_source': audit['overlap_by_source'],
        'train_sources': src_train, 'dev_sources': src_dev,
        'input_sha256': audit['input_sha256'],
        'output_sha256': {
            f: sha256(dest / f) for f in
            ('train.jsonl', 'dev.jsonl', 'laya_train.jsonl', 'laya_dev.jsonl')},
        'fingerprint_overlap': 0,
        'audit_files': ['overlap_audit.json', 'overlap_examples.csv', 'removed_train_rows.csv'],
        'notice': (
            'BioJev original Sprint8 files have content-level train/dev overlap. '
            'New BioLaya splits preserve development questions but reduce training data; '
            'not identical training protocol. Original BioJev SystemOne development '
            'accuracy may be inflated by overlap. Independent tests / rerun clean '
            'baselines are required for defensible relative performance.'),
    }
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2,
        ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest
