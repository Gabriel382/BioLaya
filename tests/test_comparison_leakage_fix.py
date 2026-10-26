"""Regression tests for BioJev Sprint8 canonical split contamination handling."""
import copy
import json
import csv
from pathlib import Path
import pytest
from biolaya.comparison.data import build, dump, records

BASE = {
    'id': 'bionli:0', 'source': 'bionli', 'state': 'Patient fever was observed.',
    'question': 'Does the statement follow?',
    'options': [
        {'key': 'entailment', 'description': 'Supported'},
        {'key': 'neutral', 'description': 'Unknown'},
        {'key': 'contradiction', 'description': 'Refuted'}],
    'answer_key': 'entailment'
}


def _make(tmp_path, train, dev):
    dump(tmp_path / 'train.jsonl', train)
    dump(tmp_path / 'dev.jsonl', dev)
    return tmp_path / 'train.jsonl', tmp_path / 'dev.jsonl'


def test_option_permutation_is_leakage(tmp_path):
    x = copy.deepcopy(BASE)
    y = copy.deepcopy(BASE)
    y['options'] = list(reversed(y['options']))
    a,b = _make(tmp_path, [x], [y])
    res = build(a,b,tmp_path/'audit', audit_only=True)
    assert res['overlapping_content_fingerprints'] == 1
    assert res['overlapping_train_rows'] == 1
    assert not (tmp_path/'audit/train.jsonl').exists()


def test_distinct_rows_with_reused_split_id_are_not_leakage(tmp_path):
    a = copy.deepcopy(BASE)
    b = copy.deepcopy(BASE)
    b['state'] = 'Independent patient observation.'
    tr,dv = _make(tmp_path, [a], [b])
    manifest = build(tr, dv, tmp_path/'clean')
    assert manifest['train_rows'] == 1
    assert manifest['fingerprint_overlap'] == 0


def test_explicit_drop_train_preserves_dev_and_all_other_rows(tmp_path):
    x=copy.deepcopy(BASE)
    y=copy.deepcopy(BASE); y['options']=list(reversed(y['options']))
    y['id']='different-dev-id'
    other=copy.deepcopy(BASE)
    other['id']='other'
    other['state']='A completely distinct example.'
    a,b = _make(tmp_path, [x, x, other], [y])
    m = build(a,b,tmp_path/'clean', overlap_policy='drop-train')
    assert m['overlapping_content_fingerprints'] == 1
    assert m['removed_train_rows'] == 2
    assert m['train_rows'] == 1
    assert m['dev_rows'] == 1
    assert m['fingerprint_overlap'] == 0
    assert list(records(tmp_path/'clean'/'dev.jsonl'))[0]['id'] == 'different-dev-id'
    assert list(records(tmp_path/'clean'/'train.jsonl'))[0]['id'] == 'other'
    assert len(list(records(tmp_path/'clean'/'laya_train.jsonl'))) == 1
    assert len(list(records(tmp_path/'clean'/'laya_dev.jsonl'))) == 1
    with (tmp_path/'clean'/'removed_train_rows.csv').open() as f:
        removed=list(csv.DictReader(f))
    assert [r['train_line'] for r in removed] == ['1','2']
    assert json.loads((tmp_path/'clean'/'overlap_audit.json').read_text())['overlapping_train_rows']==2


def test_fail_closed_but_writes_audit(tmp_path):
    a,b = _make(tmp_path, [BASE], [BASE])
    with pytest.raises(ValueError, match=r'TRAIN/DEV LEAKAGE: 1'):
        build(a,b,tmp_path/'strict')
    assert (tmp_path/'strict'/'overlap_audit.json').exists()
    assert not (tmp_path/'strict'/'train.jsonl').exists()


def test_conflicting_labels_reported_and_dev_preserved(tmp_path):
    x = copy.deepcopy(BASE)
    y = copy.deepcopy(BASE); y['answer_key']='contradiction'
    other = copy.deepcopy(BASE)
    other['id'] = 'unrelated'
    other['state'] = 'Independent training item.'
    a,b = _make(tmp_path, [x, other], [y])
    m = build(a,b,tmp_path/'clean', overlap_policy='drop-train')
    assert m['removed_train_rows'] == 1
    with (tmp_path/'clean'/'overlap_examples.csv').open() as f:
        r=list(csv.DictReader(f))
    assert r[0]['conflicting_gold_labels'] == 'True'
    assert list(records(tmp_path/'clean'/'dev.jsonl'))[0]['answer_key'] == 'contradiction'


def test_empty_train_after_filter_fails_safely(tmp_path):
    a,b = _make(tmp_path, [BASE], [BASE])
    with pytest.raises(ValueError, match='empty training set'):
        build(a,b,tmp_path/'clean', overlap_policy='drop-train')
    assert not (tmp_path/'clean'/'train.jsonl').exists()


def test_no_leak_and_caps_continue_to_work(tmp_path):
    tr=[]
    for i in range(4):
        x=copy.deepcopy(BASE)
        x['state']=f'training state {i}'
        tr.append(x)
    y=copy.deepcopy(BASE); y['state']='different development state'
    a,b = _make(tmp_path, tr, [y])
    m=build(a,b,tmp_path/'clean', max_train=2, max_dev=1, overlap_policy='drop-train')
    assert m['train_rows']==2 and m['dev_rows']==1
