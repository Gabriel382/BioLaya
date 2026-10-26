import pytest
from pathlib import Path
from biolaya.comparison.data import normalise,to_laya,build,dump
from biolaya.comparison.metrics import summarise

EX={'id':'ex1','source':'bionli','state':'Trial A succeeded.','question':'Does this hypothesis follow?',
    'options':[{'key':'entailment','description':'Supported'},{'key':'contradiction','description':'Conflicted'}],
    'answer_key':'entailment'}

def test_normalise():
    assert normalise(EX)['answer_key']=='entailment'
    r=to_laya(EX)
    assert r['expected']['decision']=='entailment'
    assert r['questions']['decision']['criteria']['entailment']=='Supported'

def test_rejects_missing_gold():
    x={**EX,'answer_key':'neutral'}
    with pytest.raises(ValueError):normalise(x)

def test_rejects_leakage(tmp_path):
    dump(tmp_path/'a.jsonl',[EX]);dump(tmp_path/'b.jsonl',[EX])
    with pytest.raises(ValueError,match='LEAKAGE'): build(tmp_path/'a.jsonl',tmp_path/'b.jsonl',tmp_path/'out')

def test_build_deterministic(tmp_path):
    x={**EX,'id':'ex2','state':'Trial B failed.'}
    dump(tmp_path/'a.jsonl',[EX]);dump(tmp_path/'b.jsonl',[x])
    m=build(tmp_path/'a.jsonl',tmp_path/'b.jsonl',tmp_path/'out')
    assert m['train_rows']==m['dev_rows']==1
    assert m['fingerprint_overlap']==0

def test_metrics():
    rows=[{'model':'x','source':'a','gold_key':'yes','prediction':'yes','confidence':.9,'elapsed_seconds':.2},
          {'model':'x','source':'a','gold_key':'yes','prediction':'no','confidence':.6,'elapsed_seconds':.3},
          {'model':'x','source':'a','gold_key':'yes','prediction':None,'error':'overflow'}]
    overall=next(x for x in summarise(rows) if x['source']=='ALL')
    assert overall['n']==3 and overall['n_success']==2 and overall['n_errors']==1
    assert overall['accuracy']==.5
