"""RE conflicting gold annotations should not block faithful single-label eval."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('five37',Path(__file__).resolve().parents[1]/'scripts/prepare_five_benchmarks.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make(task, doc, pair, label, index):
    return {'id':f'{task}-{doc}-{index}', 'task':'relation','dataset':task,
            'context':f'{doc} biomedical paper content.',
            'subject':pair[0], 'object':pair[1],
            'subject_type':'CHEMICAL', 'object_type':'PROTEIN',
            'label':label, 'candidates':['confounding label not used'],
            'metadata':{'document_id':doc}}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


@pytest.mark.parametrize('task',('chemprot','ddi2013','biored'))
def test_conflicting_same_pair_is_preserved_and_metrics_audited(tmp_path,task):
    root=tmp_path/'raw'
    train=[]
    for i in range(30):
        train.append(make(task,f'doc{i}',('E1','E2'),'RegA',i*3))
        train.append(make(task,f'doc{i}',('E1','E2'),'RegB',i*3+1))
        train.append(make(task,f'doc{i}',('E1','E3'),'RegA',i*3+2))
    val=[make(task,f'val{i}',('E1','E2'),('RegA' if i%2 else 'RegB'),i) for i in range(15)]
    test=[make(task,f'test{i}',('E1','E2'),('RegB' if i%2 else 'RegA'),i) for i in range(20)]
    write(root/task/'train.jsonl',train)
    if task!='ddi2013':write(root/task/'validation.jsonl',val)
    write(root/task/'test.jsonl',test)
    out=tmp_path/'out'
    result=mod.create_task(task,root,tmp_path/'biojev',out,tmp_path/'configs','base',42,.1)
    assert result['status']=='READY'
    assert result['evaluation']==len(test)
    assert result['labels']==['RegA','RegB']
    all_train=list(mod.read_rows(out/task/'train.jsonl'))
    all_dev=list(mod.read_rows(out/task/'dev.jsonl'))
    all_eval=list(mod.read_rows(out/task/'evaluation.jsonl'))
    assert len(all_train)+len(all_dev)==len(train)+(0 if task=='ddi2013' else len(val))
    assert [x['id'] for x in all_eval]==[x['id'] for x in test]
    assert {x['document_id'] for x in all_train}.isdisjoint({x['document_id'] for x in all_dev})
    assert {x['document_id'] for x in all_train}.isdisjoint({x['document_id'] for x in all_eval})
    assert all('RegA' not in x['question'] and 'RegB' not in x['question'] for x in all_train)
    assert all(x['id'] not in x['question'] for x in all_train)
    a=result['input_ambiguity']
    assert a['train']['conflicting_model_inputs'] > 0 or a['dev']['conflicting_model_inputs']>0
    assert a['evaluation']['total_rows']==len(test)
    assert all(x['label'] in ('RegA','RegB') for x in all_train+all_dev+all_eval)


def test_accuracy_ceiling_for_indistinguishable_inputs():
    proto={'id':'x','source':'ddi2013','text':'Same clinical context','question':'Same target pair'}
    rows=[{**proto,'label':'EFFECT'},{**proto,'label':'ADVISE'},
          {**proto,'label':'ADVISE'},{**proto,'label':'MECHANISM'}]
    a=mod.input_ambiguity(rows)
    assert a['conflicting_model_inputs']==1
    assert a['rows_on_conflicting_inputs']==4
    assert a['irreducible_errors_for_deterministic_input']==2
    assert a['empirical_accuracy_ceiling_given_only_identical_model_inputs']==0.5


def test_document_id_provenance_not_in_model_input():
    raw=make('biored','NCBI:123456',('drug','gene'),'Association',0)
    raw['context']='A biomedical paper with drug-gene association.'
    x=mod.normalize_record(raw,'biored',0)
    y=mod.normalize_record({**raw,'metadata':{'document_id':'NCBI:789'}},'biored',0)
    assert x['document_id']=='NCBI:123456'
    assert x['text']==y['text'] and x['question']==y['question']
    assert mod.document_fingerprint(x)!=mod.document_fingerprint(y)
    assert 'NCBI:123456' not in x['question'] and 'NCBI:123456' not in x['text']
