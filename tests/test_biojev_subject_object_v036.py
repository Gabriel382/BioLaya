"""Real-schema regression tests for BioJev processed ChemProt/DDI/BioRED rows."""
import importlib.util
import json
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('five36',Path(__file__).resolve().parents[1]/'scripts/prepare_five_benchmarks.py')
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def row(task, ident, context, subject, object_, label):
    return {'id':ident, 'dataset':task, 'split':'train', 'task':'relation',
            'context': context, 'subject':subject, 'object':object_,
            'subject_type':'CHEMICAL', 'object_type':'PROTEIN',
            'label':label,'candidates':['A','B','C'],
            'metadata':{'document_id':'doc:' + str(context)}}


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


@pytest.mark.parametrize('task', ['chemprot','ddi2013','biored'])
def test_source_pair_distinguishes_relation_labels(task):
    a=mod.normalize_record(row(task,'rel-1','Same article','p53','BRCA1','A'),task,0)
    b=mod.normalize_record(row(task,'rel-2','Same article','p53','PTEN','B'),task,1)
    assert a['text']==b['text']
    assert a['question']!=b['question']
    assert a['pair_annotations_present'] and b['pair_annotations_present']
    assert a['pair_ids_present']==False
    assert 'p53' in a['question'] and 'BRCA1' in a['question']
    assert a['label']=='A' and b['label']=='B'
    assert 'rel-1' not in a['question']
    assert 'candidates' not in a['question']


@pytest.mark.parametrize('task', ['chemprot','ddi2013','biored'])
def test_real_schema_end_to_end(task,tmp_path):
    root=tmp_path/'data'
    tr=[]
    for i in range(30):
        for j in range(3):
            tr.append(row(task,f'{task}-train-{i}-{j}',f'article-{i}',f'subj{j}',f'obj{j}', ['A','B','A'][j]))
    ev=[row(task,f'{task}-test-{i}',f'different-article-{i}',f'subj{i}',f'obj{i}','A' if i%2 else 'B') for i in range(12)]
    write(root/task/'train.jsonl',tr)
    write(root/task/'test.jsonl',ev)
    out=tmp_path/'out';cfg=tmp_path/'configs'
    result=mod.create_task(task,root,tmp_path/'biojev',out,cfg,'answerdotai/ModernBERT-large',42,.1)
    assert result['status']=='READY'
    assert result['evaluation']==len(ev)
    assert result['pair_annotations_train'] == result['train']
    assert result['pair_ids_train']==0
    train=list(mod.read_rows(out/task/'train.jsonl'))
    dev=list(mod.read_rows(out/task/'dev.jsonl'))
    test=list(mod.read_rows(out/task/'evaluation.jsonl'))
    assert {x['text'] for x in train}.isdisjoint({x['text'] for x in dev})
    assert [x['id'] for x in test]==[x['id'] for x in ev]
    assert result['labels']==['A','B']


def test_inherently_ambiguous_same_pair_still_blocks(tmp_path):
    task='biored';root=tmp_path/'root'
    tr=[]
    for i in range(15):
        tr.append(row(task,f'A{i}',f'doc-{i}','same','same','A'))
        tr.append(row(task,f'B{i}',f'doc-{i}','same','same','B'))
    write(root/task/'train.jsonl',tr)
    write(root/task/'test.jsonl',[row(task,'E','heldout','x','y','A')])
    with pytest.raises(ValueError,match='contradictory labels'):
        mod.create_task(task,root,tmp_path/'b',tmp_path/'out',tmp_path/'cfg','base',42,.1)


def test_structured_subject_and_object_supported():
    a=mod.normalize_record(row('chemprot','id','doc',{'text':'aspirin','id':'S1'},{'text':'COX2','id':'O2'},'A'),'chemprot',0)
    assert 'aspirin' in a['question'] and 'COX2' in a['question']
    assert a['pair_annotations_present']
