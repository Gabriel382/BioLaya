"""Regression tests based on BioLaya original BigBio adapters and BioJev test membership."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('five35',Path(__file__).resolve().parents[1]/'scripts/prepare_five_benchmarks.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')


def choice(id,source,state,label,*,arg1='T1',arg2='T2',e1='A',e2='B'):
    return {'id':id,'dataset':source,'split':'train','state':state,'question_id':'relation','type':'choice',
      'instructions':f'What relation is annotated between entity 1 ({e1}) and entity 2 ({e2})?',
      'criteria':{'A':'Relation A','B':'Relation B'},'gold':label,
      'metadata':{'entity1':e1,'entity2':e2,'arg1_id':arg1,'arg2_id':arg2}}


def test_original_biolaya_metadata_target_pair_used():
    a=mod.normalize_record(choice('a','chemprot','Same document','A',arg1='T1',arg2='T2'),'chemprot',0)
    b=mod.normalize_record(choice('b','chemprot','Same document','B',arg1='T3',arg2='T4'),'chemprot',1)
    assert a['text']==b['text']
    assert a['question'] != b['question']
    assert a['pair_ids_present'] and b['pair_ids_present']
    assert 'entity1=T1' in a['question']
    assert 'entity1=T3' in b['question']
    assert a['label']=='A' and b['label']=='B'


def test_distinct_pair_synthetic_full_pipeline(tmp_path):
    root=tmp_path/'processed'
    for task in ('chemprot','ddi2013','biored'):
        tr=[]
        for i in range(20):
            # 2 annotated pairs in each document; identical text and entity names
            tr.append(choice(f'{task}-{i}-A',task,f'document {i}','A',arg1='T1',arg2='T2'))
            tr.append(choice(f'{task}-{i}-B',task,f'document {i}','B',arg1='T3',arg2='T4'))
        ev=[choice(f'eval-{i}',task,f'different document {i}', 'A' if i%2 else 'B') for i in range(8)]
        write(root/task/'train.jsonl',tr)
        write(root/task/'test.jsonl',ev)
        if task!='ddi2013':
            write(root/task/'dev.jsonl',[choice(f'dev-{i}',task,f'validation {i}', 'A' if i%2 else 'B') for i in range(6)])
        report=mod.create_task(task,root,tmp_path,tmp_path/'out',tmp_path/'config','base',42,.1)
        assert report['status']=='READY',report
        assert report['pair_ids_train'] == report['train']
        assert report['evaluation']==8


def test_bionli_eval_duplicate_removed_before_dev(tmp_path):
    root=tmp_path/'processed'
    biojev=tmp_path/'biojev'
    path=biojev/'results/sprint4/frozen_4b/biojev-4b__bionli__test__seed42/predictions.jsonl'
    write(path,[{'id':'bionli:0','gold':'entailment'}, {'id':'bionli:1','gold':'contradiction'}])
    rows=[]
    for i in range(80):
        duplicate = i in (2,3,4,5,6,7,8,9)
        rows.append({'id':f'bionli:{i}', 'premise':f'evidence {i%2 if duplicate else i}',
            'hypothesis':'hypothesis','label':'entailment' if i%2 else 'contradiction'})
    # 8 distinct IDs with texts identical to the frozen evaluation examples
    write(root/'bionli/train.jsonl',rows)
    out=tmp_path/'out'
    report=mod.create_task('bionli',root,biojev,out,tmp_path/'config','base',42,.1)
    assert report['status']=='READY'
    assert report['additional_training_removed_for_evaluation_overlap_before_holdout']==8
    train=list(mod.read_rows(out/'bionli/train.jsonl'))
    dev=list(mod.read_rows(out/'bionli/dev.jsonl'))
    evaluation=list(mod.read_rows(out/'bionli/evaluation.jsonl'))
    assert not ({mod.fingerprint(r) for r in train}&{mod.fingerprint(r) for r in dev+evaluation})
    assert not ({mod.fingerprint(r) for r in dev}&{mod.fingerprint(r) for r in evaluation})


def test_same_pair_different_labels_preserved_for_singlelabel_training(tmp_path):
    root=tmp_path/'processed'
    rows=[]
    for i in range(8):
        rows.append(choice(f'x{i}', 'biored', f'doc-{i}', 'A'))
        rows.append(choice(f'x{i}b','biored',f'doc-{i}','B'))
    write(root/'biored/train.jsonl',rows)
    write(root/'biored/dev.jsonl',[choice('d','biored','dev doc','A')])
    write(root/'biored/test.jsonl',[choice('t','biored','test doc','B')])
    report=mod.create_task('biored',root,tmp_path,tmp_path/'out',tmp_path/'config','base',42,.1)
    assert report['status']=='READY'
    assert report['input_ambiguity']['train']['conflicting_model_inputs']==8
    train=list(mod.read_rows(tmp_path/'out/biored/train.jsonl'))
    assert len(train)==16

def test_unannotated_relation_conflict_is_audited_not_leaked(tmp_path):
    root=tmp_path/'processed'
    rows=[]
    for i in range(8):
        rows.append({'id':f'no-target-{i}-a','text':f'same doc {i}','label':'A'})
        rows.append({'id':f'no-target-{i}-b','text':f'same doc {i}','label':'B'})
    write(root/'chemprot/train.jsonl',rows)
    write(root/'chemprot/dev.jsonl',[{'id':'d','text':'valid','label':'A'}])
    write(root/'chemprot/test.jsonl',[{'id':'t','text':'test','label':'B'}])
    result=mod.create_task('chemprot',root,tmp_path,tmp_path/'out',tmp_path/'configs','base',42,.1)
    assert result['status']=='READY'
    assert result['input_ambiguity']['train']['conflicting_model_inputs']==8
    all_rows=list(mod.read_rows(tmp_path/'out/chemprot/train.jsonl'))
    assert all('no-target-' not in r['question'] for r in all_rows)
