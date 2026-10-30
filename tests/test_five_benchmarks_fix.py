import importlib.util
import json
from pathlib import Path

script=Path(__file__).resolve().parents[1]/'scripts/prepare_five_benchmarks.py'
spec=importlib.util.spec_from_file_location('five_fix',script)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


def test_adapter_shapes():
    assert mod.normalize_record({'id':'a','text':'[E1] A [E2] B','label':'CPR:3'},'chemprot',0)['label']=='CPR:3'
    assert mod.normalize_record({'id':'b','premise':'evidence','hypothesis':'claim','label':'entailment'},'bionli',0)['question']=='Hypothesis: claim'
    obj={'id':'c','dataset':'biored','state':'evidence','type':'choice','instructions':'Which relation?','criteria':{'A':'a','B':'b'},'gold':'A'}
    assert mod.normalize_record(obj,'biored',0)['label']=='A'


def test_missing_label_fails():
    try: mod.normalize_record({'id':'x','text':'Sentence','label':1},'chemprot',0)
    except ValueError as e: assert 'numeric class' in str(e)
    else: raise AssertionError('numeric labels must not be guessed')


def test_all_five_prepare(tmp_path):
    root=tmp_path/'biojev/data/processed'
    biojev=tmp_path/'biojev'
    pred=biojev/'results/sprint4/frozen_4b/biojev-4b__bionli__test__seed42/predictions.jsonl'
    write(pred,[{'id':'bionli:0','gold':'entailment'},{'id':'bionli:9','gold':'contradiction'}])
    def label(i):return 'entailment' if i%2 else 'contradiction'
    write(root/'bionli/train.jsonl',[{'id':f'bionli-{i}','premise':f'evidence {i}','hypothesis':'x','label':label(i)} for i in range(100)])
    write(root/'nli4ct/train.jsonl',[{'id':f'nli4ct:{i}','premise':f'trial {i}','hypothesis':'treatment effective','label':label(i)} for i in range(30)])
    write(root/'nli4ct/dev.jsonl',[{'id':f'dev:{i}','premise':f'other trial {i}','hypothesis':'treatment effective','label':label(i)} for i in range(5)])
    for name in ('chemprot','ddi2013','biored'):
        write(root/name/'train.jsonl',[{'id':f'{name}-{i}','text':f'[E1] text {i} [E2]','label':'A' if i%2 else 'B'} for i in range(30)])
        if name!='ddi2013':
            write(root/name/'dev.jsonl',[{'id':f'{name}-dev-{i}','text':f'[E1] dev {i} [E2]','label':'A' if i%2 else 'B'} for i in range(6)])
        write(root/name/'test.jsonl',[{'id':f'{name}-test-{i}','text':f'[E1] test {i} [E2]','label':'A' if i%2 else 'B'} for i in range(8)])
    summary={}
    for name in mod.DATASETS:
        summary[name]=mod.create_task(name,root,biojev,tmp_path/'out',tmp_path/'configs','answerdotai/ModernBERT-large',42,.1)
    assert all(d['status']=='READY' for d in summary.values())
    assert summary['bionli']['evaluation']==2
    assert summary['bionli']['evaluation_original_split']=='test'
    assert summary['nli4ct']['evaluation_original_split']=='dev'
    assert (tmp_path/'out/nli4ct/evaluation.jsonl').is_file()
    assert not (tmp_path/'out/nli4ct/test.jsonl').exists()
    assert summary['ddi2013']['internal_dev_derived_from_training'] is True


def test_wrong_bionli_ids_fail(tmp_path):
    root=tmp_path/'processed'
    pred=tmp_path/'ref/results/sprint4/frozen_4b/biojev-4b__bionli__test__seed42/predictions.jsonl'
    write(pred,[{'id':'bionli:1234'}])
    write(root/'bionli/train.jsonl',[{'id':'bionli-1','premise':'x','hypothesis':'y','label':'entailment'}])
    try: mod.create_task('bionli',root,tmp_path/'ref',tmp_path/'out',tmp_path/'config','base',42,.1)
    except ValueError as e: assert 'ID mismatch' in str(e)
    else: raise AssertionError('should not regenerate a random test')
