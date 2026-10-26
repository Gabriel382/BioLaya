#!/usr/bin/env python3
"""Offline data/metrics smoke run; does NOT validate actual weights."""
import json,tempfile
from pathlib import Path
from biolaya.comparison.data import build,dump,records
from biolaya.comparison.metrics import summarise

def main():
    examples=[{'id':str(i),'source':'bionli','state':f'Test evidence {i}',
      'question':'Evidence supports which conclusion?',
      'options':[{'key':'entailment','description':'supported'},{'key':'contradiction','description':'conflicts'}],
      'answer_key':'entailment' if i%2 else 'contradiction'} for i in range(8)]
    with tempfile.TemporaryDirectory() as t:
        t=Path(t);dump(t/'train.jsonl',examples[:6]);dump(t/'dev.jsonl',examples[6:])
        manifest=build(t/'train.jsonl',t/'dev.jsonl',t/'out')
        assert manifest['train_rows']==6 and manifest['dev_rows']==2
        assert len(list(records(t/'out/laya_train.jsonl')))==6
        rows=[{'model':'biolaya','source':'bionli','gold_key':'entailment','prediction':'entailment',
               'confidence':0.8,'elapsed_seconds':0.03}]
        assert summarise(rows)[0]['accuracy']==1.0
    print('BIO LAYA DATA/METRICS SMOKE: PASS (no model inference)')
if __name__=='__main__':main()
