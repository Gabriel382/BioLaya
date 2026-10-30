#!/usr/bin/env python3
"""Safe schema/size introspection. No patient text or records printed."""
import argparse
from pathlib import Path
import json
from collections import Counter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='../BioJev/data/processed')
    a=p.parse_args()
    for dataset in ('bionli','nli4ct','chemprot','ddi2013','biored'):
        for split in ('train','dev','validation','test'):
            path=Path(a.root)/dataset/(split+'.jsonl')
            if not path.exists():continue
            count=0
            fields=Counter()
            labels=Counter()
            with path.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():continue
                    obj=json.loads(line);count+=1
                    fields.update(obj.keys())
                    if isinstance(obj.get('label'),str):labels[obj['label']]+=1
            print(json.dumps({'dataset':dataset,'split':split,'n':count,
                'keys':sorted(fields),'label_names':dict(labels.most_common(20))}))

if __name__=='__main__':main()
