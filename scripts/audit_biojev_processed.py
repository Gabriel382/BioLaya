#!/usr/bin/env python3
"""Audit BioJev processed benchmark schemas & target pair coverage.

No raw document, patient string, entity name or training example is printed.
Compatible with original BioLaya BigBio choice records and common label rows.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from prepare_five_benchmarks import DATASETS,filepath,read_rows,normalize_record,fingerprint,document_fingerprint


def audit_file(source, split, path):
    keys=Counter();meta_keys=Counter();labels=Counter();failures=Counter()
    input_labels=defaultdict(set); input_counts=Counter(); pair_covered=0; pair_id_covered=0
    documents=Counter(); total=0; root_subject_object_coverage=0; root_subject_object_distinct=set()
    for i,row in enumerate(read_rows(path)):
        total+=1
        keys.update(row.keys())
        if row.get('subject') not in (None,'') and row.get('object') not in (None,''):
            root_subject_object_coverage+=1
            # Only counts unique pair descriptions, not raw biomedical names.
            root_subject_object_distinct.add((str(row['subject']),str(row['object'])))
        meta=row.get('metadata')
        if isinstance(meta,dict):meta_keys.update(meta.keys())
        try:
            item=normalize_record(row,source,i)
            labels[item['label']]+=1
            pair_covered+=int(item.get('pair_annotations_present',False))
            pair_id_covered+=int(item.get('pair_ids_present',False))
            inp=fingerprint(item)
            input_counts[inp]+=1
            input_labels[inp].add(item['label'])
            documents[document_fingerprint(item)]+=1
        except Exception as exc:
            failures[str(exc).split(';')[0][:135]]+=1
    return {'dataset':source,'split':split,'path':str(path),'n':total,
            'root_keys':sorted(keys),'metadata_keys':sorted(meta_keys),
            'label_names':dict(labels.most_common(30)),
            'pair_annotation_coverage':pair_covered,'pair_id_coverage':pair_id_covered,
            'root_subject_object_coverage':root_subject_object_coverage,
            'distinct_subject_object_pairs':len(root_subject_object_distinct),
            'distinct_document_contexts':len(documents),
            'repeated_model_inputs':sum(n-1 for n in input_counts.values() if n>1),
            'conflicting_label_inputs':sum(len(labs)>1 for labs in input_labels.values()),
            'normalization_errors':dict(failures.most_common(10))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='../BioJev/data/processed')
    p.add_argument('--out',default='data/standard/five/schema_audit.json')
    args=p.parse_args()
    report=[]
    for name in DATASETS:
        for split in ('train','dev','validation','test'):
            path=filepath(args.root,name,split)
            if path:
                result=audit_file(name,split,path)
                report.append(result)
                print(json.dumps(result,ensure_ascii=False))
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Wrote:',dest)
if __name__=='__main__': main()
