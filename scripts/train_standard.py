#!/usr/bin/env python3
"""Fine-tune standard ModernBERT classifier on NLI/RE data from same BioJev inputs.

Use --config for sequential general-NLI->biomedical-NLI experiments.
Never train on test or calibrate using test.
"""
import argparse,inspect,json
from pathlib import Path
import numpy as np
import yaml
from biolaya.comparison.data import records

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/standard/bionli.yaml')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding
    cfg=yaml.safe_load(Path(a.config).read_text()); d=cfg['data'];t=cfg['training'];m=cfg['model']
    tr=list(records(d['train']));dv=list(records(d['dev']))
    if a.smoke:tr=tr[:max(10,min(64,len(tr)))];dv=dv[:max(4,min(16,len(dv)))]
    labels=list(cfg['labels'])
    def rec_to_dict(rows):
        for row in rows:
            if row['label'] not in labels:raise ValueError(f"Unknown class {row['label']!r}; expected {labels}")
        return [{'text':row['text']+'\n\n'+row['question'],'label':labels.index(row['label'])} for row in rows]
    from datasets import Dataset
    train=Dataset.from_list(rec_to_dict(tr));dev=Dataset.from_list(rec_to_dict(dv))
    checkpoint=m['base']
    tok=AutoTokenizer.from_pretrained(checkpoint)
    def enc(batch):return tok(batch['text'],max_length=int(t.get('max_length',512)),truncation=True)
    train=train.map(enc,batched=True,remove_columns=['text']);dev=dev.map(enc,batched=True,remove_columns=['text'])
    model=AutoModelForSequenceClassification.from_pretrained(
        checkpoint,num_labels=len(labels),id2label={i:k for i,k in enumerate(labels)},
        label2id={k:i for i,k in enumerate(labels)},ignore_mismatched_sizes=True)
    desired=dict(output_dir=m['output'],num_train_epochs=1 if a.smoke else float(t.get('epochs',3)),
        max_steps=2 if a.smoke else -1,learning_rate=float(t.get('learning_rate',2e-5)),
        per_device_train_batch_size=int(t.get('batch_size',4)),
        per_device_eval_batch_size=int(t.get('eval_batch_size',8)),
        gradient_accumulation_steps=int(t.get('grad_accum',8)),
        bf16=bool(t.get('bf16',True)),gradient_checkpointing=bool(t.get('gradient_checkpointing',True)),
        logging_steps=int(t.get('logging_steps',20)),save_strategy='epoch',eval_strategy='epoch',
        save_total_limit=2,remove_unused_columns=True,report_to='none',seed=int(cfg.get('seed',42)))
    params=inspect.signature(TrainingArguments.__init__).parameters
    if 'eval_strategy' not in params:
        desired['evaluation_strategy']=desired.pop('eval_strategy')
    desired={k:v for k,v in desired.items() if k in params}
    args=TrainingArguments(**desired)
    from sklearn.metrics import f1_score,accuracy_score
    def metrics(p):
        preds=np.asarray(p.predictions).argmax(-1)
        return {'accuracy':float(accuracy_score(p.label_ids,preds)),
                'f1_macro':float(f1_score(p.label_ids,preds,labels=list(range(len(labels))),average='macro',zero_division=0))}
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=dev,
                   processing_class=tok,data_collator=DataCollatorWithPadding(tok),compute_metrics=metrics)
    trainer.train()
    ev=trainer.evaluate();out=Path(m['output']);final=out/'final'
    trainer.save_model(str(final));tok.save_pretrained(str(final))
    (out/'training_manifest.json').write_text(json.dumps({'config':cfg,'evaluation':ev,'status':'complete'},indent=2,default=str))
    print(json.dumps({'checkpoint':str(final),'metrics':ev},indent=2))
if __name__=='__main__':main()
