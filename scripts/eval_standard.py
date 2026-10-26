#!/usr/bin/env python3
"""Standard BioLaya held-out evaluation with per-item scores and metrics.
"""
import argparse,json,time,csv
from pathlib import Path
from biolaya.comparison.data import records,dump
from biolaya.comparison.metrics import summarise

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-length',type=int,default=512)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    tok=AutoTokenizer.from_pretrained(a.checkpoint)
    model=AutoModelForSequenceClassification.from_pretrained(a.checkpoint)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(dev).eval()
    samples=list(records(a.data))
    if a.limit:samples=samples[:a.limit]
    outputs=[]
    for idx,row in enumerate(samples,1):
        t=time.perf_counter()
        with torch.inference_mode():
            inputs=tok(row['text']+'\n\n'+row['question'],max_length=a.max_length,truncation=True,return_tensors='pt')
            prob=torch.softmax(model(**{k:v.to(dev) for k,v in inputs.items()}).logits.float(),dim=-1)[0].cpu()
        label=str(model.config.id2label[int(prob.argmax())]);true=row['label']
        outputs.append({'model':a.checkpoint,'source':row['source'],'id':row['id'],'gold_key':true,
            'prediction':label,'confidence':round(float(prob.max()),8),
            'gold_probability':round(float(prob[model.config.label2id[true]]),8),
            'probabilities':{str(model.config.id2label[i]):round(float(v),8) for i,v in enumerate(prob)},
            'elapsed_seconds':round(time.perf_counter()-t,6),'error':None})
        if idx%100==0: print(f'{idx}/{len(samples)}',flush=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    dump(out/'predictions.jsonl',outputs)
    summary=summarise(outputs)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
