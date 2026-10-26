from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from .data import records,to_laya,to_ollama,dump
from .metrics import summarise
import csv


def post(payload,base_url='http://127.0.0.1:11434',timeout=120):
    req=urllib.request.Request(base_url.rstrip('/')+'/v1/systemone',
       data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as response: return json.load(response)


def evaluate_file(data_path, models: list[str], *, backend='laya', endpoint=None,
                  output='results/comparison', limit=0,skip_existing=False,timeout=180):
    examples=list(records(data_path))
    if limit: examples=examples[:limit]
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    final=out/'predictions.jsonl'
    rows=list(records(final)) if skip_existing and final.exists() else []
    seen={(r['model'],r['source'],r['id']) for r in rows}
    for model in models:
        if backend=='laya':
            import laya
            agent=laya.load(model)
        else: agent=None
        for i,x in enumerate(examples,1):
            key=(model,str(x['source']),str(x['id']))
            if key in seen: continue
            r={'model':model,'backend':backend,'source':x['source'],'id':x['id'],'gold_key':x['answer_key']}
            start=time.perf_counter()
            try:
                if backend=='laya':
                    d=to_laya(x)
                    response=agent.predict(d['state'],d['questions'])
                elif backend=='ollama':
                    response=post(to_ollama(x,model),endpoint or 'http://127.0.0.1:11434',timeout)
                else: raise ValueError('backend must be laya or ollama')
                answers=response.get('answers',{})
                answer=answers['decision']
                guess=answer.get('choice')
                if guess is None: raise RuntimeError(f'no choice in answer: {answer}')
                probabilities=answer.get('probabilities') or answer.get('distribution') or {}
                if not isinstance(probabilities,dict): probabilities={}
                r.update({'prediction':str(guess),'confidence':answer.get('confidence'),
                          'gold_probability':probabilities.get(str(x['answer_key'])),
                          'probabilities':probabilities,'error':None})
            except Exception as e:
                r.update({'prediction':None,'confidence':None,'gold_probability':None,
                          'probabilities':None,'error':str(e)})
            r['elapsed_seconds']=round(time.perf_counter()-start,6)
            rows.append(r);seen.add(key)
            if i%50==0: print(f'{model} {i}/{len(examples)} | failures={sum(bool(a.get("error")) for a in rows if a["model"]==model)}',flush=True)
        dump(final,rows)
    report=summarise(rows)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(report[0]) if report else []);
        if report: w.writeheader();w.writerows(report)
    (out/'run.json').write_text(json.dumps({'data':str(data_path),'models':models,'backend':backend,
       'endpoint':endpoint,'limit':limit,'note':'No baseline models rerun. Development data; 512/1024 context may truncate in Laya.'},indent=2))
    return report
