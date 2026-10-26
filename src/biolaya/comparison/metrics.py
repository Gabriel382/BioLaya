from __future__ import annotations
import math
import statistics
from collections import defaultdict


def summarise(rows: list[dict], bins=15) -> list[dict]:
    """Per-(model, source) and per-model macro/micro metrics, failures counted.

    Accuracy and macro-F1 are computed only for successfully predicted items.
    n_success/n_errors always shown; failure rate cannot be hidden.
    ECE uses top-1 probability/confidence where available; Brier here is
    correctness-vs-confidence binary Brier, not multiclass Brier.
    """
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['model'],r['source'])].append(r)
        grouped[(r['model'],'ALL')].append(r)
    result=[]
    for (model,source),arr in sorted(grouped.items()):
        ok=[r for r in arr if not r.get('error') and r.get('prediction') is not None]
        gold=[str(r['gold_key']) for r in ok]
        pred=[str(r['prediction']) for r in ok]
        labels=sorted(set(gold)|set(pred))
        correct=[int(a==b) for a,b in zip(gold,pred)]
        f1s=[]
        for label in labels:
            tp=sum(g==label and p==label for g,p in zip(gold,pred))
            fp=sum(g!=label and p==label for g,p in zip(gold,pred))
            fn=sum(g==label and p!=label for g,p in zip(gold,pred))
            f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
        cal=[(float(r['confidence']),int(r['gold_key']==r['prediction'])) for r in ok
             if r.get('confidence') is not None and math.isfinite(float(r['confidence']))
             and 0<=float(r['confidence'])<=1]
        ece=None; brier=None
        # Multiclass Brier and NLL, comparable to the original benchmark only
        # when option dictionaries describe the *same* target label space.
        multi_brier=[]; nll=[]
        for r in ok:
            probs=r.get('probabilities')
            if isinstance(probs,dict) and len(probs)>=2:
                try:
                    pp={str(k):float(v) for k,v in probs.items()}
                    if any(not math.isfinite(v) or v<0 or v>1 for v in pp.values()):continue
                    if str(r['gold_key']) not in pp:continue
                    if abs(sum(pp.values())-1)>0.05:continue
                    multi_brier.append(sum((v-int(k==str(r['gold_key'])))**2 for k,v in pp.items()))
                    nll.append(-math.log(max(pp[str(r['gold_key'])],1e-12)))
                except (TypeError,ValueError):pass
        if cal:
            brier=sum((p-y)**2 for p,y in cal)/len(cal)
            ece=0.0
            for i in range(bins):
                sub=[(p,y) for p,y in cal if (i/bins <=p< (i+1)/bins or (i==bins-1 and p==1))]
                if sub: ece+=(len(sub)/len(cal))*abs(sum(p for p,y in sub)/len(sub)-sum(y for p,y in sub)/len(sub))
        latency=sorted(float(r['elapsed_seconds']) for r in ok if r.get('elapsed_seconds') is not None)
        quant=lambda q: latency[min(len(latency)-1,int((len(latency)-1)*q))] if latency else None
        result.append({'model':model,'source':source,'n':len(arr),'n_success':len(ok),'n_errors':len(arr)-len(ok),
                       'accuracy':sum(correct)/len(ok) if ok else None,
                       'f1_macro':sum(f1s)/len(f1s) if f1s else None,
                       'ece_top1':ece,'brier_correctness':brier,'brier_multiclass':sum(multi_brier)/len(multi_brier) if multi_brier else None,
                       'nll':sum(nll)/len(nll) if nll else None,'n_probabilities':len(multi_brier),
                       'n_confidence':len(cal),
                       'median_latency_s':statistics.median(latency) if latency else None,
                       'p95_latency_s':quant(.95)})
    return result
