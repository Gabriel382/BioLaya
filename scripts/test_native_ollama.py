#!/usr/bin/env python3
import argparse,json
from biolaya.comparison.evaluate import post

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='biolaya-systemone:421m')
    p.add_argument('--endpoint',default='http://127.0.0.1:11434');a=p.parse_args()
    questions={
      'category':{'type':'choice','instructions':'What condition is best supported by the evidence?',
         'criteria':{'pneumonia':'Bacterial pneumonia','migraine':'Migraine','asthma':'Asthma'}},
      'flag':{'type':'noul','instructions':'Does the evidence mention fever?'},
      'confidence_level':{'type':'score','instructions':'Rate strength of evidence',
         'criteria':['No evidence','Some evidence','Strong evidence']},
    }
    response=post({'model':a.model,'state':'Fever and lobar infiltrate on chest radiograph.',
                   'questions':questions},a.endpoint)
    answers=response.get('answers',{})
    for q,t in [('category','choice'),('flag','noul'),('confidence_level','score')]:
        answer=answers.get(q)
        if not isinstance(answer,dict):raise AssertionError(f'Missing {q}: {response}')
        if answer.get('type')!=t:raise AssertionError(f'{q} wrong type: {answer}')
        key={'choice':'choice','noul':'noul','score':'score'}[t]
        if key not in answer:raise AssertionError(f'{q} missing {key}: {answer}')
    print('BIO LAYA NATIVE OLLAMA SYSTEMONE E2E: PASS')
    print(json.dumps(response,indent=2))
if __name__=='__main__':main()
