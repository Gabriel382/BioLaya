from biolaya.comparison.data import dump,records
from biolaya.comparison.standard import prepare

def test_prepare_from_canonical(tmp_path):
 row={'source':'chemprot','id':'1','state':'A inhibits B','question':'Relation?',
      'options':[{'key':'CPR:3','description':'upregulates'},{'key':'CPR:4','description':'downregulates'}],
      'answer_key':'CPR:4'}
 other={**row,'id':'2','state':'C upregulates D','answer_key':'CPR:3'}
 dump(tmp_path/'tr.jsonl',[row,other]);dump(tmp_path/'dev.jsonl',[row])
 m=prepare(tmp_path/'tr.jsonl',tmp_path/'dev.jsonl',tmp_path/'out')
 assert m['labels']==['CPR:3','CPR:4']
 assert list(records(tmp_path/'out/train.jsonl'))[0]['label']=='CPR:4'
