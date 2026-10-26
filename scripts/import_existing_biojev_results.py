#!/usr/bin/env python3
"""Copy only existing BioJev results, recording provenance. Never run baselines."""
import argparse,hashlib,json,shutil
from pathlib import Path

SOURCES={
 'sprint4_frozen_4b':'results/sprint4/frozen_4b/summary.csv',
 'sprint4_frozen_9b':'results/sprint4/frozen_9b/summary.csv',
 'sprint4_supervised':'results/sprint4/supervised_baselines/summary.csv',
 'sprint8_systemone_repaired':'results/sprint8/native_ollama_eval_full_repaired/summary.csv',
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--biojev-root',default='../BioJev')
 p.add_argument('--out',default='results/reference_biojev');a=p.parse_args()
 root=Path(a.biojev_root);dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
 manifest={}
 for k,rel in SOURCES.items():
    src=root/rel
    if not src.exists():manifest[k]={'status':'missing','path':str(src)};continue
    tgt=dest/(k+'.csv');shutil.copy2(src,tgt)
    digest=hashlib.sha256(tgt.read_bytes()).hexdigest()
    manifest[k]={'status':'copied','origin':str(src.resolve()),'sha256':digest,'destination':str(tgt)}
 (dest/'reference_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
