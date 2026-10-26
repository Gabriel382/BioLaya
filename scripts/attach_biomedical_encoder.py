#!/usr/bin/env python3
"""Experiment: replace Laya's ModernBERT encoder with biomedical MLM DAPT.

Safety: strict key/shape parity, original checkpoint remains untouched.
No silent partial loading; the resulting checkpoint must pass Laya parity tests.
"""
import argparse, json, copy
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--laya-checkpoint',required=True)
    p.add_argument('--dapt-dir',default='outputs/sprint2/biomodernbert_100m/final')
    p.add_argument('--out',default='outputs/biolaya/systemone/dapt_seed')
    a=p.parse_args()
    if Path(a.out).exists(): raise SystemExit('Output exists; refuse overwrite')
    import torch
    from transformers import AutoModel
    from laya.train import load_checkpoint,save_checkpoint
    decision,tok,cfg=load_checkpoint(a.laya_checkpoint)
    adapted=AutoModel.from_pretrained(a.dapt_dir)
    current=decision.encoder.state_dict(); new=adapted.state_dict()
    missing=sorted(set(current)-set(new));extra=sorted(set(new)-set(current))
    shape={k:[list(current[k].shape),list(new[k].shape)] for k in current.keys()&new.keys() if current[k].shape!=new[k].shape}
    if missing or extra or shape:
        raise RuntimeError('DAPT encoder incompatible; refusing merge: '+json.dumps({'missing':missing[:20],
                           'extra':extra[:20],'mismatched_shapes':shape},indent=2))
    # The saved Laya tokenizer must match DAPT's tokenizer exactly.
    from transformers import AutoTokenizer
    dtok=AutoTokenizer.from_pretrained(a.dapt_dir)
    if tok.get_vocab()!=dtok.get_vocab(): raise RuntimeError('Tokenizers differ; refusing unsafe encoder swap')
    decision.encoder.load_state_dict(new,strict=True)
    save_checkpoint(decision,tok,cfg,a.out)
    (Path(a.out)/'encoder_source.json').write_text(json.dumps({'source':a.dapt_dir,'method':'strict encoder replacement'},indent=2))
    print('Saved DAPT-initialized Laya checkpoint. Validate before training: ',a.out)
if __name__=='__main__':main()
