#!/usr/bin/env python3
"""Native Laya export gate; do not misuse the Qwen causal GGUF converter.

GGUF must include the *entire* Laya decision network. A generic
ModernBERT embedding GGUF is not a BioLaya SystemOne model.

Pass an already verified Laya-compatible GGUF or a converter command template
from a matching release. Export scripts are evolving across runtimes.
"""
import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

def gguf_check(path):
    p=Path(path)
    if not p.exists() or p.stat().st_size<1024: raise RuntimeError(f'GGUF absent/incomplete: {p}')
    with p.open('rb') as f:
        if f.read(4)!=b'GGUF': raise RuntimeError('Not a GGUF file')
    return p.resolve()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='outputs/biolaya/systemone/final')
    p.add_argument('--gguf',required=True)
    p.add_argument('--converter-command',help='Explicit trusted Laya converter: e.g. python path/to/converter.py {checkpoint} --outfile {gguf}')
    p.add_argument('--ollama-tag',default='biolaya-systemone:421m')
    p.add_argument('--num-ctx',type=int,default=1024)
    p.add_argument('--create',action='store_true')
    p.add_argument('--output-dir',default='outputs/biolaya/systemone/ollama')
    a=p.parse_args()
    if a.converter_command:
        cmd=[x.replace('{checkpoint}',str(Path(a.checkpoint).resolve())).replace('{gguf}',str(Path(a.gguf).resolve()))
             for x in shlex.split(a.converter_command)]
        subprocess.run(cmd,check=True)
    gguf=gguf_check(a.gguf)
    if a.num_ctx<512: raise ValueError('num_ctx must be >=512')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    mf=out/'Modelfile'
    mf.write_text(f'FROM {gguf}\nCAPABILITY decision\nPARAMETER num_ctx {a.num_ctx}\n',encoding='utf-8')
    print('Prepared',mf,'\nNative Laya GGUF decision-head support MUST be validated with end-to-end API parity.')
    if a.create:
        subprocess.run(['ollama','create',a.ollama_tag,'-f',str(mf)],check=True)
        print('Registered',a.ollama_tag,'; run scripts/test_native_ollama.py before publishing')
if __name__=='__main__':main()
