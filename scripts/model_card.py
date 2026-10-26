#!/usr/bin/env python3
"""Generate honest HF/Ollama cards, only from real run files."""
import argparse,csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',required=True)
    p.add_argument('--model-name',default='BioLaya-SystemOne');p.add_argument('--out',default='outputs/MODEL_CARD.md')
    a=p.parse_args()
    with open(a.metrics,newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise SystemExit('No metrics in summary CSV')
    results='\n'.join('| {model} | {source} | {n} | {accuracy} | {f1_macro} | {n_errors} |'.format(**{k:r.get(k,'') for k in ['model','source','n','accuracy','f1_macro','n_errors']}) for r in rows)
    card=f"""# {a.model_name}\n\nA Laya/ModernBERT biomedical decision model.\n\n## Evaluation\n\n| Model | Source | N | Accuracy | Macro-F1 | Errors |\n|---|---|---:|---:|---:|---:|\n{results}\n\n**Caveats:** If the data split was used during training or model selection, these are development results, not an independent test. Training was not clinically validated. Confidence is not assured to be calibrated outside the calibration domain. Benchmark comparisons across different protocols are not valid.\n\n## Code\n\nhttps://github.com/Gabriel382/BioLaya\n\n## Acknowledgments\n\nBased on Laya (Convai Innovations). See upstream Apache-2.0 attribution and weights license.\n"""
    dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(card)
    print(dst)
if __name__=='__main__':main()
