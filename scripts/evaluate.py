from __future__ import annotations

import argparse, json
from biolaya.evaluation import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-file", required=True)
    p.add_argument("--model", default="english")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-examples", type=int)
    p.add_argument("--output-dir", default="results/eval")
    a = p.parse_args()
    print(json.dumps(evaluate(a.dataset_file, model=a.model, device=a.device, max_examples=a.max_examples, output_dir=a.output_dir), indent=2))

if __name__ == "__main__":
    main()
