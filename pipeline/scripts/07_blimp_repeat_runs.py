import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.config import load_yaml
from src.evaluation.blimp_nl import evaluate_blimp_subset


def main():
    parser = argparse.ArgumentParser(
        description="Run BLiMP-NL evaluation multiple times on one checkpoint and write "
        "per-minimal-pair, per-run logprobs to a CSV."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "configs" / "eval_base.yaml",
                         help="Path to a configs/eval_*.yaml (gives model + tokenizer paths).")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of repeated BLiMP-NL runs.")
    parser.add_argument("--output", type=Path, default=None,
                         help="CSV output path (default: <eval output_dir>/blimp_nl_repeat_runs.csv).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_dir = cfg["model"]["path"]
    tokenizer_path = cfg["model"]["tokenizer_path"]
    normalize_by_length = cfg.get("blimp", {}).get("normalize_by_length", True)

    print(f"Loading model from {model_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    output_path = args.output or Path(cfg["paths"]["output_dir"]) / "blimp_nl_repeat_runs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_names = get_dataset_config_names("juletxara/blimp-nl")
    all_rows = []

    for subset_name in subset_names:
        print(f"Loading BLiMP-NL subset: {subset_name}")
        subset_data = load_dataset("juletxara/blimp-nl", subset_name, split="train")

        for run in range(1, args.n_runs + 1):
            rows, summary = evaluate_blimp_subset(
                model, tokenizer, subset_name, subset_data, device, normalize_by_length
            )
            for row in rows:
                row["run"] = run
            all_rows.extend(rows)
            print(f"  run {run}/{args.n_runs}: accuracy = {summary['accuracy']:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df):,} rows ({len(subset_names)} subsets x {args.n_runs} runs) to {output_path}")


if __name__ == "__main__":
    main()
