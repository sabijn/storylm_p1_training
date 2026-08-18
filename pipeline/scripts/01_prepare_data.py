import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_yaml
from src.data.prepare import prepare_and_split, save_prepared


def main():
    parser = argparse.ArgumentParser(
        description="Load configured dataset sources, split into train/dev/test, and save to disk."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to a data_*.yaml config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    print(f"Preparing data from {len(cfg['sources'])} source(s)...")
    dataset_dict = prepare_and_split(cfg)

    output_dir = cfg["paths"]["output_dir"]
    save_prepared(dataset_dict, output_dir)

    for split_name, split in dataset_dict.items():
        print(f"  {split_name}: {len(split):,} examples")
    print(f"Saved prepared dataset to {output_dir}")


if __name__ == "__main__":
    main()
