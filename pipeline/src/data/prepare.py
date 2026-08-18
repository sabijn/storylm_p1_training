from typing import Any
import os

from datasets import DatasetDict, concatenate_datasets, load_from_disk

from .sources import load_all_sources


def prepare_and_split(data_cfg: dict[str, Any]) -> DatasetDict:
    """Load all configured sources, combine them, and split into train/dev/test.
    """
    datasets = load_all_sources(data_cfg["sources"])
    combined = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

    split_cfg = data_cfg["split"] # info on fractions
    seed = split_cfg.get("seed", 42)
    dev_fraction = split_cfg["dev_fraction"]
    test_fraction = split_cfg["test_fraction"]
    holdout_fraction = dev_fraction + test_fraction

    combined = combined.shuffle(seed=seed)

    train_holdout = combined.train_test_split(test_size=holdout_fraction, seed=seed)
    train_ds = train_holdout["train"]
    holdout_ds = train_holdout["test"]

    dev_relative_size = dev_fraction / holdout_fraction
    dev_test = holdout_ds.train_test_split(test_size=1 - dev_relative_size, seed=seed)

    return DatasetDict({"train": train_ds, "dev": dev_test["train"], "test": dev_test["test"]})


def save_prepared(dataset_dict: DatasetDict, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_dict.save_to_disk(output_dir)


def load_prepared(dataset_dir: str) -> DatasetDict:
    return load_from_disk(dataset_dir)
