from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset


def load_source(source_cfg: dict[str, Any]) -> Dataset:
    """Load one configured dataset source and normalize it to a single "text" column."""
    source_type = source_cfg["type"]
    text_column = source_cfg["text_column"]

    if source_type == "hf":
        dataset = load_dataset(source_cfg["hf_name"], split=source_cfg.get("split", "train"))
    elif source_type == "csv":
        df = pd.read_csv(source_cfg["path"])
        dataset = Dataset.from_pandas(df, preserve_index=False)
    else:
        raise ValueError(f"Unknown source type: {source_type!r} (expected 'hf' or 'csv')")

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")

    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    dataset = dataset.map(
        lambda ex: {"text": "" if ex["text"] is None else str(ex["text"]).replace("\n", " ").strip()}
    )
    dataset = dataset.filter(lambda ex: len(ex["text"]) > 0)

    return dataset


def load_all_sources(sources_cfg: list[dict[str, Any]]) -> list[Dataset]:
    return [load_source(cfg) for cfg in sources_cfg]
