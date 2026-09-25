from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer


def load_source(source_cfg: dict[str, Any]) -> Dataset:
    """Load one configured dataset source and normalize it to a single "text" column."""
    source_type = source_cfg["type"]
    text_column = source_cfg["text_column"]

    if source_type == "hf":
        dataset = load_dataset(source_cfg["hf_name"], split=source_cfg.get("split", "train"))
    elif source_type == "csv":
        df = pd.read_csv(source_cfg["path"])
        dataset = Dataset.from_pandas(df, preserve_index=False)
    elif source_type == "local":
        # A pre-existing local `save_to_disk` dataset (e.g. from scripts/prepare_sonar.py),
        # optionally filtered to specific categories and/or randomly subsampled.
        dataset = load_from_disk(source_cfg["path"])

        categories = source_cfg.get("categories")
        if categories:
            categories = set(categories)
            dataset = dataset.filter(lambda ex: ex["category"] in categories)

        sample_words = source_cfg.get("sample_words")
        sample_tokens = source_cfg.get("sample_tokens")
        sample_fraction = source_cfg.get("sample_fraction")
        sample_size = source_cfg.get("sample_size")
        if sum(x is not None for x in (sample_words, sample_tokens, sample_fraction, sample_size)) > 1:
            raise ValueError("Set at most one of sample_words, sample_tokens, sample_fraction, sample_size.")

        if sample_words is not None:
            # Whitespace word count as a fast, tokenizer-agnostic proxy for a token budget
            # (multiply a target token count by a tokenizer's known fertility, e.g. ~1.2, to
            # get sample_words) - prefer sample_tokens when exact token counts matter.
            dataset = dataset.shuffle(seed=source_cfg.get("seed", 42))
            dataset = _select_by_word_budget(dataset, text_column, sample_words)
        elif sample_tokens is not None:
            tokenizer_path = source_cfg.get("tokenizer_path")
            if not tokenizer_path:
                raise ValueError("sample_tokens requires tokenizer_path (the tokenizer to count tokens with).")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            dataset = dataset.shuffle(seed=source_cfg.get("seed", 42))
            dataset = _select_by_token_budget(dataset, text_column, tokenizer, sample_tokens)
        elif sample_fraction is not None or sample_size is not None:
            dataset = dataset.shuffle(seed=source_cfg.get("seed", 42))
            n = sample_size if sample_size is not None else round(len(dataset) * sample_fraction)
            dataset = dataset.select(range(min(n, len(dataset))))
    else:
        raise ValueError(f"Unknown source type: {source_type!r} (expected 'hf', 'csv', or 'local')")

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")

    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    dataset = dataset.map(
        lambda ex: {"text": "" if ex["text"] is None else str(ex["text"]).replace("\n", " ").strip()}
    )
    dataset = dataset.filter(lambda ex: len(ex["text"]) > 0)

    return dataset


def _select_by_word_budget(dataset: Dataset, text_column: str, target_words: int, batch_size: int = 1000) -> Dataset:
    """Select a prefix of an already-shuffled dataset whose cumulative whitespace word
    count reaches `target_words`, without counting words past that point - so this only
    costs work proportional to the sample actually selected, not the full dataset."""
    total_words = 0
    n_selected = 0
    for start in range(0, len(dataset), batch_size):
        batch_texts = dataset[start : start + batch_size][text_column]
        for text in batch_texts:
            total_words += len((text or "").split())
            n_selected += 1
            if total_words >= target_words:
                break
        if total_words >= target_words:
            break
    return dataset.select(range(n_selected))


def _select_by_token_budget(
    dataset: Dataset, text_column: str, tokenizer, target_tokens: int, batch_size: int = 1000
) -> Dataset:
    """Select a prefix of an already-shuffled dataset whose cumulative tokenized length (via
    `tokenizer`, no special tokens) reaches `target_tokens`, tokenizing only as far as needed
    - same early-stopping strategy as `_select_by_word_budget`, but exact rather than a
    whitespace-word proxy."""
    total_tokens = 0
    n_selected = 0
    for start in range(0, len(dataset), batch_size):
        batch_texts = dataset[start : start + batch_size][text_column]
        batch_ids = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        for ids in batch_ids:
            total_tokens += len(ids)
            n_selected += 1
            if total_tokens >= target_tokens:
                break
        if total_tokens >= target_tokens:
            break
    return dataset.select(range(n_selected))


def load_all_sources(sources_cfg: list[dict[str, Any]]) -> list[Dataset]:
    return [load_source(cfg) for cfg in sources_cfg]
