from typing import Any

from datasets import Dataset


def tokenize_dataset(dataset: Dataset, tokenizer, text_column: str = "text") -> Dataset:
    def _tokenize(examples: dict[str, list[str]]) -> dict[str, Any]:
        texts = [f"{tokenizer.bos_token}{t}{tokenizer.eos_token}" for t in examples[text_column]]
        return {"input_ids": tokenizer(texts, add_special_tokens=False)["input_ids"]}

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def group_texts(examples: dict[str, list[list[int]]], block_size: int) -> dict[str, list[list[int]]]:
    """Concatenate tokenized docs and split into fixed-length block_size chunks.

    A hand-rolled model needs block_size + 1 tokens per example to manually
    build shifted (x, y) pairs; HF causal LMs derive labels from input_ids and
    shift internally, so chunks here are exactly block_size long.
    """
    concatenated: list[int] = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)

    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": []}

    concatenated = concatenated[:total_length]

    return {
        "input_ids": [
            concatenated[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
    }


def pack_dataset(dataset: Dataset, tokenizer, block_size: int, text_column: str = "text") -> Dataset:
    tokenized = tokenize_dataset(dataset, tokenizer, text_column=text_column)
    return tokenized.map(
        lambda batch: group_texts(batch, block_size),
        batched=True,
        remove_columns=tokenized.column_names,
        desc="Packing into fixed-length blocks",
    )
