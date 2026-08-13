import pandas as pd
import torch

from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

def read_chiscor(config, bos_token, eos_token):
    df = pd.read_csv(config['data_file'], index_col=0)
    stories = df["story_raw"].dropna().astype(str).tolist()

    return Dataset.from_dict({
        "text": [f"{bos_token}{story.strip()}{eos_token}" for story in stories]
    })


def add_special_tokens(example, bos_token, eos_token):
    return {"text": f"{bos_token}{example['text'].strip()}{eos_token}"}


def tokenize_function(examples, tokenizer, bos_token, eos_token):
    return {
        "input_ids": [
            tokenizer.encode(text, allowed_special={bos_token, eos_token})
            for text in examples["text"]
        ]
    }


def group_texts(examples, block_size):
    """
    HF-style packing:
    concatenate tokenized documents and split into fixed chunks.

    We use block_size + 1 so each chunk can be split into:
      x = [:block_size]
      y = [1:block_size+1]
    """
    concatenated = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)

    chunk_len = block_size + 1
    total_length = (len(concatenated) // chunk_len) * chunk_len

    if total_length == 0:
        return {"input_ids": []}

    concatenated = concatenated[:total_length]

    return {
        "input_ids": [
            concatenated[i:i + chunk_len]
            for i in range(0, total_length, chunk_len)
        ]
    }

def read_hf_dataset(hf_name, bos_token, eos_token, column="text"):
    dataset = load_dataset(hf_name, split="train")

    if column != "text":
        dataset = dataset.rename_column("story", "text")
    dataset = dataset.map(
        lambda ex: add_special_tokens(ex, bos_token, eos_token)
    )

    return dataset

def prepare_datasets(tokenizer, config, bos_token, eos_token):
    datasets = config['wandb']['dataset'].split('+')
    loaded_datasets = []

    for d in datasets:
        if d == 'chiscor':
            loaded_datasets.append(read_chiscor(config['paths'], bos_token, eos_token))
        elif d == 'sftw':
            loaded_datasets.append(read_hf_dataset("Sabijn/StoriesfortheWin", bos_token, eos_token, "story"))
        elif d == 'babybabel_without_books':
            loaded_datasets.append(read_hf_dataset("BabyLM-community/babylm-nld", bos_token, eos_token))
        else:
            raise NotImplementedError

    train_splits = []
    val_splits = []

    for d in loaded_datasets:
        splits = d.train_test_split(test_size=0.1, seed=42)
        train_splits.append(splits['train']), val_splits.append(splits['test'])

    # combine
    train_ds = concatenate_datasets(
        train_splits).shuffle(seed=42)

    val_ds = concatenate_datasets(
        val_splits).shuffle(seed=42)

    # tokenize
    train_ds = train_ds.map(
        lambda batch: tokenize_function(batch, tokenizer, bos_token, eos_token),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train dataset",
    )

    val_ds = val_ds.map(
        lambda batch: tokenize_function(batch, tokenizer, bos_token, eos_token),
        batched=True,
        remove_columns=val_ds.column_names,
        desc="Tokenizing validation dataset",
    )

    # pack into fixed-length chunks
    block_size = config["model"]["block_size"]

    train_ds = train_ds.map(
        lambda batch: group_texts(batch, block_size),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Packing train dataset",
    )

    val_ds = val_ds.map(
        lambda batch: group_texts(batch, block_size),
        batched=True,
        remove_columns=val_ds.column_names,
        desc="Packing validation dataset",
    )

    # make PyTorch-ready
    train_ds.set_format(type="torch", columns=["input_ids"])
    val_ds.set_format(type="torch", columns=["input_ids"])

    return train_ds, val_ds


def collate_lm(batch):
    # batch is a list of {"input_ids": tensor(block_size + 1)}
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    return input_ids


def create_dataloader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_lm,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )