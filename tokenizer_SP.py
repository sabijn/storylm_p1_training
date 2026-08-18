import os

import sentencepiece as spm
from datasets import load_dataset


def download_dataset(dataset_name: str = "Sabijn/StoriesfortheWin", split: str = "train"):
    """Download a HuggingFace dataset and return it."""
    print(f"Downloading dataset '{dataset_name}', split={split})...")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Downloaded {len(dataset):,} examples.")
    return dataset


def write_corpus(dataset, output_file: str = "dutch_corpus.txt", text_column: str = "story") -> str:
    """Write dataset to a plain text file with one document per line."""
    print(f"Writing corpus to '{output_file}'...")
    written = 0
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example[text_column].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
                written += 1
            else:
                skipped += 1

    size_mb = os.path.getsize(output_file) / 1e6
    print(f"Wrote {written:,} lines ({size_mb:.1f} MB). Skipped {skipped:,} empty examples.")
    return output_file


def train_tokenizer(
    corpus_file: str,
    model_prefix: str = "tokenizer_8k_nl",
    vocab_size: int = 8000,
    character_coverage: float = 0.9995,
    input_sentence_size: int = 5_000_000,
):
    """Train a SentencePiece BPE tokenizer on the given corpus file."""
    print(f"Training {vocab_size}-vocab BPE tokenizer on '{corpus_file}'...")
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,  # 0.9995 handles Dutch accented chars (é, ë, ij)
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    print(f"Tokenizer saved: {model_file}, {vocab_file}")
    
    return model_file


if __name__ == "__main__":
    # Config
    DATASET_NAME = "Sabijn/StoriesfortheWin"
    DATASET_SPLIT = "train"
    CORPUS_FILE = "/local/perdijks/datasets/storiesFTW.txt"
    MODEL_PREFIX = "/local/perdijks/training_gpt/tokenizer/tokenizer_8k_nl"
    VOCAB_SIZE = 8000

    # Pipeline
    dataset = download_dataset(DATASET_NAME, DATASET_SPLIT)
    corpus_file = write_corpus(dataset, CORPUS_FILE)
    model_file = train_tokenizer(corpus_file, MODEL_PREFIX, VOCAB_SIZE)