import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_from_disk
from transformers import AutoTokenizer


def count_tokens(tokenizer, dataset, text_column: str, batch_size: int = 1000) -> int:
    texts = dataset[text_column]
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_ids = tokenizer(batch, add_special_tokens=False)["input_ids"]
        total_tokens += sum(len(ids) for ids in batch_ids)
    return total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Report document/word/token counts for one SoNaR-500 category, from the "
        "already-extracted local dataset (requires scripts/prepare_sonar.py to have been run "
        "once). Category codes are listed in scripts/prepare_sonar.py's CATEGORY_NAMES, e.g. "
        "WR-P-P-B = books, WS-U-E-A = autocues."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/scratch-shared/sperdijk/storylm_p1_training/data/sonar_extracted"),
        help="Path to the extracted SoNaR dataset (scripts/prepare_sonar.py's output_dir).",
    )
    parser.add_argument("--category", required=True, help="SoNaR category code, e.g. WR-P-P-B.")
    parser.add_argument(
        "--tokenizer-path",
        action="append",
        dest="tokenizer_paths",
        required=True,
        help="Path to a tokenizer to count with. Pass multiple times to compare tokenizers.",
    )
    args = parser.parse_args()

    print(f"Loading extracted SoNaR dataset from {args.dataset_dir}...")
    dataset = load_from_disk(str(args.dataset_dir))
    print(dataset)

    category_dataset = dataset.filter(lambda ex: ex["category"] == args.category)
    n_documents = len(category_dataset)
    print(f"\n{args.category}: {n_documents:,} documents")
    if n_documents == 0:
        return

    n_words = sum(len(t.split()) for t in category_dataset["text"])
    print(f"{args.category}: {n_words:,} whitespace words")

    for tokenizer_path in args.tokenizer_paths:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        n_tokens = count_tokens(tokenizer, category_dataset, "text")
        print(f"{args.category} [{tokenizer_path}]: {n_tokens:,} tokens (vocab_size={tokenizer.vocab_size})")


if __name__ == "__main__":
    main()
