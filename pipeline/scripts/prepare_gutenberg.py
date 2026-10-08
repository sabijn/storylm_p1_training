import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset, load_from_disk

from src.common.config import load_yaml

# Anchors on the "*** START OF THIS/THE PROJECT GUTENBERG EBOOK ... ***" marker (always
# present, case varies) plus an optional following "Produced by ..." credit line (wording
# varies a lot across books, e.g. "... the Distributed Proofreaders Team" vs "... the PG
# Distributed Proofreaders Team" - matched loosely rather than pinned to exact phrasing).
# Everything up to and including the match - the license block and Title/Author/Release
# Date metadata before it too - is the header. Documents that don't match are left as-is
# (their boilerplate preamble stays in the text) rather than dropped.
GUTENBERG_HEADER_END_RE = re.compile(
    r"\*\*\*[ \t]*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*[ \t]*\n+"
    r"(?:Produced by[^\n]*\n+)?",
    re.IGNORECASE,
)


def strip_gutenberg_header(text: str) -> str:
    match = GUTENBERG_HEADER_END_RE.search(text)
    return text[match.end() :].lstrip("\n") if match else text


def main():
    parser = argparse.ArgumentParser(
        description="Download one language split of manu/project_gutenberg and strip each "
        "book's Project Gutenberg boilerplate header. Writes one HF dataset with columns: "
        "id, text. Run this once; category/sampling choices for training happen later, "
        "cheaply, via the `local` source type in data_base.yaml/data_continued.yaml."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/gutenberg_extract.yaml.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = cfg["output_dir"]

    if Path(output_dir).exists():
        try:
            dataset = load_from_disk(output_dir)
            print(f"Found existing cleaned dataset at {output_dir} ({len(dataset):,} documents), skipping.")
            return
        except Exception as e:
            print(f"Found {output_dir} but couldn't load it ({e}) - re-running.")

    print(f"Downloading {cfg['data_files']}...")
    dataset = load_dataset("parquet", data_files=cfg["data_files"], split="train")
    print(f"  {len(dataset):,} documents")

    n_matched = sum(1 for text in dataset["text"] if GUTENBERG_HEADER_END_RE.search(text))
    print(f"Header pattern matched {n_matched:,} / {len(dataset):,} documents (non-matches are kept as-is).")

    print("Stripping headers...")
    dataset = dataset.map(lambda ex: {"text": strip_gutenberg_header(ex["text"])})

    print(f"Saving {len(dataset):,} documents to {output_dir}...")
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
