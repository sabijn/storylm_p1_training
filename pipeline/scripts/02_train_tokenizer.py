import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sentencepiece as spm
from transformers import LlamaTokenizerFast
from transformers.convert_slow_tokenizer import SentencePieceExtractor

from src.common.config import load_yaml
from src.data.prepare import load_prepared


def write_corpus(dataset, output_file: str) -> str:
    os.makedirs(Path(output_file).parent, exist_ok=True)
    written = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
                written += 1
    print(f"Wrote {written:,} lines to {output_file}")
    return output_file


def train_sentencepiece(corpus_file: str, model_prefix: str, tok_cfg: dict) -> str:
    os.makedirs(Path(model_prefix).parent, exist_ok=True)
    special = tok_cfg["special_tokens"]
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=tok_cfg["vocab_size"],
        model_type=tok_cfg.get("model_type", "bpe"),
        character_coverage=tok_cfg.get("character_coverage", 0.9995),
        input_sentence_size=tok_cfg.get("input_sentence_size", 0),
        shuffle_input_sentence=True,
        # fixed ids AND matching piece strings, so the HF wrapper's bos/eos/pad/unk
        # tokens always resolve to real vocab entries instead of being silently
        # registered as new added tokens past the end of the model's embedding table
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=special["pad"],
        unk_piece=special["unk"],
        bos_piece=special["bos"],
        eos_piece=special["eos"],
    )
    return f"{model_prefix}.model"


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer and wrap it as a HF fast tokenizer."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/tokenizer.yaml.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tok_cfg = cfg["tokenizer"]
    special = tok_cfg["special_tokens"]

    print(f"Loading prepared dataset from {cfg['data']['prepared_dataset_dir']}...")
    dataset_dict = load_prepared(cfg["data"]["prepared_dataset_dir"])
    split_dataset = dataset_dict[cfg["data"].get("split", "train")]

    corpus_file = write_corpus(split_dataset, cfg["paths"]["corpus_file"])

    output_dir = cfg["paths"]["output_dir"]
    model_prefix = str(Path(output_dir) / "sentencepiece")
    print(f"Training {tok_cfg['vocab_size']}-vocab SentencePiece {tok_cfg.get('model_type', 'bpe')} tokenizer...")
    model_file = train_sentencepiece(corpus_file, model_prefix, tok_cfg)

    # LlamaTokenizerFast is used here purely as a generic SentencePiece -> HF
    # fast-tokenizer wrapper; it does not tie the tokenizer to the Llama model
    # architecture (any AutoModelForCausalLM can use it).
    extracted = SentencePieceExtractor(model_file).extract(model_type=None)
    tokenizer = LlamaTokenizerFast(
        vocab=extracted["vocab"],
        merges=extracted["merges"],
        pad_token=special["pad"],
        unk_token=special["unk"],
        bos_token=special["bos"],
        eos_token=special["eos"],
    )
    if tokenizer.vocab_size != tok_cfg["vocab_size"]:
        raise RuntimeError(
            f"LlamaTokenizerFast wrapping produced vocab_size={tokenizer.vocab_size}, "
            f"expected {tok_cfg['vocab_size']} (from {model_file})."
        )

    tokenizer.save_pretrained(output_dir)
    print(f"Saved HF-compatible tokenizer to {output_dir} (vocab size {tokenizer.vocab_size})")


if __name__ == "__main__":
    main()
