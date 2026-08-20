import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.config import load_yaml
from src.data.prepare import load_prepared

SEP = "-" * 80


def show_data_samples(dataset, n: int) -> None:
    print(f"\n{SEP}\n1. RAW DATA SAMPLES\n{SEP}")
    for i in range(min(n, len(dataset))):
        text = dataset[i]["text"]
        preview = text if len(text) <= 500 else text[:500] + " [...]"
        print(f"\n--- sample {i} ({len(text)} chars) ---\n{preview}")


def show_tokenized_samples(dataset, tokenizer, n: int) -> None:
    print(f"\n{SEP}\n2. TOKENIZED DATA SAMPLES\n{SEP}")
    for i in range(min(n, len(dataset))):
        test = dataset[i]['text']
        print(test if len(test) <= 500 else test[:500] + " [...]")
        text = f"{tokenizer.bos_token}{dataset[i]['text']}{tokenizer.eos_token}"
        print("text", text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        preview_ids = ids[:50]
        tokens = tokenizer.convert_ids_to_tokens(preview_ids)
        print(f"\n--- sample {i} ({len(ids)} tokens, showing first {len(preview_ids)}) ---")
        print(f"ids:    {preview_ids}")
        print(f"tokens: {tokens}")


def show_generations(model, tokenizer, device, dataset, n: int, max_new_tokens: int) -> None:
    print(f"\n{SEP}\n3. GENERATIONS\n{SEP}")
    model.eval()
    for i in range(min(n, len(dataset))):
        text = dataset[i]["text"]
        prompt_words = text.split()[:15]
        prompt = " ".join(prompt_words)
        input_text = f"{tokenizer.bos_token}{prompt}"
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n--- generation {i} ---")
        print(f"prompt:    {prompt!r}")
        print(f"generated: {generated!r}")


def main():
    parser = argparse.ArgumentParser(description="Inspect data samples, tokenization, and generations for a trained model.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "configs" / "eval_base.yaml",
                         help="Path to a configs/eval_*.yaml (gives model path + data config).")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to sample from (default: config's split, or 'test').")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples/generations to show.")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Number of tokens to generate per sample.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.config.parent / cfg["data"]["config"])
    split_name = args.split or cfg["data"].get("split", "test")

    model_dir = cfg["model"]["path"]
    tokenizer_path = cfg["model"]["tokenizer_path"]
    print(f"Model dir:     {model_dir}")
    print(f"Tokenizer:     {tokenizer_path}")
    print(f"Data dir:      {data_cfg['paths']['output_dir']} (split={split_name})")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(tokenizer)
    print("Loading dataset...")
    dataset_dict = load_prepared(data_cfg["paths"]["output_dir"])
    dataset = dataset_dict[split_name]

    show_data_samples(dataset, args.n_samples)
    show_tokenized_samples(dataset, tokenizer, args.n_samples)

    print(f"\nLoading model from {model_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    print(f"Device: {device} | params: {model.num_parameters():,}")

    show_generations(model, tokenizer, device, dataset, args.n_samples, args.max_new_tokens)

    print(f"\n{SEP}\nDone.\n{SEP}")


if __name__ == "__main__":
    main()
