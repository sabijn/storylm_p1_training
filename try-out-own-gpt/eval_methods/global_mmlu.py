import torch
import torch.nn.functional as F

import argparse
from pathlib import Path
import os
import re
import csv
import yaml
from dataclasses import dataclass
from sklearn.metrics import f1_score
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model import GPTLanguageModel
from utils import load_tokenizer


@dataclass
class GlobalMMLUExample:
    question: str
    choices: list[str]
    answer: str
    subject: str
    split: str | None = None
    category: str | None = None
    example_id: str | None = None


def load_model(vocab_size, device, config):
    model = GPTLanguageModel(vocab_size, **config["model"])
    state_dict = torch.load(config["paths"]["output_model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def encode_text(tokenizer, text):
    if hasattr(tokenizer, "__call__"):
        try:
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
            if isinstance(encoded, dict) and "input_ids" in encoded:
                return encoded["input_ids"]
        except TypeError:
            pass

    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            return ids.long()
        return torch.tensor([ids], dtype=torch.long)

    raise ValueError("Tokenizer must support either __call__(..., return_tensors='pt') or .encode(text)")


@torch.no_grad()
def continuation_logprob(model, tokenizer, device, prompt: str, continuation: str, normalize: bool = False) -> float:
    prompt_ids = encode_text(tokenizer, prompt).to(device)
    full_ids = encode_text(tokenizer, prompt + continuation).to(device)

    block_size = model.config.block_size if hasattr(model, "config") else None
    if block_size is None:
        block_size = model.block_size if hasattr(model, "block_size") else None
    if block_size is None:
        block_size = 256  # fallback if needed

    prompt_len = prompt_ids.size(1)
    full_len = full_ids.size(1)

    if full_len <= prompt_len:
        raise ValueError(f"Continuation produced no extra tokens: {continuation!r}")

    continuation_len = full_len - prompt_len

    # Truncate from the left if sequence is too long
    if full_len > block_size:
        full_ids = full_ids[:, -block_size:]
        full_len = full_ids.size(1)
        prompt_len = full_len - continuation_len

        if prompt_len < 0:
            raise ValueError(
                f"Continuation itself is longer than block_size={block_size}. "
                f"continuation_len={continuation_len}"
            )

    outputs = model(full_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    log_probs = F.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    num_tokens = full_len - prompt_len

    for pos in range(prompt_len, full_len):
        token_id = full_ids[0, pos]
        total_logprob += log_probs[0, pos - 1, token_id].item()

    if normalize:
        return total_logprob / num_tokens
    return total_logprob


def build_prompt(example: GlobalMMLUExample) -> str:
    letters = ["A", "B", "C", "D"]
    prompt_lines = [
        "Beantwoord de volgende meerkeuzevraag.",
        "Geef alleen het juiste antwoord: A, B, C of D.",
        "",
        f"Vraag: {example.question}",
    ]

    for letter, choice in zip(letters, example.choices):
        prompt_lines.append(f"{letter}. {choice}")

    prompt_lines.append("")
    prompt_lines.append("Antwoord:")
    return "\n".join(prompt_lines)


@torch.no_grad()
def predict_example(model, tokenizer, device, example: GlobalMMLUExample, normalize: bool = False):
    prompt = build_prompt(example)
    options = [" A", " B", " C", " D"]

    scores = {
        label: continuation_logprob(model, tokenizer, device, prompt, cont, normalize=normalize)
        for label, cont in zip(["A", "B", "C", "D"], options)
    }

    pred_label = max(scores, key=scores.get)
    return pred_label, scores, prompt


def get_field(example, candidates, default=None):
    for key in candidates:
        if key in example:
            return example[key]
    return default


def normalize_choices(example_dict):
    direct_choice_keys = ["option_a", "option_b", "option_c", "option_d"]
    if all(k in example_dict for k in direct_choice_keys):
        return [example_dict[k] for k in direct_choice_keys]

    if "choices" in example_dict:
        choices = example_dict["choices"]
        if isinstance(choices, dict):
            for value_key in ["text", "choices", "options"]:
                if value_key in choices and len(choices[value_key]) >= 4:
                    return list(choices[value_key])[:4]
        elif isinstance(choices, (list, tuple)) and len(choices) >= 4:
            return list(choices)[:4]

    alt_keys = ["A", "B", "C", "D"]
    if all(k in example_dict for k in alt_keys):
        return [example_dict[k] for k in alt_keys]

    raise KeyError(f"Could not find four answer choices in example keys: {list(example_dict.keys())}")


def normalize_answer(answer_value) -> str:
    if answer_value is None:
        raise ValueError("Answer field is missing.")

    answer = str(answer_value).strip()
    if answer in {"A", "B", "C", "D"}:
        return answer

    # Sometimes labels are 0..3
    if answer in {"0", "1", "2", "3"}:
        return ["A", "B", "C", "D"][int(answer)]

    upper = answer.upper()
    if upper in {"A", "B", "C", "D"}:
        return upper

    raise ValueError(f"Unsupported answer label: {answer_value!r}")


def load_global_mmlu_nl(preferred_dataset_name: str = "CohereLabs/Global-MMLU"):
    last_error = None
    for dataset_name in [preferred_dataset_name, "CohereForAI/Global-MMLU"]:
        try:
            ds = load_dataset(dataset_name, "nl")
            return ds, dataset_name
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to load Global-MMLU nl. Last error: {last_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=Path, default=Path("/local/perdijks/training_gpt/global_mmlu_results"))
    parser.add_argument("--dataset_name", type=str, default="CohereLabs/Global-MMLU")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--normalize_logprob", action="store_true")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = Path(config["paths"]["output_model_path"]).stem

    EOS = config["tokens"]["eos_token"]
    BOS = config["tokens"]["bos_token"]

    special_tokens = {
        BOS: 100264,
        EOS: 100257,
    }

    tokenizer = load_tokenizer(special_tokens)
    model = load_model(tokenizer.n_vocab, device, config)

    results_dir = Path(args.result_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict, resolved_dataset_name = load_global_mmlu_nl(args.dataset_name)
    if args.split not in dataset_dict:
        raise ValueError(f"Split {args.split!r} not found. Available splits: {list(dataset_dict.keys())}")

    split_ds = dataset_dict[args.split]
    print(f"Loaded dataset: {resolved_dataset_name} / nl / {args.split} with {len(split_ds)} examples")

    examples = []
    for row in split_ds:
        question = get_field(row, ["question", "input", "prompt"])
        choices = normalize_choices(row)
        answer = normalize_answer(get_field(row, ["answer", "target", "label", "gold"]))
        subject = get_field(row, ["subject", "topic", "subtask"], default="unknown")
        category = get_field(row, ["category", "domain"], default=None)
        example_id = get_field(row, ["example_id", "id"], default=None)

        if question is None:
            raise ValueError(f"Could not find question field in keys: {list(row.keys())}")

        examples.append(
            GlobalMMLUExample(
                question=question,
                choices=choices,
                answer=answer,
                subject=subject,
                split=args.split,
                category=category,
                example_id=example_id,
            )
        )

    all_rows = []
    correct = 0

    for i, example in enumerate(examples):
        pred_label, scores, prompt = predict_example(
            model,
            tokenizer,
            device,
            example,
            normalize=args.normalize_logprob,
        )

        is_correct = int(pred_label == example.answer)
        correct += is_correct

        all_rows.append(
            {
                "idx": i,
                "example_id": example.example_id,
                "subject": example.subject,
                "category": example.category,
                "question": example.question,
                "choice_A": example.choices[0],
                "choice_B": example.choices[1],
                "choice_C": example.choices[2],
                "choice_D": example.choices[3],
                "gold_label": example.answer,
                "pred_label": pred_label,
                "correct": is_correct,
                "score_A": scores["A"],
                "score_B": scores["B"],
                "score_C": scores["C"],
                "score_D": scores["D"],
                "prompt": prompt,
            }
        )

        if (i + 1) % 100 == 0 or (i + 1) == len(examples):
            acc = correct / (i + 1)
            print(f"[{i + 1}/{len(examples)}] accuracy={acc:.4f}")

    golds = [row["gold_label"] for row in all_rows]
    preds = [row["pred_label"] for row in all_rows]

    accuracy = correct / len(examples) if examples else 0.0
    weighted_f1 = f1_score(golds, preds, average="weighted") if examples else 0.0
    macro_f1 = f1_score(golds, preds, average="macro") if examples else 0.0

    safe_model_name = sanitize_filename(model_name)
    split_name = sanitize_filename(args.split)
    pred_file = results_dir / f"global_mmlu_nl_{split_name}_{safe_model_name}_predictions.csv"
    summary_file = results_dir / f"global_mmlu_nl_{split_name}_{safe_model_name}_summary.txt"

    with open(pred_file, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "example_id",
                "subject",
                "category",
                "question",
                "choice_A",
                "choice_B",
                "choice_C",
                "choice_D",
                "gold_label",
                "pred_label",
                "correct",
                "score_A",
                "score_B",
                "score_C",
                "score_D",
                "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    with open(summary_file, "w", encoding="utf8") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"dataset_name: {resolved_dataset_name}\n")
        f.write("config_name: nl\n")
        f.write(f"split: {args.split}\n")
        f.write(f"num_examples: {len(examples)}\n")
        f.write(f"accuracy: {accuracy:.6f}\n")
        f.write(f"weighted_f1: {weighted_f1:.6f}\n")
        f.write(f"macro_f1: {macro_f1:.6f}\n")
        f.write(f"normalize_logprob: {args.normalize_logprob}\n")
        f.write(f"predictions_file: {pred_file}\n")

    print("Finished evaluation.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Predictions saved to: {pred_file}")
    print(f"Summary saved to: {summary_file}")
