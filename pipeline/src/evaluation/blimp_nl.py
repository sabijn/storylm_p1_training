import os
import re

import pandas as pd
import torch
from datasets import get_dataset_config_names, load_dataset


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


@torch.no_grad()
def sentence_logprob(model, tokenizer, sentence: str, device, normalize_by_length: bool = True) -> float:
    """Log-probability of `sentence` under a causal LM (standard BLiMP minimal-pair scoring)."""
    input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)

    logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    total_logprob = token_log_probs.sum().item()
    num_pred_tokens = shift_labels.numel()

    if normalize_by_length and num_pred_tokens > 0:
        return total_logprob / num_pred_tokens
    return total_logprob


def evaluate_blimp_subset(model, tokenizer, subset_name, subset_data, device, normalize_by_length=True):
    rows = []
    correct = 0
    total = 0

    for idx, example in enumerate(subset_data):
        good_sent = example["sentence_good"]
        bad_sent = example["sentence_bad"]
        phenomenon = example.get("linguistic_phenomenon", subset_name)

        good_score = sentence_logprob(model, tokenizer, good_sent, device, normalize_by_length)
        bad_score = sentence_logprob(model, tokenizer, bad_sent, device, normalize_by_length)

        is_correct = good_score > bad_score
        correct += int(is_correct)
        total += 1

        rows.append({
            "subset": subset_name,
            "index": idx,
            "linguistic_phenomenon": phenomenon,
            "sentence_good": good_sent,
            "sentence_bad": bad_sent,
            "good_score": good_score,
            "bad_score": bad_score,
            "margin": good_score - bad_score,
            "correct": int(is_correct),
        })

    accuracy = correct / total if total > 0 else float("nan")
    return rows, {"subset": subset_name, "n_examples": total, "n_correct": correct, "accuracy": accuracy}


def evaluate_blimp_nl(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str = "model",
    normalize_by_length: bool = True,
) -> pd.DataFrame:
    """Evaluate a causal LM on all juletxara/blimp-nl subsets and write per-subset + summary CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    subset_names = get_dataset_config_names("juletxara/blimp-nl")
    summary_rows = []

    for subset_name in subset_names:
        print(f"Evaluating BLiMP-NL subset: {subset_name}")
        subset_data = load_dataset("juletxara/blimp-nl", subset_name, split="train")

        detailed_rows, summary = evaluate_blimp_subset(
            model, tokenizer, subset_name, subset_data, device, normalize_by_length
        )
        summary_rows.append(summary)

        subset_csv_path = os.path.join(output_dir, f"{model_name}_{sanitize_filename(subset_name)}.csv")
        pd.DataFrame(detailed_rows).to_csv(subset_csv_path, index=False)
        print(
            f"  -> accuracy = {summary['accuracy']:.4f} "
            f"({summary['n_correct']}/{summary['n_examples']})"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("subset")
    summary_csv_path = os.path.join(output_dir, f"{model_name}_blimp_nl_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    macro_acc = summary_df["accuracy"].mean()
    with open(os.path.join(output_dir, f"{model_name}_blimp_nl_macro_accuracy.txt"), "w") as f:
        f.write(f"{macro_acc:.6f}\n")

    print(f"\nSaved BLiMP-NL summary to {summary_csv_path}")
    print(f"Macro-average accuracy: {macro_acc:.4f}")

    return summary_df
