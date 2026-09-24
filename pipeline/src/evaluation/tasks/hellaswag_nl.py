"""HellaSwag-NL: zero-shot 4-way commonsense continuation.

Source: jon-tow/okapi_hellaswag, config "nl" - a ChatGPT machine translation (Lai et al.
2023) of the English HellaSwag, CC BY-NC 4.0 (non-commercial). Only a "validation" split has
real labels (test labels are withheld, matching the original English HellaSwag convention).
"""

from datasets import load_dataset

from ..multiple_choice import evaluate_multiple_choice


def load_hellaswag_nl(split: str = "validation") -> list[dict]:
    dataset = load_dataset("jon-tow/okapi_hellaswag", "nl", split=split)
    return [
        {"prompt": row["ctx"], "choices": [f" {ending}" for ending in row["endings"]], "label": int(row["label"])}
        for row in dataset
    ]


def evaluate_hellaswag_nl(model, tokenizer, device, split: str = "validation", normalize_by_length: bool = True) -> dict:
    examples = load_hellaswag_nl(split)
    return evaluate_multiple_choice(model, tokenizer, device, examples, normalize_by_length)
