"""Belebele-NL: zero-shot 4-way reading comprehension.

Source: facebook/belebele, config nld_Latn. Test-only by design (no train/validation split
exists for this dataset) - it can only ever be evaluated zero-shot. Unlike HellaSwag-NL, the
answer options aren't natural continuations of the passage, so we build our own prompt
template and score each option letter as the continuation.
"""

from datasets import load_dataset

from ..multiple_choice import evaluate_multiple_choice

_LETTERS = ["A", "B", "C", "D"]


def _build_prompt(row: dict) -> str:
    options = "\n".join(f"{letter}) {row[f'mc_answer{i + 1}']}" for i, letter in enumerate(_LETTERS))
    return f"{row['flores_passage']}\n\nVraag: {row['question']}\n{options}\nAntwoord:"


def load_belebele_nl(split: str = "test") -> list[dict]:
    dataset = load_dataset("facebook/belebele", "nld_Latn", split=split)
    return [
        {
            "prompt": _build_prompt(row),
            "choices": [f" {letter}" for letter in _LETTERS],
            "label": int(row["correct_answer_num"]) - 1,
        }
        for row in dataset
    ]


def evaluate_belebele_nl(model, tokenizer, device, split: str = "test", normalize_by_length: bool = True) -> dict:
    examples = load_belebele_nl(split)
    return evaluate_multiple_choice(model, tokenizer, device, examples, normalize_by_length)
