"""StoryCloze-NL: zero-shot narrative-coherence continuation (2-way).

Source: aialt/MuBench (Han et al., "MuBench: Assessment of Multilingual Capabilities of
Large Language Models", arxiv 2506.19468), config StoryClozeDataset_lighteval_nl - the raw
story text as the prompt and the two actual candidate ending sentences as choices (no
instruction-wrapper template, no letter-token indirection), scored the same way as
HellaSwag-NL/COPA-NL. Deliberately not local_template_nl/en_template_nl: those require the
model to follow an "answer with A or B" instruction format, a skill these small,
non-instruction-tuned models don't have - the signal would be mostly noise.
"""

from datasets import load_dataset

from ..multiple_choice import evaluate_multiple_choice


def load_storycloze_nl(split: str = "test") -> list[dict]:
    dataset = load_dataset("aialt/MuBench", "StoryClozeDataset_lighteval_nl", split=split)
    return [
        # `prompt` has no trailing space and `choices` have no leading space (they're meant
        # to continue the story as one sentence), so a space is inserted between them here.
        {"prompt": row["prompt"], "choices": [f" {choice}" for choice in row["choices"]], "label": int(row["label"])}
        for row in dataset
    ]


def evaluate_storycloze_nl(model, tokenizer, device, split: str = "test", normalize_by_length: bool = True) -> dict:
    examples = load_storycloze_nl(split)
    return evaluate_multiple_choice(model, tokenizer, device, examples, normalize_by_length)
