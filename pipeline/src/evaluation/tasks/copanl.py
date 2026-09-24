"""COPA-NL: zero-shot causal commonsense reasoning (2-way choice).

Source: direct GitHub release archive (wietsedv/NLP-NL, tag copa-nl-v1.0) - the DUMB
benchmark's own upstream source for this task, publicly downloadable with no auth needed.

Scored zero-shot via the same continuation-logprob approach as StoryCloze-NL/HellaSwag-NL:
for each premise, the two alternatives are scored as continuations of "{premise} {connective}"
and the higher-scoring one is picked - the standard way COPA is evaluated zero-shot.
"""

import io
import json
import tarfile
import urllib.request

from ..multiple_choice import evaluate_multiple_choice

_ARCHIVE_URL = "https://github.com/wietsedv/NLP-NL/archive/refs/tags/copa-nl-v1.0.tar.gz"
_CONNECTIVES = {"cause": "omdat", "effect": "daardoor"}


def _load_rows(split: str) -> list[dict]:
    with urllib.request.urlopen(_ARCHIVE_URL) as response:
        archive_bytes = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(f"COPA-NL/{split}.jsonl"))
        fileobj = tf.extractfile(member)
        return [json.loads(line) for line in fileobj.read().decode("utf-8").splitlines() if line.strip()]


def load_copanl(split: str = "test") -> list[dict]:
    examples = []
    for row in _load_rows(split):
        connective = _CONNECTIVES[row["question"]]
        examples.append(
            {
                "prompt": f"{row['premise']} {connective} ",
                "choices": [row["choice1"], row["choice2"]],
                "label": int(row["label"]),
            }
        )
    return examples


def evaluate_copanl(model, tokenizer, device, split: str = "test", normalize_by_length: bool = True) -> dict:
    examples = load_copanl(split)
    return evaluate_multiple_choice(model, tokenizer, device, examples, normalize_by_length)
