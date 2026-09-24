"""XCOMPS-NL: zero-shot conceptual-property minimal pairs.

Source: raw GitHub JSON (LinyangHe/XCOMPS, no HF dataset mirror). Structured exactly like a
BLiMP minimal pair (acceptable_sent vs unacceptable_sent), so `a_preferred_rate` from the
shared pairwise scorer directly is accuracy.
"""

import json
import urllib.request

from ..pairwise import evaluate_pairwise

_DATA_URL = "https://raw.githubusercontent.com/LinyangHe/XCOMPS/main/data/comps_nl.json"


def load_xcomps_nl() -> list[dict]:
    with urllib.request.urlopen(_DATA_URL) as response:
        data = json.load(response)
    return [
        {
            "sentence_a": row["acceptable_sent"],
            "sentence_b": row["unacceptable_sent"],
            "negative_sample_type": row.get("negative_sample_type"),
        }
        for row in data.values()
    ]


def evaluate_xcomps_nl(model, tokenizer, device, normalize_by_length: bool = True) -> dict:
    pairs = load_xcomps_nl()
    result = evaluate_pairwise(model, tokenizer, device, pairs, normalize_by_length)
    return {
        "accuracy": result["a_preferred_rate"],
        "n_examples": result["n_examples"],
        "n_correct": result["n_a_preferred"],
    }
