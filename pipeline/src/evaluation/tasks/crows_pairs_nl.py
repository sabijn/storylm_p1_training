"""Dutch CrowS-Pairs: zero-shot stereotype-bias probe.

Source: raw GitHub CSV (jerryspan/Dutch-CrowS-Pairs, no HF dataset mirror; Strazda & Spanakis,
"Dutch CrowS-Pairs", arxiv 2507.16442). Unlike the other pairwise/accuracy tasks, this is a
BIAS metric, not a correctness metric: `stereotype_score` is the fraction of pairs where the
model prefers the more-stereotypical sentence. 50% is the unbiased ideal - neither 0% nor
100% is "good".
"""

import pandas as pd

from ..pairwise import evaluate_pairwise

_DATA_URL = "https://raw.githubusercontent.com/jerryspan/Dutch-CrowS-Pairs/main/datasets/crows_dutch.csv"


def load_crows_pairs_nl() -> list[dict]:
    # latin-1: the source CSV isn't valid UTF-8 (has cp1252-style smart-quote bytes).
    df = pd.read_csv(_DATA_URL, encoding="latin-1")
    return [
        {
            "sentence_a": row["sent_more"],
            "sentence_b": row["sent_less"],
            "bias_type": row["bias_type"],
        }
        for _, row in df.iterrows()
    ]


def evaluate_crows_pairs_nl(model, tokenizer, device, normalize_by_length: bool = True) -> dict:
    pairs = load_crows_pairs_nl()
    result = evaluate_pairwise(model, tokenizer, device, pairs, normalize_by_length)
    return {
        "stereotype_score": result["a_preferred_rate"],
        "n_examples": result["n_examples"],
        "n_more_stereotyped_preferred": result["n_a_preferred"],
    }
