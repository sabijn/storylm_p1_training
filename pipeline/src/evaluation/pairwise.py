"""Shared zero-shot pairwise (minimal-pair) scoring. Used for XCOMPS-NL (acceptable vs unacceptable - "a_preferred_rate"
is accuracy) and Dutch CrowS-Pairs (more- vs less-stereotypical - "a_preferred_rate" is the
stereotype score, where 50% is the unbiased ideal, not 100%)."""

from .blimp_nl import sentence_logprob


def evaluate_pairwise(model, tokenizer, device, pairs: list[dict], normalize_by_length: bool = True) -> dict:
    """`pairs` is a list of {"sentence_a": str, "sentence_b": str, ...}. Returns the
    fraction of pairs where sentence_a scores higher than sentence_b."""
    n_a_preferred = 0
    for pair in pairs:
        score_a = sentence_logprob(model, tokenizer, pair["sentence_a"], device, normalize_by_length)
        score_b = sentence_logprob(model, tokenizer, pair["sentence_b"], device, normalize_by_length)
        n_a_preferred += int(score_a > score_b)

    n_examples = len(pairs)
    rate = n_a_preferred / n_examples if n_examples > 0 else float("nan")
    return {"a_preferred_rate": rate, "n_examples": n_examples, "n_a_preferred": n_a_preferred}
