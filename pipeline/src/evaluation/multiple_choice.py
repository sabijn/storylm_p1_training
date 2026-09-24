"""Zero-shot multiple-choice scoring: given a shared prompt and several candidate
continuations, score each continuation's length-normalized log-probability under the model and
predict the argmax. Used for HellaSwag-NL and COPA-NL (raw-text continuations), Belebele-NL
(constructed prompt + letter continuations), and StoryCloze-NL (letter-choice continuations
against a pre-rendered instruction prompt)."""

import torch


@torch.no_grad()
def score_continuation(model, tokenizer, prompt: str, continuation: str, device, normalize_by_length: bool = True) -> float:
    """Log-probability of `continuation` given `prompt`, i.e. only the continuation's own
    tokens are scored (the shared prompt prefix is conditioned on, not scored itself)."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]

    # Guard against tokenization not being a clean prefix extension (rare, e.g. merges
    # across the prompt/continuation boundary) - still take the tail as an approximation.
    n_prompt_tokens = len(prompt_ids)
    if n_prompt_tokens >= len(full_ids):
        n_prompt_tokens = max(len(full_ids) - 1, 0)

    # Long prompts (e.g. Belebele passages) can exceed the model's context window - drop
    # from the start of the prompt so the continuation itself is never truncated away.
    max_len = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    if max_len is not None and len(full_ids) > max_len:
        n_drop = len(full_ids) - max_len
        full_ids = full_ids[n_drop:]
        n_prompt_tokens = max(n_prompt_tokens - n_drop, 0)

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)

    # token_log_probs[i] is log P(full_ids[i+1] | full_ids[:i+1]); continuation tokens
    # start at full_ids[n_prompt_tokens], i.e. token_log_probs[n_prompt_tokens - 1:].
    continuation_log_probs = token_log_probs[max(n_prompt_tokens - 1, 0):]
    if continuation_log_probs.numel() == 0:
        return float("-inf")

    total = continuation_log_probs.sum().item()
    if normalize_by_length:
        return total / continuation_log_probs.numel()
    return total


def score_choices(model, tokenizer, prompt: str, choices: list[str], device, normalize_by_length: bool = True) -> list[float]:
    return [score_continuation(model, tokenizer, prompt, choice, device, normalize_by_length) for choice in choices]


def evaluate_multiple_choice(
    model,
    tokenizer,
    device,
    examples: list[dict],
    normalize_by_length: bool = True,
) -> dict:
    """`examples` is a list of {"prompt": str, "choices": list[str], "label": int}. Returns
    {"accuracy": float, "n_examples": int, "n_correct": int}."""
    n_correct = 0
    for example in examples:
        scores = score_choices(model, tokenizer, example["prompt"], example["choices"], device, normalize_by_length)
        predicted = max(range(len(scores)), key=lambda i: scores[i])
        n_correct += int(predicted == example["label"])

    n_examples = len(examples)
    accuracy = n_correct / n_examples if n_examples > 0 else float("nan")
    return {"accuracy": accuracy, "n_examples": n_examples, "n_correct": n_correct}
