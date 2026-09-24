"""SQuAD-NL: fine-tuned extractive question answering (span extraction).

Source: direct GitHub release URLs (wietsedv/NLP-NL, tag squad-nl-v1.0) - the DUMB
benchmark's own upstream source, v2.0 (includes unanswerable questions), publicly
downloadable with no auth needed.

Uses GPT2ForQuestionAnswering. Note: the pretrained models in this pipeline have a 256-token
context window, which is small for SQuAD-style passages - contexts are truncated to fit
(keeping the question intact), with no sliding-window/stride handling, so answers that only
occur past the truncation point are treated as unanswerable during training. This is a
real, expected limitation of fine-tuning a small-context model on this task, not a bug.
"""

import json
import re
import string
import urllib.request
from collections import Counter

from datasets import Dataset

from ..finetune import FinetuneConfig, free_model

_URL_TEMPLATE = "https://github.com/wietsedv/NLP-NL/raw/squad-nl-v1.0/SQuAD-NL/nl/{split}-{version}.json"


def _load_split(split: str, version: str = "v2.0") -> list[dict]:
    url = _URL_TEMPLATE.format(split=split, version=version)
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
    return data["data"]


def _preprocess_example(example: dict, tokenizer, max_length: int, max_question_length: int = 64) -> dict:
    question_ids = tokenizer(
        example["question"] + tokenizer.eos_token, add_special_tokens=False, truncation=True, max_length=max_question_length
    )["input_ids"]
    max_context_length = max(max_length - len(question_ids), 1)

    context_enc = tokenizer(
        example["context"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_context_length,
        return_offsets_mapping=True,
    )
    context_ids = context_enc["input_ids"]
    offsets = context_enc["offset_mapping"]
    q_len = len(question_ids)

    answers = example["answers"]
    start_position = end_position = 0
    if len(answers["text"]) > 0:
        answer_start_char = answers["answer_start"][0]
        answer_end_char = answer_start_char + len(answers["text"][0])
        start_token = end_token = None
        for i, (s, e) in enumerate(offsets):
            if s <= answer_start_char < e:
                start_token = q_len + i
            if s < answer_end_char <= e:
                end_token = q_len + i
        if start_token is not None and end_token is not None:
            start_position, end_position = start_token, end_token

    input_ids = question_ids + context_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "start_positions": start_position,
        "end_positions": end_position,
    }


def _build_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    processed = [_preprocess_example(row, tokenizer, max_length) for row in rows]
    return Dataset.from_list(processed)


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(de|het|een)\b", " ", text)
    return " ".join(text.split())


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _best_metric_over_references(prediction: str, references: list[str], metric_fn) -> float:
    if not references:
        references = [""]
    return max(metric_fn(prediction, ref) for ref in references)


def evaluate_squadnl(model_dir: str, tokenizer, device: str, output_dir: str, cfg: FinetuneConfig = FinetuneConfig()) -> dict:
    from transformers import DataCollatorWithPadding, GPT2ForQuestionAnswering, Trainer, TrainingArguments

    max_length = 256
    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(model_dir)
        max_length = getattr(model_config, "n_positions", None) or getattr(model_config, "max_position_embeddings", max_length)
    except Exception:
        pass

    train_rows = _load_split("train")
    dev_rows = _load_split("dev")
    test_rows = _load_split("test")

    train_ds = _build_dataset(train_rows, tokenizer, max_length)
    dev_ds = _build_dataset(dev_rows, tokenizer, max_length)
    test_ds = _build_dataset(test_rows, tokenizer, max_length)

    model = GPT2ForQuestionAnswering.from_pretrained(model_dir, pad_token_id=tokenizer.pad_token_id)
    model.to(device)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        seed=cfg.seed,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        logging_steps=50,
        disable_tqdm=True,
        label_names=["start_positions", "end_positions"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    predictions = trainer.predict(test_ds)
    start_logits, end_logits = predictions.predictions

    em_total = f1_total = 0.0
    for i, row in enumerate(test_rows):
        pred_start = int(start_logits[i].argmax())
        pred_end = int(end_logits[i].argmax())
        input_ids = test_ds[i]["input_ids"]

        if pred_start == 0 or pred_end < pred_start:
            predicted_text = ""
        else:
            predicted_text = tokenizer.decode(input_ids[pred_start : pred_end + 1], skip_special_tokens=True)

        references = row["answers"]["text"]
        em_total += _best_metric_over_references(predicted_text, references, _exact_match)
        f1_total += _best_metric_over_references(predicted_text, references, _f1_score)

    free_model(trainer.model, trainer)

    n_examples = len(test_rows)
    return {
        "exact_match": em_total / n_examples if n_examples else float("nan"),
        "f1": f1_total / n_examples if n_examples else float("nan"),
        "n_examples": n_examples,
    }
