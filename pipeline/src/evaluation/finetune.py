"""Shared fine-tune-then-evaluate machinery for the classification-style extra eval tasks
(SIB-200, SICK-NL). Each call loads a *fresh* classification head on top of the backbone
weights at `model_dir` (the checkpoint being evaluated is never modified), fine-tunes it,
evaluates, then frees the copy - so tasks can be run back-to-back without accumulating GPU
memory."""

import gc
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, GPT2ForSequenceClassification, Trainer, TrainingArguments


@dataclass
class FinetuneConfig:
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42


def free_model(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_classifier_trainer(
    model_dir: str,
    tokenizer,
    device: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    num_labels: int,
    output_dir: str,
    cfg: FinetuneConfig = FinetuneConfig(),
) -> Trainer:
    """Construct a GPT2ForSequenceClassification (backbone from `model_dir`, a fresh
    randomly-initialized head) and a Trainer ready to fine-tune it. `train_dataset` /
    `eval_dataset` must already have `input_ids`/`attention_mask` (tokenized) and an
    integer `label` column."""
    model = GPT2ForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels, pad_token_id=tokenizer.pad_token_id
    )
    model.to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        seed=cfg.seed,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        logging_steps=50,
        disable_tqdm=True,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )


def finetune_and_evaluate_classifier(
    model_dir: str,
    tokenizer,
    device: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    num_labels: int,
    output_dir: str,
    cfg: FinetuneConfig = FinetuneConfig(),
) -> dict:
    """Fine-tune then evaluate plain accuracy - the common case (SIB-200, SICK-NL). For
    tasks needing custom post-processing of predictions (e.g. COPA-NL's pairwise
    reformulation), use `build_classifier_trainer` directly instead."""
    trainer = build_classifier_trainer(
        model_dir, tokenizer, device, train_dataset, eval_dataset, num_labels, output_dir, cfg
    )
    trainer.train()
    metrics = trainer.evaluate()
    free_model(trainer.model, trainer)
    return {"accuracy": metrics["eval_accuracy"], "eval_loss": metrics["eval_loss"]}
