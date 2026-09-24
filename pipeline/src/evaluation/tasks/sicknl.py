"""SICK-NL: fine-tuned natural language inference (3-way: entailment/neutral/contradiction).

Source: maximedb/sick_nl (train/validation/test splits).
"""

from datasets import load_dataset

from ..finetune import FinetuneConfig, build_classifier_trainer, free_model

_LABEL2ID = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}


def _load_split(split: str):
    return load_dataset("maximedb/sick_nl", split=split)


def _tokenize(dataset, tokenizer):
    def encode(batch):
        pairs = [f"{a}{tokenizer.eos_token}{b}" for a, b in zip(batch["sentence_A"], batch["sentence_B"])]
        enc = tokenizer(pairs, truncation=True)
        enc["label"] = [_LABEL2ID[label] for label in batch["entailment_label"]]
        return enc

    return dataset.map(encode, batched=True, remove_columns=dataset.column_names)


def evaluate_sicknl(model_dir: str, tokenizer, device: str, output_dir: str, cfg: FinetuneConfig = FinetuneConfig()) -> dict:
    train_raw = _load_split("train")
    val_raw = _load_split("validation")
    test_raw = _load_split("test")

    train_ds = _tokenize(train_raw, tokenizer)
    val_ds = _tokenize(val_raw, tokenizer)
    test_ds = _tokenize(test_raw, tokenizer)

    trainer = build_classifier_trainer(
        model_dir, tokenizer, device, train_ds, val_ds, num_labels=len(_LABEL2ID), output_dir=output_dir, cfg=cfg
    )
    trainer.train()
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    free_model(trainer.model, trainer)

    return {
        "accuracy": test_metrics["test_accuracy"],
        "val_accuracy": val_metrics["eval_accuracy"],
    }
