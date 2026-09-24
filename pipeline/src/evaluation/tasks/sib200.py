"""SIB-200 (Dutch): fine-tuned topic classification (7-way NLU).

Source: Davlan/sib200, config nld_Latn.
"""

from datasets import load_dataset

from ..finetune import FinetuneConfig, build_classifier_trainer, free_model


def _load_split(split: str):
    return load_dataset("Davlan/sib200", "nld_Latn", split=split)


def _tokenize(dataset, tokenizer, label2id: dict[str, int]):
    def encode(batch):
        enc = tokenizer(batch["text"], truncation=True)
        enc["label"] = [label2id[c] for c in batch["category"]]
        return enc

    return dataset.map(encode, batched=True, remove_columns=dataset.column_names)


def evaluate_sib200(model_dir: str, tokenizer, device: str, output_dir: str, cfg: FinetuneConfig = FinetuneConfig()) -> dict:
    train_raw = _load_split("train")
    val_raw = _load_split("validation")
    test_raw = _load_split("test")

    # Union of categories across all splits, not just train, in case a category happens to
    # be absent from one split (e.g. when testing against a small slice).
    all_categories = set(train_raw["category"]) | set(val_raw["category"]) | set(test_raw["category"])
    label2id = {label: i for i, label in enumerate(sorted(all_categories))}

    train_ds = _tokenize(train_raw, tokenizer, label2id)
    val_ds = _tokenize(val_raw, tokenizer, label2id)
    test_ds = _tokenize(test_raw, tokenizer, label2id)

    trainer = build_classifier_trainer(
        model_dir, tokenizer, device, train_ds, val_ds, num_labels=len(label2id), output_dir=output_dir, cfg=cfg
    )
    trainer.train()
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    free_model(trainer.model, trainer)

    return {
        "accuracy": test_metrics["test_accuracy"],
        "val_accuracy": val_metrics["eval_accuracy"],
        "n_labels": len(label2id),
    }
