import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wandb
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.common.config import load_yaml
from src.common.wandb_utils import init_wandb
from src.data.packing import pack_dataset
from src.data.prepare import load_prepared
from src.evaluation.perplexity import evaluate_perplexity
from src.models.build_model import build_model, get_block_size


def main():
    parser = argparse.ArgumentParser(description="Train a small causal LM from scratch with the HF Trainer.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a model_base_*.yaml config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.config.parent / cfg["data"]["config"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])

    model = build_model(cfg["architecture"], cfg["model"], vocab_size=tokenizer.vocab_size)
    block_size = get_block_size(model)
    print(f"Built {cfg['architecture']} model with {model.num_parameters():,} parameters (block size {block_size})")

    print(f"Loading prepared dataset from {data_cfg['paths']['output_dir']}...")
    dataset_dict = load_prepared(data_cfg["paths"]["output_dir"])

    print("Tokenizing and packing train split...")
    train_dataset = pack_dataset(dataset_dict["train"], tokenizer, block_size)
    print("Tokenizing and packing dev split...")
    dev_dataset = pack_dataset(dataset_dict["dev"], tokenizer, block_size)
    print(f"  train: {len(train_dataset):,} blocks | dev: {len(dev_dataset):,} blocks")

    init_wandb(cfg["wandb"], run_config={"architecture": cfg["architecture"], **cfg["model"]})

    training_args = TrainingArguments(
        **cfg["training"],
        report_to=["wandb"],
        run_name=cfg["wandb"].get("run_name"),
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    final_dir = Path(training_args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved base model to {final_dir}")

    print("Evaluating on dev split...")
    metrics = evaluate_perplexity(trainer, eval_dataset=dev_dataset)
    print(f"Dev loss: {metrics['loss']:.4f} | perplexity: {metrics['perplexity']:.2f}")
    trainer.log({"eval/perplexity": metrics["perplexity"]})

    wandb.finish()


if __name__ == "__main__":
    main()
