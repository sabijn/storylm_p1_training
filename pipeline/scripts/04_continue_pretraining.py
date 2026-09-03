import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.common.config import load_yaml
from src.common.wandb_utils import init_wandb
from src.data.packing import pack_dataset
from src.data.prepare import load_prepared
from src.evaluation.callbacks import TokenMilestoneCallback
from src.evaluation.perplexity import evaluate_perplexity
from src.models.build_model import get_block_size


def main():
    parser = argparse.ArgumentParser(description="Continue pretraining a saved base model on a new dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/model_continued.yaml.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.config.parent / cfg["data"]["config"])
    base_cfg = cfg["base_model"]

    print(f"Loading base model from {base_cfg['path']}...")
    model = AutoModelForCausalLM.from_pretrained(base_cfg["path"])
    tokenizer = AutoTokenizer.from_pretrained(base_cfg["tokenizer_path"])
    block_size = get_block_size(model)

    print(f"Loading prepared dataset from {data_cfg['paths']['output_dir']}...")
    dataset_dict = load_prepared(data_cfg["paths"]["output_dir"])

    print("Tokenizing and packing train split...")
    train_dataset = pack_dataset(dataset_dict["train"], tokenizer, block_size)
    print("Tokenizing and packing dev split...")
    dev_dataset = pack_dataset(dataset_dict["dev"], tokenizer, block_size)
    print(f"  train: {len(train_dataset):,} blocks | dev: {len(dev_dataset):,} blocks")

    init_wandb(cfg["wandb"], run_config={"base_model_path": base_cfg["path"]})

    training_args = TrainingArguments(
        **cfg["training"],
        report_to=["wandb"],
        run_name=cfg["wandb"].get("run_name"),
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    callbacks = []
    milestone_callback = None
    blimp_cfg = cfg.get("blimp", {})
    if blimp_cfg.get("token_milestones_millions"):
        tokens_per_step = (
            block_size
            * training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size
        )
        print(
            f"Token milestones (M tokens): {blimp_cfg['token_milestones_millions']} "
            f"(~{tokens_per_step:,} tokens/step)"
        )
        milestone_callback = TokenMilestoneCallback(
            tokenizer=tokenizer,
            milestones_millions=blimp_cfg["token_milestones_millions"],
            tokens_per_step=tokens_per_step,
            output_dir=training_args.output_dir,
            normalize_by_length=blimp_cfg.get("normalize_by_length", True),
        )
        callbacks.append(milestone_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Continuing pretraining...")
    trainer.train()

    if milestone_callback is not None:
        keep_steps = set(milestone_callback.milestone_steps.values())
        if trainer.state.best_model_checkpoint:
            keep_steps.add(int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1]))
        for checkpoint_dir in Path(training_args.output_dir).glob("checkpoint-*"):
            try:
                step = int(checkpoint_dir.name.split("-")[-1])
            except ValueError:
                continue
            if step not in keep_steps:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print(f"Pruned non-milestone/non-best checkpoints, kept steps: {sorted(keep_steps)}")

    final_dir = Path(training_args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved continued-pretraining model to {final_dir}")

    print("Evaluating on dev split...")
    metrics = evaluate_perplexity(trainer, eval_dataset=dev_dataset)
    print(f"Dev loss: {metrics['loss']:.4f} | perplexity: {metrics['perplexity']:.2f}")
    trainer.log({"eval/perplexity": metrics["perplexity"]})

    wandb.finish()


if __name__ == "__main__":
    main()
