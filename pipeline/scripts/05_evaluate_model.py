import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.common.config import load_yaml
from src.common.wandb_utils import init_wandb
from src.data.packing import pack_dataset
from src.data.prepare import load_prepared
from src.evaluation.blimp_nl import evaluate_blimp_nl
from src.evaluation.perplexity import evaluate_perplexity
from src.models.build_model import get_block_size


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model: val loss/perplexity + BLiMP-NL.")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/eval_*.yaml.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.config.parent / cfg["data"]["config"])

    model_dir = cfg["model"]["path"]
    print(f"Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    block_size = get_block_size(model)
    split_name = cfg["data"].get("split", "test")

    print(f"Loading prepared dataset from {data_cfg['paths']['output_dir']} (split={split_name})...")
    dataset_dict = load_prepared(data_cfg["paths"]["output_dir"])
    eval_dataset = pack_dataset(dataset_dict[split_name], tokenizer, block_size)
    print(f"  {split_name}: {len(eval_dataset):,} blocks")

    output_dir = cfg["paths"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    init_wandb(cfg["wandb"], run_config={"model_dir": model_dir, "split": split_name})

    # Trainer is used purely as an evaluation driver here (no actual training),
    # the same loss/perplexity path the training scripts already use.
    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=cfg.get("eval_batch_size", 16),
        report_to=[],
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(model=model, args=eval_args, eval_dataset=eval_dataset, data_collator=data_collator)

    print("Computing loss / perplexity...")
    metrics = evaluate_perplexity(trainer)
    print(f"{split_name} loss: {metrics['loss']:.4f} | perplexity: {metrics['perplexity']:.2f}")

    print("Running BLiMP-NL evaluation...")
    model_name = Path(model_dir).parent.name or "model"
    blimp_cfg = cfg.get("blimp", {})
    summary_df = evaluate_blimp_nl(
        model,
        tokenizer,
        device,
        output_dir,
        model_name=model_name,
        normalize_by_length=blimp_cfg.get("normalize_by_length", True),
    )

    log_payload = {
        f"eval/{split_name}_loss": metrics["loss"],
        f"eval/{split_name}_perplexity": metrics["perplexity"],
        "blimp_nl/macro_accuracy": summary_df["accuracy"].mean(),
    }
    for _, row in summary_df.iterrows():
        log_payload[f"blimp_nl/{row['subset']}_accuracy"] = row["accuracy"]
    wandb.log(log_payload)
    wandb.finish()

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
