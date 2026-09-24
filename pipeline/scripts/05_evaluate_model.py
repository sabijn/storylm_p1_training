import argparse
import json
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
from src.evaluation.finetune import FinetuneConfig
from src.evaluation.perplexity import evaluate_perplexity
from src.evaluation.tasks.registry import TASK_REGISTRY
from src.models.build_model import get_block_size

_FINETUNE_CFG_FIELDS = set(FinetuneConfig.__dataclass_fields__)


def run_extra_tasks(cfg, model, model_dir, tokenizer, device, output_dir) -> dict:
    """Run the tasks listed under the config's `tasks:` block (see eval_base.yaml for the
    full list of available task names and what each option means - this includes BLiMP-NL,
    as task "blimp_nl"). Returns {task_name: result_dict}.
    """
    tasks_cfg = cfg.get("tasks") or {}
    results = {}
    model_name = Path(model_dir).parent.name or "model"
    for task_name, task_kwargs in tasks_cfg.items():
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_name!r}. Available: {sorted(TASK_REGISTRY)}")
        spec = TASK_REGISTRY[task_name]
        task_kwargs = dict(task_kwargs or {})
        task_output_dir = Path(output_dir) / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning extra eval task: {task_name} ({spec.kind})...")
        if spec.kind == "zero_shot":
            result = spec.fn(model, tokenizer, device, **task_kwargs)
        elif spec.kind == "zero_shot_output":
            result = spec.fn(model, tokenizer, device, str(task_output_dir), model_name=model_name, **task_kwargs)
        else:  # finetune
            finetune_kwargs = {k: task_kwargs.pop(k) for k in list(task_kwargs) if k in _FINETUNE_CFG_FIELDS}
            finetune_cfg = FinetuneConfig(**finetune_kwargs)
            result = spec.fn(model_dir, tokenizer, device, str(task_output_dir), cfg=finetune_cfg, **task_kwargs)

        print(f"  {task_name}: {result}")
        with open(task_output_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        results[task_name] = result
    return results


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

    log_payload = {
        f"eval/{split_name}_loss": metrics["loss"],
        f"eval/{split_name}_perplexity": metrics["perplexity"],
    }

    task_results = run_extra_tasks(cfg, model, model_dir, tokenizer, device, output_dir)
    for task_name, result in task_results.items():
        for metric_name, value in result.items():
            if isinstance(value, (int, float)):
                log_payload[f"{task_name}/{metric_name}"] = value
    if task_results:
        results_path = Path(output_dir) / "extra_tasks_results.json"
        with open(results_path, "w") as f:
            json.dump(task_results, f, indent=2)
        print(f"\nExtra task results saved to {results_path}")

    wandb.log(log_payload)
    wandb.finish()

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
