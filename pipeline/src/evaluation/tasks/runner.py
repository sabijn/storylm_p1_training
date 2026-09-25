"""Shared dispatcher for the extra eval tasks registry (see registry.py). Used by both
05_evaluate_model.py (a single post-training eval pass) and TokenMilestoneCallback (the same
full task suite, repeated at every token milestone during a continued-pretraining run)."""

import json
from pathlib import Path

from ..finetune import FinetuneConfig
from .registry import TASK_REGISTRY

_FINETUNE_CFG_FIELDS = set(FinetuneConfig.__dataclass_fields__)


def run_extra_tasks(tasks_cfg: dict, model, model_dir: str, tokenizer, device, output_dir: str) -> dict:
    """Run every task in `tasks_cfg` (a {task_name: kwargs} dict, same shape as an
    eval_*.yaml/model_continued.yaml `tasks:` block) against the live `model`. `model_dir`
    must be an on-disk checkpoint whose weights match `model` - finetune-kind tasks load a
    fresh head from there (the checkpoint itself is never modified); zero-shot tasks ignore
    it. Returns {task_name: result_dict}. Every task gets its own
    <output_dir>/<task_name>/results.json, regardless of kind, and that's also the directory
    handed to any task that writes its own richer output (currently only blimp_nl, for its
    per-subset breakdown).
    """
    tasks_cfg = tasks_cfg or {}
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
