import json
import math
import shutil
from pathlib import Path

import wandb
from transformers import TrainerCallback

from .tasks.runner import run_extra_tasks


class TokenMilestoneCallback(TrainerCallback):
    """Drives token-anchor-based analysis for a single continuous training run.

    Milestones are given in millions of tokens seen and converted to target training steps
    via `tokens_per_step` (tokens consumed per optimizer step). At each milestone step, this
    forces an off-schedule checkpoint (independent of `save_steps`/`save_total_limit`), runs
    every task in `tasks_cfg` (same shape/registry as an eval_*.yaml `tasks:` block -
    zero-shot tasks run directly against the live model, finetune tasks train+evaluate a
    throwaway head off the just-saved checkpoint, writing into `checkpoint_dir/tasks/`), logs
    every task's numeric results to W&B (keyed by both step and tokens seen), then copies the
    checkpoint - task outputs included - to a permanent `milestone-<M>M` directory (never
    pruned) alongside a metadata json.
    """

    def __init__(
        self,
        tokenizer,
        milestones_millions: list[int],
        tokens_per_step: int,
        output_dir: str,
        tasks_cfg: dict | None = None,
    ):
        self.tokenizer = tokenizer
        self.tokens_per_step = tokens_per_step
        self.output_dir = Path(output_dir)
        self.tasks_cfg = tasks_cfg or {}

        self.milestone_steps = {M: math.ceil(M * 1_000_000 / tokens_per_step) for M in milestones_millions}
        self._due_this_step: list[int] = []

    def on_step_end(self, args, state, control, **kwargs):
        due = [M for M, step in self.milestone_steps.items() if step == state.global_step]
        if due:
            self._due_this_step = due
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self._due_this_step:
            return control
        due, self._due_this_step = self._due_this_step, []

        step = state.global_step
        tokens_seen = step * self.tokens_per_step
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        model = kwargs["model"]
        device = next(model.parameters()).device

        was_training = model.training
        model.eval()
        try:
            task_results = run_extra_tasks(
                self.tasks_cfg, model, str(checkpoint_dir), self.tokenizer, device, str(checkpoint_dir / "tasks")
            )
        finally:
            model.train(was_training)

        print(f"[TokenMilestone] step {step} (~{tokens_seen / 1e6:.1f}M tokens): {task_results}")

        wandb_payload = {"train/tokens_seen": tokens_seen}
        for task_name, result in task_results.items():
            for metric_name, value in result.items():
                if isinstance(value, (int, float)):
                    wandb_payload[f"{task_name}/{metric_name}"] = value
        wandb.log(wandb_payload, step=step)

        for M in due:
            milestone_dir = self.output_dir / f"milestone-{M}M"
            shutil.rmtree(milestone_dir, ignore_errors=True)
            shutil.copytree(checkpoint_dir, milestone_dir)
            self.tokenizer.save_pretrained(milestone_dir)
            with open(milestone_dir / "milestone_meta.json", "w") as f:
                json.dump(
                    {
                        "milestone_millions": M,
                        "tokens_seen": tokens_seen,
                        "global_step": step,
                        "task_results": task_results,
                    },
                    f,
                    indent=2,
                )
            print(f"[TokenMilestone] saved milestone-{M}M -> {milestone_dir}")

        return control
