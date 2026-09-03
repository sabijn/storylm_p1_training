import json
import math
import shutil
from pathlib import Path

import wandb
from transformers import TrainerCallback

from .blimp_nl import evaluate_blimp_nl_macro_accuracy


class TokenMilestoneCallback(TrainerCallback):
    """Drives token-anchor-based BLiMP-NL analysis for a single continuous training run.

    Milestones are given in millions of tokens seen and converted to target training steps
    via `tokens_per_step` (tokens consumed per optimizer step). At each milestone step, this
    forces an off-schedule checkpoint (independent of `save_steps`/`save_total_limit`), copies
    it to a permanent `milestone-<M>M` directory (never pruned), evaluates BLiMP-NL
    macro-accuracy on the live model, logs it to W&B (keyed by both step and tokens seen), and
    writes a small metadata json next to the checkpoint.
    """

    def __init__(
        self,
        tokenizer,
        milestones_millions: list[int],
        tokens_per_step: int,
        output_dir: str,
        normalize_by_length: bool = True,
    ):
        self.tokenizer = tokenizer
        self.tokens_per_step = tokens_per_step
        self.output_dir = Path(output_dir)
        self.normalize_by_length = normalize_by_length

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

        was_training = model.training
        model.eval()
        try:
            macro_accuracy = evaluate_blimp_nl_macro_accuracy(
                model,
                self.tokenizer,
                next(model.parameters()).device,
                normalize_by_length=self.normalize_by_length,
            )
        finally:
            model.train(was_training)

        print(
            f"[TokenMilestone] step {step} (~{tokens_seen / 1e6:.1f}M tokens): "
            f"blimp macro accuracy = {macro_accuracy:.4f}"
        )
        wandb.log(
            {"blimp_nl/macro_accuracy": macro_accuracy, "train/tokens_seen": tokens_seen},
            step=step,
        )

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
                        "blimp_nl_macro_accuracy": macro_accuracy,
                    },
                    f,
                    indent=2,
                )
            print(f"[TokenMilestone] saved milestone-{M}M -> {milestone_dir}")

        return control
