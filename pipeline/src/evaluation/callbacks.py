import csv
import json
import math
import shutil
from pathlib import Path

import wandb
from transformers import TrainerCallback

from .blimp_nl import evaluate_blimp_nl_by_subset


class TokenMilestoneCallback(TrainerCallback):
    """Drives token-anchor-based BLiMP-NL analysis for a single continuous training run.

    Milestones are given in millions of tokens seen and converted to target training steps
    via `tokens_per_step` (tokens consumed per optimizer step). At each milestone step, this
    forces an off-schedule checkpoint (independent of `save_steps`/`save_total_limit`), copies
    it to a permanent `milestone-<M>M` directory (never pruned), evaluates BLiMP-NL per-subset
    accuracy (plus the derived macro-average) on the live model, logs all of it to W&B (keyed
    by both step and tokens seen), appends it to a cumulative
    `<output_dir>/blimp_nl_per_subset.csv`, and writes a metadata json next to the checkpoint.
    """

    SUBSET_CSV_FIELDS = ["milestone_millions", "tokens_seen", "global_step", "subset", "n_examples", "n_correct", "accuracy"]

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
        self.subset_csv_path = self.output_dir / "blimp_nl_per_subset.csv"

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
            subset_summaries = evaluate_blimp_nl_by_subset(
                model,
                self.tokenizer,
                next(model.parameters()).device,
                normalize_by_length=self.normalize_by_length,
            )
        finally:
            model.train(was_training)

        macro_accuracy = sum(s["accuracy"] for s in subset_summaries) / len(subset_summaries)

        print(
            f"[TokenMilestone] step {step} (~{tokens_seen / 1e6:.1f}M tokens): "
            f"blimp macro accuracy = {macro_accuracy:.4f}"
        )
        wandb_payload = {"blimp_nl/macro_accuracy": macro_accuracy, "train/tokens_seen": tokens_seen}
        for summary in subset_summaries:
            wandb_payload[f"blimp_nl/subsets/{summary['subset']}"] = summary["accuracy"]
        wandb.log(wandb_payload, step=step)

        self._append_subset_csv(due, step, tokens_seen, subset_summaries)

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
                        "blimp_nl_subsets": subset_summaries,
                    },
                    f,
                    indent=2,
                )
            print(f"[TokenMilestone] saved milestone-{M}M -> {milestone_dir}")

        return control

    def _append_subset_csv(self, milestones: list[int], step: int, tokens_seen: int, subset_summaries: list[dict]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.subset_csv_path.exists()
        with open(self.subset_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUBSET_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for M in milestones:
                for summary in subset_summaries:
                    writer.writerow(
                        {
                            "milestone_millions": M,
                            "tokens_seen": tokens_seen,
                            "global_step": step,
                            "subset": summary["subset"],
                            "n_examples": summary["n_examples"],
                            "n_correct": summary["n_correct"],
                            "accuracy": summary["accuracy"],
                        }
                    )
