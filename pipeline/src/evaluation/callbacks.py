import wandb
from transformers import TrainerCallback

from .blimp_nl import evaluate_blimp_nl_macro_accuracy


class BlimpNLCallback(TrainerCallback):
    """Runs a lightweight BLiMP-NL pass every `eval_steps` training steps and logs the
    bootstrap-resampled macro-average accuracy to the active W&B run as mean/lower/upper
    (mean +/- std), so the three can be added to a single wandb line-plot panel and read as
    one mean-with-band graph (see evaluate_blimp_nl_macro_accuracy)."""

    def __init__(
        self,
        tokenizer,
        eval_steps: int,
        normalize_by_length: bool = True,
        n_bootstrap: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.normalize_by_length = normalize_by_length
        self.n_bootstrap = n_bootstrap

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_steps <= 0 or state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return control

        model = kwargs["model"]
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            macro_accuracy_mean, macro_accuracy_std = evaluate_blimp_nl_macro_accuracy(
                model,
                self.tokenizer,
                device,
                normalize_by_length=self.normalize_by_length,
                n_bootstrap=self.n_bootstrap,
            )
        finally:
            model.train(was_training)

        print(
            f"[BLiMP-NL] step {state.global_step}: "
            f"macro accuracy = {macro_accuracy_mean:.4f} ± {macro_accuracy_std:.4f}"
        )
        # Logged as mean/lower/upper (not mean/std) so the three land on one wandb line-plot
        # panel and read together as a mean-with-band curve.
        wandb.log(
            {
                "blimp_nl/macro_accuracy_mean": macro_accuracy_mean,
                "blimp_nl/macro_accuracy_lower": macro_accuracy_mean - macro_accuracy_std,
                "blimp_nl/macro_accuracy_upper": macro_accuracy_mean + macro_accuracy_std,
            },
            step=state.global_step,
        )
        return control
