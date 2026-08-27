import wandb
from transformers import TrainerCallback

from .blimp_nl import evaluate_blimp_nl_macro_accuracy


class BlimpNLCallback(TrainerCallback):
    """Runs a lightweight BLiMP-NL pass every `eval_steps` training steps and logs the
    macro-average accuracy to the active W&B run (see evaluate_blimp_nl_macro_accuracy)."""

    def __init__(self, tokenizer, eval_steps: int, normalize_by_length: bool = True):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.normalize_by_length = normalize_by_length

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_steps <= 0 or state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return control

        model = kwargs["model"]
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            macro_accuracy = evaluate_blimp_nl_macro_accuracy(
                model, self.tokenizer, device, normalize_by_length=self.normalize_by_length
            )
        finally:
            model.train(was_training)

        print(f"[BLiMP-NL] step {state.global_step}: macro accuracy = {macro_accuracy:.4f}")
        wandb.log({"blimp_nl/macro_accuracy": macro_accuracy}, step=state.global_step)
        return control
