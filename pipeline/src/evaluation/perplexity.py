import math

from transformers import Trainer


def evaluate_perplexity(trainer: Trainer, eval_dataset=None) -> dict[str, float]:
    """Run trainer.evaluate() and derive perplexity from the resulting loss."""
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    loss = metrics["eval_loss"]
    return {"loss": loss, "perplexity": math.exp(loss)}
