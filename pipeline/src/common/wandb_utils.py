from typing import Any

import wandb


def init_wandb(wandb_cfg: dict[str, Any], run_config: dict[str, Any] | None = None):
    """Start a W&B run from a config's `wandb:` block.

    Must be called before `transformers.Trainer` is constructed: the Trainer's
    WandbCallback only calls `wandb.init()` itself if no run is active yet, so
    calling this first makes training logs and any metrics we log manually
    (perplexity, BLiMP-NL) land in the same run.
    """
    return wandb.init(
        entity=wandb_cfg["entity"],
        project=wandb_cfg["project"],
        name=wandb_cfg.get("run_name"),
        tags=wandb_cfg.get("tags"),
        dir=wandb_cfg.get("dir"),
        config=run_config or {},
    )
