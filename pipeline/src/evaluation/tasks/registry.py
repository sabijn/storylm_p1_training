"""Registry of the extra eval tasks selectable via an eval_*.yaml `tasks:` block.

Three kinds:
- "zero_shot": takes (model, tokenizer, device, **kwargs) - fast, runs against the live model.
- "zero_shot_output": takes (model, tokenizer, device, output_dir, model_name, **kwargs) - same
  as zero_shot but also writes files to output_dir (currently just blimp_nl, for its
  per-subset + summary CSVs).
- "finetune": takes (model_dir, tokenizer, device, output_dir, cfg, **kwargs) - trains a
  fresh throwaway copy of the model per task, evaluates it, discards it. Slower, and the
  checkpoint being evaluated is never itself modified.
"""

from dataclasses import dataclass
from typing import Callable

from ..blimp_nl import evaluate_blimp_nl_task
from .belebele_nl import evaluate_belebele_nl
from .copanl import evaluate_copanl
from .crows_pairs_nl import evaluate_crows_pairs_nl
from .hellaswag_nl import evaluate_hellaswag_nl
from .sib200 import evaluate_sib200
from .sicknl import evaluate_sicknl
from .squadnl import evaluate_squadnl
from .storycloze_nl import evaluate_storycloze_nl
from .xcomps_nl import evaluate_xcomps_nl


@dataclass
class TaskSpec:
    name: str
    kind: str  # "zero_shot" | "zero_shot_output" | "finetune"
    fn: Callable


TASK_REGISTRY: dict[str, TaskSpec] = {
    "blimp_nl": TaskSpec("blimp_nl", "zero_shot_output", evaluate_blimp_nl_task),
    "hellaswag_nl": TaskSpec("hellaswag_nl", "zero_shot", evaluate_hellaswag_nl),
    "storycloze_nl": TaskSpec("storycloze_nl", "zero_shot", evaluate_storycloze_nl),
    "belebele_nl": TaskSpec("belebele_nl", "zero_shot", evaluate_belebele_nl),
    "xcomps_nl": TaskSpec("xcomps_nl", "zero_shot", evaluate_xcomps_nl),
    "crows_pairs_nl": TaskSpec("crows_pairs_nl", "zero_shot", evaluate_crows_pairs_nl),
    "copanl": TaskSpec("copanl", "zero_shot", evaluate_copanl),
    "sib200": TaskSpec("sib200", "finetune", evaluate_sib200),
    "sicknl": TaskSpec("sicknl", "finetune", evaluate_sicknl),
    "squadnl": TaskSpec("squadnl", "finetune", evaluate_squadnl),
}
