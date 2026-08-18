from typing import Any

from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedModel

# Common size fields, as they appear in yaml model configs, mapped onto each
# architecture's own HF Config field names. LlamaConfig already uses these
# names directly; GPT2Config needs the translation below.
_GPT2_FIELD_MAP = {
    "hidden_size": "n_embd",
    "intermediate_size": "n_inner",
    "num_attention_heads": "n_head",
    "num_hidden_layers": "n_layer",
    "max_position_embeddings": "n_positions",
}

_COMMON_SIZE_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_hidden_layers",
    "max_position_embeddings",
)


def build_model(architecture: str, model_cfg: dict[str, Any], vocab_size: int) -> PreTrainedModel:
    """Build a randomly-initialized causal LM from a yaml-configured architecture + sizes."""
    extra = model_cfg.get("extra", {})

    if architecture == "gpt2":
        config = GPT2Config(
            vocab_size=vocab_size,
            **{_GPT2_FIELD_MAP[k]: model_cfg[k] for k in _COMMON_SIZE_FIELDS},
            **extra,
        )
        return GPT2LMHeadModel(config)

    if architecture == "llama":
        config = LlamaConfig(
            vocab_size=vocab_size,
            **{k: model_cfg[k] for k in _COMMON_SIZE_FIELDS},
            **extra,
        )
        return LlamaForCausalLM(config)

    raise ValueError(f"Unknown architecture: {architecture!r} (expected 'gpt2' or 'llama')")


def get_block_size(model: PreTrainedModel) -> int:
    """Read the model's max sequence length regardless of architecture.

    GPT2Config exposes this as `n_positions`; LlamaConfig as
    `max_position_embeddings`. Packing scripts use this as the single source
    of truth for chunk size instead of duplicating it in yaml.
    """
    config = model.config
    return getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions")
