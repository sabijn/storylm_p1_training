# pipeline

HF-`Trainer`-based pipeline to train a small Dutch LM from scratch, continue pretraining it
on a new dataset, and evaluate both with validation loss/perplexity + BLIMP-NL. Every stage
is driven by a yaml file in [configs/](configs/); nothing is hardcoded in the scripts.

## Order of operations

```bash
cd pipeline
pip install -r requirements.txt

# 1. Split the base-pretraining data into train/dev/test and cache it to disk
python scripts/01_prepare_data.py --config configs/data_base.yaml

# 2. Train a SentencePiece tokenizer on the train split, wrapped as a HF fast tokenizer
python scripts/02_train_tokenizer.py --config configs/tokenizer.yaml

# 3. Train a base model from scratch (pick gpt2 or llama sizing)
python scripts/03_train_base_model.py --config configs/model_base_gpt2.yaml

# 4. Evaluate the base model: dev/test loss + perplexity, and BLIMP-NL
python scripts/05_evaluate_model.py --config configs/eval_base.yaml

# 5. Prepare a different dataset, then continue pretraining the base model on it
python scripts/01_prepare_data.py --config configs/data_continued.yaml
python scripts/04_continue_pretraining.py --config configs/model_continued.yaml

# 6. Evaluate the continued-pretraining model the same way
python scripts/05_evaluate_model.py --config configs/eval_continued.yaml
```

`01_prepare_data.py` and `05_evaluate_model.py` are generic — they're reused for both the
base and continued-pretraining stages by pointing them at different configs.

## configs/

| File | Controls |
| --- | --- |
| `data_base.yaml` / `data_continued.yaml` | Dataset sources (HF hub or local CSV), text column, train/dev/test split fractions + seed, output dir |
| `tokenizer.yaml` | Vocab size, SentencePiece settings, special tokens, which prepared split to train on |
| `model_base_gpt2.yaml` / `model_base_llama.yaml` | Architecture, sizes (hidden/intermediate size, heads, layers, context length), `TrainingArguments` fields, wandb run info |
| `model_continued.yaml` | Base model + tokenizer paths to load, which data config to continue on, training/wandb fields |
| `eval_base.yaml` / `eval_continued.yaml` | Model/tokenizer to evaluate, which data config + split, BLIMP-NL settings, wandb run info |

Model config `model:` fields (`hidden_size`, `intermediate_size`, `num_attention_heads`,
`num_hidden_layers`, `max_position_embeddings`) are architecture-agnostic — the same names
are used for both GPT2 and Llama configs, translated internally by
[src/models/build_model.py](src/models/build_model.py). Architecture-specific extras (e.g.
Llama's `num_key_value_heads` for grouped-query attention, GPT2's dropout rates) go under
`model.extra`.

## Notes

- All paths in the example configs follow the `/local/perdijks/...` server convention used
  elsewhere in this repo — update `paths:`/`output_dir:` fields for your own environment.
- Device placement is left to `Trainer`/`accelerate` — set `CUDA_VISIBLE_DEVICES` externally
  rather than hardcoding it.
- W&B: each config's `wandb:` block (`entity`, `project`, `run_name`, `tags`) is used to start
  a run *before* the `Trainer` is constructed, so training curves and the post-training
  perplexity/BLIMP-NL metrics all land in the same run.

## (Solved) Bugs 🐛
- 🟢 The trained SentencePiece tokenizer is wrapped in a LlamaTokenizerFast. Transformers <5.0.0 requires a .model file to be loaded in LlamaTokenizerFast. However, Transformers >5.0.0 (now running 5.10.2) requires the vocabulary to be a list/dict (see also (https://huggingface.co/docs/transformers/v5.10.4/en/model_doc/llama)) so the .model file was ignored and LlamaTokenizerFast fell back onto an empty vocab with only special tokens (unk/pad/eos/bos). Now added an extractor for the SentencePiece tokenizer.
- 🟢 AutoModelForCausalLM.from_pretrained loads in evaluation mode by default (see [documentation](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained))
