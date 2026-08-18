# Repo for StorytimeLM training

## Directories
|- archive              Old code used to play around, kept around for safety reasons.   
|- try-out-own-gpt      Code to build/train a gpt-like model from scratch and evaluate it on multiple benchmarks.  
|- pipeline             HF-Trainer pipeline: tokenizer training, data prep (train/dev/test split), from-scratch base model training (GPT2/Llama, configurable via yaml), continued pretraining, and evaluation (val loss/perplexity + BLIMP-NL), logged to W&B.  

## Log
### Thursday August 13th 2026 & Friday August 14th 2026
**Tokenizer choices**:
- Library: Sentencepiece instead of HF based on the results of Ali (2024)
- Sizes: 16K, 32K favourable for TinyStories (Paraskeva, 2026) (in comparison: Eldan & Li (2023) use 10k-most common from pretrained GPT-neo tokenizer)
- Data: open question, in or out distribution?

**(Initial) Model choices options**:
| Architecture    | Positional embeddings          | Activation Function              | Normalisation
| --------        | -------                        | --------                         | ------- |
| GPT             | absolute positional embeddings | GELU                             | LayerNorm
| Llama           | rotary positional embeddings   | SiLU (simpler so more efficient) | RMS Layer Norm (faster and more stable)

*RMS Layer Norm*: LayerNorm removes a uniform shift across the features and normalizes the scale. RMSNorm normalizes the scale but retains information about the vector's mean. [Link here](https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html)

*Rotary position embeddings*: encodes position by rotating query and key vectors inside self-attention. It moves the position operation from the model input into every attention layer. Within each attention head, adjacent query and key features are treated as two-dimensional pairs. This pair is rotated (by position *t* and a channel-specific frequency) and encodes relative position information. 

*SiLU*: slightly longer off-zero with input values just below zero.

Further: Llama grouped-query attention and no learned bias vector in linear projections.

**Model sizes chosen by others**  
- Eldan and Li (2023): 1M/3M/9M/28M/33M  
- Paraskeva (2026): 20M/60M/180M

| Model size    | Hidden size (model_dim, d_size) | Intermediate size             | Attention heads | Transformer layers    |
| --------      | -------                         | --------                      | -------         | -------               |
| 20M           | 384                             |  768                          | 6               | 6                     |
| 60M           | 512                             |  1024                         | 8               | 16                    |
| 180M          | 768                             |  1792                         | 12              | 24                    |

- GPT-2-small (Radford, 2019): 117M, 12 layers, model size (d_size) 768
- Nanochat: tuneable via `--depth` parameter (e.g. GPT-2-big `depth = 26`). `model_dim = depth * 64`. `num_heads = model_dim / 128`. Weight decay scales with `1/depth^2` [Link here](https://github.com/karpathy/nanochat/discussions/481)
- Context window? 

### Monday August 17th
**Calculating model parameters**
- Token embeddings: vocab_size * model_dim
- Multi-head attention: 4 * (model_dim * model_dim) (i.e. W_Q, W_V, W_K, W_O (output projection))
- FFN: 2 * (intermediate size * model_dim) (the times two is expansion and projection)
- Layer-norm: 4 * model_dim (pre-attention, pre-FFN, beta and gamma)
- One layer: multi-head attention + FFN + layer_norm
- All transformer layers: amount_layers * one_layer
- Final components: 2 * model_dim (final layer_norm) + vocab_dim * model_dim (LM-head)
- Total: token_embeddings + all_transformer_layers + final_components

*Example Paraskeva 20M*. 
- Token embeddings: 16K * 384
- Multi-head attention: 4 * (384 * 384) (i.e. W_Q, W_V, W_K, W_O (output projection))
- FFN: 2 * (768 * 384) (the times two is expansion and projection)
- Layer-norm: 4 * 384 (pre-attention, pre-FFN, beta and gamma)
- One layer: (4 * (384 * 384)) + 2 * (768 * 384) + (4 * 384)
- All transformer layers: 6 * ((4 * (384 * 384)) + 2 * (768 * 384) + (4 * 384))
- Final components: 2 * 384 (final layer_norm) + 16_000 * 384 (LM-head)
- Total: (16_000 * 384) + (6 * ((4 * (384 * 384)) + 2 * (768 * 384) + (4 * 384))) + (2 * 384 + 16_000 * 384)
- Approx: 19.3M = 20M

## Tuesday August 18th
- Tokenizer 16K and 32K
- Tokenizer trained on BabyLM data, StoriesFTW, and combination
- Kies gewoon een model size en begin
