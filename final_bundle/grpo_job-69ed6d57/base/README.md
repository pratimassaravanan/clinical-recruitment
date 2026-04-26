# Base Model

The final adapter in `pratimassaravanan/grpo` is built on top of:

- `https://huggingface.co/Qwen/Qwen3-1.7B`

## Why this is separate

The final adapter is the tuned delta. The base model should be downloaded from the upstream Qwen repo rather than copied into this workspace.

## Use with the final adapter

- Base model: `Qwen/Qwen3-1.7B`
- Final adapter: `https://huggingface.co/pratimassaravanan/grpo`

If you are reconstructing the full runtime stack, combine the base model with the published adapter using PEFT / Transformers.

## Related local files

- Best-run training script: `train_sft_grpo_hfjob.py`
- Alternate GRPO scripts: `train_grpo_hfjob.py`, `train_grpo_from_sft.py`
