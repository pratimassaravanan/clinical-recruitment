# Final Model Run

This subfolder points to the final best run and its related assets.

## Final Published Model

- Model repo: `https://huggingface.co/pratimassaravanan/grpo`
- Best run / job: `69ed6d57`
- Method: `SFT warmup + 80-step GRPO`
- Base model tag on Hub: `Qwen/Qwen3-1.7B`

Download the adapter and tokenizer files directly from the model repo.

## Related Assets

- Training script used for the best run: local file `train_sft_grpo_hfjob.py`
- Space blob URL for that script:
  `https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/train_sft_grpo_hfjob.py`
- Blog post for the final story:
  `https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/Blog.md`

## Plots / Proof of Training

Download from:

- `https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/tree/main/article_assets`

Key files:

- `grpo_80step_training.png`
- `all_runs_comparison.png`
- `collapse_analysis.png`
- `enrollment_progression.png`
- `grpo_comparison.png`

## Comparison Models

- `https://huggingface.co/pratimassaravanan/grpo_output`
- `https://huggingface.co/pratimassaravanan/clinical-qwen3-4b-sft-lora`

These are comparison runs, not the final best model.
