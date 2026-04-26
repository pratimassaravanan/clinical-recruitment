# Final Bundle: `pratimassaravanan/grpo` (`job 69ed6d57`)

This folder is the clean handoff for the final best run.

It intentionally does **not** duplicate remote model weights or Hub artifacts. Each subfolder contains pointers to the canonical download location so the workspace stays small and there is only one source of truth.

## Final Canonical Assets

- Final model: `https://huggingface.co/pratimassaravanan/grpo`
- Final environment: `https://pratimassaravanan-clinical-recruitment.hf.space`
- OpenEnv UI: `https://pratimassaravanan-clinical-recruitment.hf.space/web/`
- Space repo: `https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment`
- Blog: `https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/Blog.md`
- Artifacts and plots: `https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts`

## Folder Layout

- `final_model_run/`: where to download the final adapter, plots, and training-script-related assets
- `base/`: where to download the base model used by the final adapter
- `openenv_files/`: which local repo files are the canonical environment implementation
- `comparison_models/`: links to earlier published runs kept for comparison only
- `manifest.json`: machine-readable summary of the final best run

## Notes

- Best published model: `pratimassaravanan/grpo`
- Earlier comparison models: `pratimassaravanan/grpo_output` and `pratimassaravanan/clinical-qwen3-4b-sft-lora`
- Current running jobs are **not** the final answer unless they later publish something better
- The current local `train.py` in repo root is newer than older pasted snippets; the best-run training script is still `train_sft_grpo_hfjob.py`
