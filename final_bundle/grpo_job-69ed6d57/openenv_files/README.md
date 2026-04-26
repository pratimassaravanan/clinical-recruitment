# OpenEnv Files

This subfolder intentionally does not duplicate the environment source files.

The canonical implementation already lives in this repository root. Use these local files directly:

- `app.py`
- `server/app.py`
- `env.py`
- `models.py`
- `openenv_adapter.py`
- `openenv.yaml`
- `custom_gradio_ui.py`
- `README.md`
- `Blog.md`

## Live OpenEnv Links

- Environment: `https://pratimassaravanan-clinical-recruitment.hf.space`
- OpenEnv UI: `https://pratimassaravanan-clinical-recruitment.hf.space/web/`
- Dashboard alias: `https://pratimassaravanan-clinical-recruitment.hf.space/dashboard`

## What these files cover

- `env.py`: core benchmark dynamics and reward logic
- `models.py`: typed observation / action / state schema
- `openenv_adapter.py`: OpenEnv adapter with session-safe wrapper and anti-abuse protections
- `app.py` and `server/app.py`: FastAPI + OpenEnv serving layer
- `openenv.yaml`: OpenEnv manifest and task metadata
- `custom_gradio_ui.py`: custom guide tab for the `/web` interface

## Important note on training code

The final best-run training script is `train_sft_grpo_hfjob.py`.

The current repo-root `train.py` is a newer local SFT script and should not be confused with older pasted versions that still assumed nested HTTP observations.
