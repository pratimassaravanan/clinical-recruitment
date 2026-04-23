---
title: Adaptive Clinical Recruitment
emoji: "\U0001F3E5"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /web
pinned: false
license: mit
---

# Adaptive Clinical Trial Recruitment Environment

> A long-horizon benchmark for sequential trial-planning decisions across screening, recontact, site allocation, planning, and memory use.

**Hackathon positioning: Theme #2. Not a Wild Card entry.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Current Repo Truth

- `180`-step episodes with one simulated day per step
- `3` public tasks: `easy_bench`, `medium_bench`, `hard_bench`
- Typed `Observation`, `Action`, and `State` models in `models.py`
- `8` implemented action types
- `37`-dimensional numeric feature vector in `training/neural_policy.py`
- FastAPI/OpenEnv endpoints in `app.py` and `server/app.py`
- `4` repo baselines: HCAPO, MiRA, KLong, and MemexRL

The repo also contains auxiliary research helpers and reporting scripts. The claims in this README are limited to the core benchmark path, the four trainable baselines, and the regenerated artifacts under `data/sweep_results/`, `docs/images/`, and `paper/`.

## Submission Links

- Hugging Face Space: `https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment`
- Live environment URL: `https://pratimassaravanan-clinical-recruitment.hf.space`
- Mini-blog draft for Hugging Face Article: `HUGGINGFACE_BLOG.md`
- Communication deck: `docs/communication/adaptive_clinical_recruitment_presentation.html`
- Poster pack: `docs/communication/adaptive_clinical_recruitment_posters.pdf`
- OpenEnv + TRL starter notebook: `notebooks/training_grpo_openenv.ipynb`

The repo keeps a draft copy of the mini-blog in `HUGGINGFACE_BLOG.md` so the same content can be mirrored to a Hugging Face article later if needed.

## Benchmark Surfaces

### Tasks

| Task | Sites | Budget | Target | Main pressure |
|------|-------|--------|--------|---------------|
| `easy_bench` | 1 | `$120K` | `80` | Basic funnel dynamics |
| `medium_bench` | 3 | `$150K` | `120` | Multi-site coordination |
| `hard_bench` | 5 | `$100K` | `150` | Budget, retention, and constraint pressure |

### Action interface

| Action | Required fields | Notes |
|--------|-----------------|-------|
| `screen_patient` | `patient_id` | Choose from `observation.available_patients` |
| `recontact` | `patient_id` | Choose from `observation.recontact_candidates` |
| `allocate_to_site` | `patient_id`, `site_id` | Choose from `observation.allocation_candidates` and `observation.site_performance` |
| `adjust_strategy` | `strategy_change` | Examples: `increase_outreach`, `relax_criteria`, `tighten_criteria`, `focus_site_A`, `negotiate_site_A` |
| `plan_next_phase` | `target_phase` | Valid phases: `screening`, `conversion`, `allocation`, `retention`, `recovery` |
| `summarize_and_index` | `memory_key` | `memory_payload` is optional |
| `retrieve_relevant_history` | `memory_query` | Returns context via `observation.retrieved_memory_context` |
| `stop_recruitment` | none | Ends the episode early |

Site negotiation is represented through `adjust_strategy` values such as `negotiate_site_A`; it is not a separate top-level action.

### Observations and features

The typed observation includes:

- Funnel state and action-specific candidate pools
- Per-site performance metrics
- Milestones, constraints, and delayed-effects state
- Plan state and indexed-memory summaries
- Token budget and token-efficiency signals
- Counterfactual hints and rollout estimates

The trainable baselines consume the `37`-dimensional feature vector extracted by `training/neural_policy.py:extract_state_features()`.

### Repo baselines

| Baseline | Main mechanism |
|----------|----------------|
| `HCAPO` | Hierarchical subgoals and hindsight relabeling |
| `MiRA` | Milestone-aware potential shaping |
| `KLong` | Multi-scale temporal aggregation and TD(lambda)-style credit assignment |
| `MemexRL` | Episodic memory with learned write/read behavior |

These are described as repo baselines inspired by long-horizon RL ideas. The docs do not claim externally validated reproductions of external named methods.

## Fresh 5-Seed Sweep

Current 5-seed sweep artifacts were generated on `2026-04-21`.

| Baseline | Mean | Std | 95% CI |
|----------|------|-----|--------|
| `HCAPO` | `0.2215` | `0.0127` | `[0.2100, 0.2303]` |
| `KLong` | `0.2152` | `0.0222` | `[0.1977, 0.2286]` |
| `MemexRL` | `0.2148` | `0.0270` | `[0.1943, 0.2352]` |
| `MiRA` | `0.2094` | `0.0095` | `[0.2023, 0.2165]` |

No pairwise comparison reaches `p < 0.05`.

- `HCAPO` has the highest mean in the current sweep, but that is not a statistically significant lead.
- `HCAPO vs KLong` is `p = 0.3849`, so the older overclaim of clear baseline separation is stale.
- The current baseline suite is best interpreted as a stress test for the benchmark and evaluation path, not as evidence that one method clearly dominates.
- Recorded integration result: `30/30` checks passed across the three tasks.

Fresh outputs live in:

- `data/sweep_results/benchmark_report.{md,json}`
- `data/sweep_results/sweep_results.{csv,json}`
- `data/sweep_results/significance_tests.json`
- `data/sweep_results/integration_tests.json`
- `docs/images/agent_comparison.{png,svg}`
- `docs/images/seed_heatmap.{png,svg}`
- `docs/images/score_boxplot.{png,svg}`

The three main benchmark diagrams are regenerated from `scripts/generate_docs_diagrams.py`:

- `docs/images/environment_architecture.png`
- `docs/images/agent_architectures.png`
- `docs/images/training_pipeline.png`

## Installation

### Base serving stack

```bash
pip install -r requirements.txt
```

### Training and reporting extras

```bash
pip install -r requirements-research.txt numpy
```

`requirements.txt` covers the API and OpenEnv serving path. The training and sweep scripts also import `numpy`, and the reporting scripts use `pandas` and `matplotlib`.

## Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Available endpoints:

- `GET /`
- `GET /health`
- `POST /reset?task_id=easy_bench`
- `POST /step`
- `GET /state`
- `GET /tasks`

The built-in OpenEnv UI is attached at `/web`, and `/dashboard` redirects there. Set `ENABLE_WEB_INTERFACE=false` only if you explicitly want to disable it.

## Python Usage

```python
from env import ClinicalRecruitmentEnv
from models import Action

env = ClinicalRecruitmentEnv()
result = env.reset(task="medium_bench")

while not result.done:
    obs = result.observation

    if obs.allocation_candidates and obs.site_performance:
        action = Action(
            action_type="allocate_to_site",
            patient_id=obs.allocation_candidates[0]["id"],
            site_id=next(iter(obs.site_performance)),
        )
    elif obs.recontact_candidates:
        action = Action(
            action_type="recontact",
            patient_id=obs.recontact_candidates[0]["id"],
        )
    elif obs.available_patients:
        action = Action(
            action_type="screen_patient",
            patient_id=obs.available_patients[0]["id"],
            hypothesis="noise_dominant",
            confidence=0.7,
        )
    else:
        action = Action(
            action_type="adjust_strategy",
            strategy_change="increase_outreach",
        )

    result = env.step(action)

print(result.info.get("final_score"))
```

For `recontact`, use `observation.recontact_candidates`. For `allocate_to_site`, use `observation.allocation_candidates` and a valid `site_id` from `observation.site_performance`.

## Experiment Commands

```bash
python experiments/train_agents.py --agent all --episodes 50
python experiments/train_agents.py --agent hcapo --episodes 50 --tasks easy_bench medium_bench hard_bench
python experiments/full_sweep.py --seeds 1 7 21 42 123 --episodes 30 --eval-episodes 5
python experiments/run_research.py --episodes 3
python experiments/run_progressive_training.py
python scripts/generate_docs_diagrams.py
python scripts/generate_charts.py
```

`experiments/full_sweep.py` writes fresh reports to `data/sweep_results/` and refreshes the sweep charts under `docs/images/`.

## OpenEnv GRPO Starter

The current starter notebook is:

```bash
notebooks/training_grpo_openenv.ipynb
```

It is a minimal OpenEnv + TRL GRPO bootstrap against the live Space and current `8`-action interface. It should be treated as a starting point for hackathon training runs, not as benchmark evidence by itself.

## API Example

```bash
curl -X POST "http://localhost:7860/reset?task_id=easy_bench"

curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "screen_patient",
    "patient_id": "<id from observation.available_patients>",
    "hypothesis": "noise_dominant",
    "confidence": 0.7
  }'
```

Episodes finish with a graded `final_score` in `result.info["final_score"]`, clamped into `(0, 1)`.

## Verification

Recorded local validation counts:

| Suite | Result |
|-------|--------|
| `test_env.py` | `76/76` |
| `test_agents.py` | `43/43` |
| `test_research_modules.py` | `109/109` |
| `test_local_serving.py` | `77/77` |

Additional generated artifacts:

- `paper/main.pdf` is included in the repo.
- `scripts/generate_docs_diagrams.py` is the source for the three main benchmark diagrams.
- `data/sweep_results/benchmark_report.md` contains the current benchmark summary used by the docs.

## Caveats

- This is a synthetic benchmark, not a deployment-ready clinical operations system.
- The repo contains extra research helpers and scaffolds; not every auxiliary module should be described as a validated benchmark contribution.
- The current 5-seed sweep does not show statistically significant separation among the four repo baselines.
- Older drafts that described extra actions, roadmap-complete coverage, or clear baseline dominance are stale and should be ignored.

## License

MIT License. See `LICENSE`.
