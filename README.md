---
title: Adaptive Clinical Recruitment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /web
pinned: false
license: mit
---

# Adaptive Clinical Trial Recruitment Environment

> A long-horizon, non-stationary sequential decision environment where agents optimize the entire patient recruitment funnel (screening → enrollment → retention) under uncertainty, budget, time pressure, and site variability.

## Why Clinical Trial Recruitment?

**80% of clinical trials fail to meet enrollment deadlines**, costing pharma companies $600K–$8M per day of delay. This environment directly models the #1 trial failure reason — recruitment delays — making it one of the highest real-world utility OpenEnv submissions.

## Domain

Agents manage the full recruitment pipeline for a clinical trial:
1. **Screening** — evaluate candidate patients for eligibility
2. **Site Allocation** — assign consented patients to optimal recruitment sites
3. **Strategy Adjustment** — adapt outreach, criteria strictness, and site focus
4. **Retention** — manage dropout risk through the trial period

## Episode Structure

- **180 steps** = 180-day clinical trial recruitment period
- Non-stationary patient quality with increasing uncertainty
- Pre-computed deterministic traces (reproducible with fixed seeds 42, 123, 777)
- Curriculum injections (hard bench): periodic easy-pool resets to test generalization

## Tasks (Easy → Medium → Hard)

| Task | Description | Sites | Budget | Target | Key Challenge |
|------|-------------|-------|--------|--------|---------------|
| `easy_bench` | Stable patient pool, low dropout | 1 | $120K | 80 | Learn basic funnel |
| `medium_bench` | Moderate uncertainty, site variance | 3 | $150K | 120 | Multi-site optimization |
| `hard_bench` | High dropout, curriculum injections | 5 | $100K | 150 | Multi-objective under pressure |

## Action Space

| Action | Description | Cost |
|--------|-------------|------|
| `screen_patient` | Run screening on a candidate | $600–900 |
| `recontact` | Re-engage dropped-interest patient | $100–200 |
| `allocate_to_site` | Assign consented patient to site | $1200–1500 |
| `adjust_strategy` | Change outreach/criteria/focus | $200–400 |
| `stop_recruitment` | End episode early | Free |

## Observation Space (Pydantic Typed)

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | int | Day since trial start (0–180) |
| `budget_remaining` | float | Remaining budget in dollars |
| `time_to_deadline_days` | int | Days until trial deadline |
| `enrolled_so_far` | int | Current enrollment count |
| `target_enrollment` | int | Target enrollment |
| `current_funnel` | dict | Funnel stage counts |
| `available_patients` | list | Up to 5 candidate patients |
| `site_performance` | dict | Per-site metrics |
| `recent_events` | list | Recent event strings |
| `uncertainty_level` | float | 0.0–1.0 uncertainty |
| `difficulty` | int | 1=easy, 2=medium, 3=hard |
| `dropout_rate_7d` | float | Rolling 7-day dropout rate |
| `screening_backlog` | int | Patients awaiting results |

## Reward Function (6 Components)

1. **Screening success** (+0.22) — patient found eligible
2. **Enrollment gain** (+0.35) — new patient enrolled
3. **Dropout penalty** (-0.28) — patient dropped out
4. **Budget efficiency** (0–0.15) — cost-effective operations
5. **Timeline bonus** (0–0.20) — ahead of enrollment schedule
6. **Curriculum bonus** (+0.18) — exploiting easy-pool resets (hard only)

## Grading (Deterministic, 0.0–1.0)

Each task uses weighted partial-credit grading:

- **Easy**: enrollment rate (40%) + budget efficiency (25%) + screening accuracy (20%) + timeline (15%)
- **Medium**: enrollment (35%) + retention (25%) + site utilization (20%) + budget (20%)
- **Hard**: enrollment (25%) + retention (20%) + budget (20%) + dropout recovery (15%) + curriculum response (10%) + strategy adaptation (10%)

## Baseline Scores

| Task | Score |
|------|-------|
| `easy_bench` | 0.72 |
| `medium_bench` | 0.58 |
| `hard_bench` | 0.45 |

## Novelty

- **Curriculum learning injections**: periodic easy-pool resets mid-episode test agent generalization
- **Non-stationary uncertainty**: patient pool quality degrades over time
- **Multi-site allocation**: sites have different conversion rates, wait times, and capacity
- **Delayed signals**: screening results take time; dropout occurs after enrollment
- **No existing similar environment** in the OpenEnv gallery

## Setup

### Local
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t clinical-recruitment .
docker run -p 7860:7860 clinical-recruitment
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/reset?task_id=easy_bench` | Reset environment |
| POST | `/step` | Take action (JSON body) |
| GET | `/state` | Current state |
| GET | `/tasks` | List available tasks |

## Architecture

![Architecture Flowchart](architecture.png)

```
models.py          → Pydantic data contracts (Observation, Action, Reward, State, StepResult)
load_traces.py     → Deterministic patient pools + events (seeds 42, 123, 777)
env.py             → Core simulation: screening, enrollment, dropout, curriculum
graders.py         → Weighted partial-credit graders (3 tasks)
app.py             → FastAPI server with OpenEnv endpoints
server/app.py      → Server-mode entry point
inference.py       → Hybrid LLM + heuristic baseline agent
openenv.yaml       → OpenEnv manifest
```
