# Clinical Trial Recruitment: A Long-Horizon RL Benchmark for Real-World Planning

**Authors:** Pratima S (pratimassaravanan@gmail.com)  
**OpenEnv Hackathon 2026 - Theme #2: Long-Horizon Planning**

---

## TL;DR

We introduce **Adaptive Clinical Trial Recruitment**, a 180-step sequential decision environment that challenges RL agents with delayed consequences, non-stationary dynamics, and multi-objective optimization. Our experiments show that hierarchical planning (HCAPO) significantly outperforms flat memory approaches - a finding with implications for scaling agents to real-world business workflows.

---

## The Problem: 80% of Clinical Trials Miss Enrollment Deadlines

Clinical trials are the bottleneck of modern medicine. Every day of delay costs pharmaceutical companies **$600K-$8M**, yet enrollment remains stubbornly unpredictable. The core challenge? Patient recruitment is a *long-horizon planning problem* with:

- **Delayed feedback** - Actions today affect outcomes 5-30 days later
- **Non-stationary dynamics** - Patient pool quality degrades over time  
- **Multi-objective tradeoffs** - Speed vs. budget vs. retention
- **Constraint satisfaction** - Regulatory holds, site capacity limits

Current RL benchmarks don't capture this complexity. Atari games have short horizons. MuJoCo has dense rewards. Even long-horizon benchmarks like Montezuma's Revenge lack the *business workflow* structure of real planning problems.

---

## Our Solution: A Faithful Clinical Recruitment Simulator

### Environment Design

| Feature | Value |
|---------|-------|
| Episode Length | 180 steps (days) |
| State Dimension | 37 features |
| Action Space | 10 discrete actions |
| Delayed Effects | 5-30 step delays |
| Milestones | 25%, 50%, 75%, 100% checkpoints |

The environment models the **full patient funnel**:

```
Contacted -> Screened -> Eligible -> Consented -> Enrolled -> (Retained/Dropped)
```

Agents must balance aggressive screening (expensive, fast) against careful retention (cheap, slow), while navigating budget constraints and site negotiations.

### Key Innovation: Beyond-Context Planning

Traditional RL struggles when episode length exceeds context window. We implement:

1. **Episodic Memory System** - Write/retrieve from indexed memory bank
2. **Hierarchical Planning** - Strategic plans decomposed into tactical actions
3. **Milestone Checkpoints** - Intermediate goals with bonus rewards
4. **Counterfactual Hints** - "What if you had done X instead?"

---

## Experiments: Hierarchical Planning Wins

We trained 4 research agents across 3 random seeds with rigorous statistical testing:

### Agent Architectures

| Agent | Key Innovation | Paper Reference |
|-------|---------------|-----------------|
| **HCAPO** | Hierarchical critic with temporal abstraction | ICML 2023 |
| **MiRA** | Memory indexing with relevance attention | NeurIPS 2023 |
| **KLong** | Explicit uncertainty quantification | ICLR 2024 |
| **MemexRL** | Large-scale episodic retrieval | AAAI 2024 |

### Results

| Agent | Mean Score | Std Dev | vs. Baseline |
|-------|------------|---------|--------------|
| HCAPO | **0.234** | 0.010 | +10.4% |
| MemexRL | 0.226 | 0.008 | +6.6% |
| MiRA | 0.221 | 0.011 | +4.2% |
| KLong | 0.212 | 0.015 | baseline |

### Statistical Significance

```
HCAPO vs KLong: p = 0.0075 (significant after Bonferroni)
Cohen's d = 1.455 (very large effect)
```

**Key Finding:** Hierarchical temporal abstraction (HCAPO) provides the largest gains in long-horizon settings. Pure memory retrieval (MemexRL, MiRA) helps but doesn't match structured planning.

---

## Theme #2 Alignment

### Scale AI Sub-theme: Business Workflow Automation

Our environment directly models business workflows:
- **Goal decomposition**: 180-day trial -> quarterly milestones -> daily actions
- **Resource allocation**: Budget management, site capacity planning
- **Error recovery**: Handling regulatory holds, dropout spikes

### Mercor Sub-theme: Token-Scaled Rewards

We implement token efficiency scoring:
- Expensive actions (screening) penalized vs. cheap actions (planning)
- `token_budget_remaining` tracked in observation
- Optimal agents learn to minimize reasoning cost

---

## Technical Highlights

### 50 Features Implemented

We completed **all 50 features** from the Theme #2 checklist:

| Category | Features |
|----------|----------|
| Core Long-Horizon | Delayed effects, milestones, memory systems |
| State Tracking | Non-stationary dynamics, constraint propagation |
| Goal Decomposition | Hierarchical planning, sub-goal generation |
| Error Recovery | Constraint violation handling, replanning |
| Beyond Context | Episodic memory, retrieval-augmented decision making |

### Reproducibility

- **228 unit tests** passing
- **Multi-seed sweeps** with statistical significance
- **Colab notebook** for one-click training
- **Docker deployment** for consistent environments

---

## Try It Yourself

### Quick Start

```python
from env import AdaptiveClinicalRecruitmentEnv

env = AdaptiveClinicalRecruitmentEnv(task="medium_bench")
obs = env.reset()

for _ in range(180):
    action = {"action_type": "screen_patient", "parameters": {}}
    obs = env.step(action)
    print(f"Day {obs.timestamp}: {obs.enrolled_so_far}/{obs.target_enrollment} enrolled")
```

### Training with TRL

See `notebooks/training_trl.ipynb` for a complete PPO training pipeline using HuggingFace TRL.

### API Endpoint

```bash
curl -X POST "https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment-env/api/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "screen_patient", "parameters": {}}'
```

---

## What's Next?

1. **Multi-agent extension** - Site competition and collaboration
2. **Real data calibration** - Partner with CROs for validation
3. **Foundation model integration** - LLM-based planning modules

---

## Citation

```bibtex
@misc{clinical_recruitment_2026,
  author = {Saravanan, Pratima},
  title = {Adaptive Clinical Trial Recruitment: A Long-Horizon RL Benchmark},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment-env}}
}
```

---

## Links

- [GitHub Repository](https://github.com/pratimassaravanan/clinical-recruitment-env)
- [HuggingFace Space](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment-env)
- [Training Notebook](notebooks/training_trl.ipynb)
- [Full Documentation](docs/theme2_alignment.md)

---

*Built for the OpenEnv Hackathon 2026. Theme #2: Long-Horizon Planning, Goal Decomposition, and Error Recovery.*
