# 3-Minute Pitch: Adaptive Clinical Trial Recruitment

## Timing Guide
- **0:00-0:30** - Hook & Problem (30 sec)
- **0:30-1:15** - Solution (45 sec)
- **1:15-2:00** - Results (45 sec)
- **2:00-2:45** - Theme Alignment (45 sec)
- **2:45-3:00** - Call to Action (15 sec)

---

## Script Outline

### [0:00-0:30] HOOK: The $8M/Day Problem

> "80% of clinical trials fail to meet enrollment deadlines. Every day of delay costs up to $8 million dollars. That's not a technology problem - it's a *planning* problem."

**Key point:** Ground in real stakes. This isn't academic - it's a multi-billion dollar industry pain point.

---

### [0:30-1:15] SOLUTION: A Long-Horizon Planning Benchmark

> "We built a 180-step sequential decision environment that captures everything that makes clinical recruitment hard:"

**Bullet points to hit:**
1. **Delayed consequences** - "Actions today affect outcomes 30 days later"
2. **Non-stationary dynamics** - "Patient pool quality degrades over time"
3. **Multi-objective optimization** - "Balance speed, budget, and retention"
4. **Beyond context limits** - "Episodes longer than any transformer's memory"

> "We implemented *all 50 features* from the Theme #2 checklist - delayed effects, milestone tracking, episodic memory, hierarchical planning, error recovery."

**Show diagram:** Environment architecture (patient funnel -> sites -> agent)

---

### [1:15-2:00] RESULTS: Hierarchical Planning Wins

> "We didn't just build a benchmark - we trained 4 research agents and ran rigorous experiments."

**Key results:**
- "HCAPO - hierarchical planning - scores **0.234**"
- "MemexRL - memory retrieval - scores **0.226**"  
- "KLong - flat baseline - scores **0.212**"
- "**10% improvement** with hierarchical temporal abstraction"
- "p-value 0.0075 - survives Bonferroni correction"

> "The finding? Memory helps, but *structured planning* matters more for long horizons."

**Show chart:** Agent comparison bar chart with error bars

---

### [2:00-2:45] THEME ALIGNMENT: Why This Matters for AI Agents

> "This directly addresses Theme #2's core questions:"

**Scale AI alignment:**
> "Business workflows ARE long-horizon planning. Goal decomposition, resource allocation, error recovery - that's what our environment tests."

**Mercor alignment:**
> "We track token efficiency. Agents learn to minimize reasoning cost while maximizing outcomes - exactly what you want in production."

**Technical completeness:**
- "228 passing tests"
- "NeurIPS-ready reproducibility report"
- "One-click Colab training"
- "Live HuggingFace deployment"

---

### [2:45-3:00] CALL TO ACTION

> "Clinical recruitment is one example. The real opportunity is *any* business workflow with long horizons and delayed feedback. Sales pipelines. Supply chains. Customer success."

> "Try the environment. Train your own agents. Help us build the benchmark suite that AI agents actually need."

**End with:**
> "Questions? Demo at our HuggingFace Space."

---

## Backup Q&A Prep

### "How is this different from existing benchmarks?"

> "Most benchmarks have short horizons or dense rewards. Atari averages 500 steps but rewards every frame. We have 180 steps with effects delayed 30+ days. That's qualitatively different planning."

### "Why clinical trials specifically?"

> "Three reasons: real stakes ($8M/day), published data for calibration, and it's the canonical long-horizon planning problem in industry. If you can solve recruitment, you can solve any business workflow."

### "What's the hardest part for agents?"

> "Milestone 3 - the 75% checkpoint. Agents need to shift from aggressive screening to retention focus. Most fail because they don't plan that transition."

### "How do you handle the beyond-context-limit problem?"

> "Episodic memory system. Agents can write summaries and retrieve them later. We track retrieval relevance and penalize unnecessary memory operations."

---

## Visual Aids Checklist

- [ ] Slide 1: Problem statement with $8M stat
- [ ] Slide 2: Environment architecture diagram
- [ ] Slide 3: Agent comparison chart with p-values
- [ ] Slide 4: Theme #2 alignment table
- [ ] Slide 5: QR code to HuggingFace Space

---

## Presenter Notes

- **Energy:** Start strong with the hook. Numbers grab attention.
- **Pacing:** Slow down for the results section - let stats land.
- **Gesture:** Point to diagrams, don't just show them.
- **Close:** Make eye contact on the call to action.

---

*Total word count: ~400 words spoken = comfortable 3-minute pace*
