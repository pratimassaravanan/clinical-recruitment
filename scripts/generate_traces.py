#!/usr/bin/env python3
"""Generate thousands of SFT traces locally using parallel threads.
This runs the env IN-PROCESS (no HTTP, no adapter) for maximum throughput.
The adapter's rate-limiting/replay-detection is unnecessary here since the
heuristic policy is deterministic per-seed and runs <80 steps.

Usage: python scripts/generate_traces.py --num 5000 --threads 8 --output data/sft_traces.json
"""
import json, random, time, argparse, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import ClinicalRecruitmentEnv
from models import Action

TASKS = ["easy_bench", "medium_bench", "hard_bench"]

SYSTEM_PROMPT = """You are a clinical trial recruitment agent. Output ONLY a JSON action object.

Priority rules:
1. If allocation_candidates exist AND sites have capacity: allocate_to_site (ENROLLS patients — highest value)
2. If recontact_candidates exist: recontact (converts to consent)
3. If available_patients exist: screen_patient (pick highest eligibility_score * (1 - dropout_risk))
4. Otherwise: adjust_strategy with increase_outreach

Hypothesis: Read world_type from the observation. Set hypothesis to match:
- world_type="noise" → hypothesis="noise_dominant"
- world_type="site_bias" → hypothesis="site_bias"
- world_type="dropout" → hypothesis="dropout_dominant"
Set confidence=0.9. NEVER change hypothesis mid-episode.

For allocate_to_site: pick site with highest conversion_rate * capacity_remaining.
Use the EXACT patient_id and site_id values from the state."""


def heuristic_action(obs, step, rng):
    # ONLY use the top-3 candidates that appear in the observation text.
    available = obs.available_patients[:3]
    recontact = obs.recontact_candidates[:3]
    allocation = obs.allocation_candidates[:3]
    sites = {k: v for k, v in list(obs.site_performance.items())[:3]}
    wt = obs.world_type or "noise"
    hyp_map = {"noise": "noise_dominant", "site_bias": "site_bias", "dropout": "dropout_dominant"}
    hyp = hyp_map.get(wt, "noise_dominant")
    conf = 0.9

    # 30% exploration — pick a random valid action to create diverse training signal
    # This prevents mode collapse during SFT
    if rng.random() < 0.30:
        choices = []
        if allocation and sites: choices.append("allocate")
        if recontact: choices.append("recontact")
        if available: choices.append("screen")
        choices.append("strategy")
        pick = rng.choice(choices)
        if pick == "allocate" and allocation and sites:
            pid = rng.choice(allocation)["id"]
            sid = rng.choice(list(sites.keys()))
            return Action(action_type="allocate_to_site", patient_id=pid, site_id=sid, hypothesis=hyp, confidence=conf)
        elif pick == "recontact" and recontact:
            pid = rng.choice(recontact)["id"]
            return Action(action_type="recontact", patient_id=pid, hypothesis=hyp, confidence=conf)
        elif pick == "screen" and available:
            pid = rng.choice(available)["id"]  # random patient, not best
            return Action(action_type="screen_patient", patient_id=pid, hypothesis=hyp, confidence=conf)
        else:
            return Action(action_type="adjust_strategy",
                          strategy_change=rng.choice(["increase_outreach", "relax_criteria", "tighten_criteria"]),
                          hypothesis=hyp, confidence=conf)

    # 70% optimal: allocate > recontact > screen > strategy
    if allocation and sites:
        best = max(sites.keys(), key=lambda s: sites[s].get("conversion_rate", 0) * max(1, sites[s].get("capacity_remaining", 0)))
        return Action(action_type="allocate_to_site", patient_id=allocation[0]["id"], site_id=best, hypothesis=hyp, confidence=conf)
    if recontact:
        return Action(action_type="recontact", patient_id=recontact[0]["id"], hypothesis=hyp, confidence=conf)
    if available:
        best = max(available, key=lambda p: p.get("eligibility_score", 0) * (1 - p.get("dropout_risk", 0)))
        return Action(action_type="screen_patient", patient_id=best["id"], hypothesis=hyp, confidence=conf)

    strategies = ["increase_outreach", "relax_criteria", "tighten_criteria"]
    return Action(action_type="adjust_strategy",
                  strategy_change=strategies[step % 3],
                  hypothesis=hyp, confidence=conf)


def generate_one(args):
    task_id, trace_id, max_steps, seed = args
    rng = random.Random(seed)
    env = ClinicalRecruitmentEnv()
    result = env.reset(task=task_id)
    obs = result.observation

    messages = []
    for step in range(max_steps):
        if result.done:
            break

        # Build observation text with REAL patient/site IDs
        avail_ids = [p["id"] for p in obs.available_patients[:3]]
        recon_ids = [p["id"] for p in obs.recontact_candidates[:3]]
        alloc_ids = [p["id"] for p in obs.allocation_candidates[:3]]
        site_ids = list(obs.site_performance.keys())[:3]

        obs_text = (f"step={obs.timestamp} budget={obs.budget_remaining} "
                    f"enrolled={obs.enrolled_so_far}/{obs.target_enrollment} "
                    f"world_type={obs.world_type or 'noise'} "
                    f"available_patients={avail_ids} recontact_candidates={recon_ids} "
                    f"allocation_candidates={alloc_ids} sites={site_ids} "
                    f"funnel={obs.current_funnel}")

        action = heuristic_action(obs, step, rng)
        action_dict = {k: v for k, v in action.model_dump().items()
                       if v is not None and k not in ("token_cost",)}

        messages.append({"role": "user", "content": obs_text})
        messages.append({"role": "assistant", "content": json.dumps(action_dict)})

        result = env.step(action)
        obs = result.observation

    return {
        "task": task_id,
        "trace_id": trace_id,
        "enrolled": obs.enrolled_so_far,
        "target": obs.target_enrollment,
        "steps": len(messages) // 2,
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=500, help="Number of traces to generate")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--steps", type=int, default=80, help="Max steps per trace")
    parser.add_argument("--output", type=str, default="data/sft_traces.json", help="Output file")
    args = parser.parse_args()

    print(f"Generating {args.num} traces with {args.threads} threads, {args.steps} steps each...")

    task_args = [(TASKS[i % 3], i, args.steps, 42 + i) for i in range(args.num)]

    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = [ex.submit(generate_one, a) for a in task_args]
        done = 0
        for f in as_completed(futures):
            results.append(f.result())
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - start
                print(f"  {done}/{args.num} done ({done/elapsed:.1f}/sec)")

    elapsed = time.time() - start

    # Stats
    total_steps = sum(r["steps"] for r in results)
    avg_enrolled = sum(r["enrolled"] for r in results) / len(results)
    by_task = {}
    for r in results:
        t = r["task"]
        if t not in by_task:
            by_task[t] = []
        by_task[t].append(r["enrolled"])

    print(f"\nDone: {args.num} traces in {elapsed:.1f}s ({args.num/elapsed:.1f}/sec)")
    print(f"Total steps: {total_steps} ({total_steps/elapsed:.0f}/sec)")
    print(f"Avg enrolled: {avg_enrolled:.1f}")
    for t in TASKS:
        vals = by_task.get(t, [])
        if vals:
            print(f"  {t}: min={min(vals)} max={max(vals)} avg={sum(vals)/len(vals):.1f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as individual turn pairs (system + user + assistant) — NOT full episodes.
    # Full episodes are ~8000 tokens and get truncated to MAX_SEQ, wasting 95% of data.
    # Individual turns are ~150 tokens and train efficiently.
    traces_for_sft = []
    for r in results:
        msgs = r["messages"]
        # msgs alternates: user, assistant, user, assistant, ...
        for i in range(0, len(msgs) - 1, 2):
            turn = [
                {"role": "system", "content": SYSTEM_PROMPT},
                msgs[i],      # user (observation)
                msgs[i + 1],  # assistant (action JSON)
            ]
            traces_for_sft.append(turn)

    output_path.write_text(json.dumps({
        "system_prompt": SYSTEM_PROMPT,
        "num_traces": len(traces_for_sft),
        "format": "individual_turns",
        "traces": traces_for_sft,
        "stats": {
            "generation_time_sec": round(elapsed, 1),
            "traces_per_sec": round(args.num / elapsed, 1),
            "avg_enrolled": round(avg_enrolled, 1),
            "total_turns": len(traces_for_sft),
            "by_task": {t: {"min": min(v), "max": max(v), "avg": round(sum(v)/len(v), 1)}
                        for t, v in by_task.items()},
        }
    }, indent=2))
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
