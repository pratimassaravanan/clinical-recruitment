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
1. If allocation_candidates exist AND sites have capacity: allocate_to_site (ENROLLS patients)
2. If recontact_candidates exist: recontact (converts to consent)
3. If available_patients exist: screen_patient (starts the funnel)
4. Otherwise: adjust_strategy

Use the EXACT patient_id and site_id values from the state."""


def heuristic_action(obs, step, rng):
    available = obs.available_patients
    recontact = obs.recontact_candidates
    allocation = obs.allocation_candidates
    sites = obs.site_performance
    wt = obs.world_type or "noise"
    hyp_map = {"noise": "noise_dominant", "site_bias": "site_bias", "dropout": "dropout_dominant"}
    hyp = hyp_map.get(wt, "noise_dominant")

    # 15% random exploration
    if rng.random() < 0.15:
        choices = []
        if available: choices.append("screen")
        if recontact: choices.append("recontact")
        if allocation and sites: choices.append("allocate")
        choices.append("strategy")
        pick = rng.choice(choices)
        if pick == "allocate" and allocation and sites:
            return Action(action_type="allocate_to_site", patient_id=rng.choice(allocation)["id"],
                          site_id=rng.choice(list(sites.keys())), hypothesis=hyp, confidence=0.8)
        elif pick == "recontact" and recontact:
            return Action(action_type="recontact", patient_id=rng.choice(recontact)["id"],
                          hypothesis=hyp, confidence=0.7)
        elif pick == "screen" and available:
            return Action(action_type="screen_patient", patient_id=rng.choice(available)["id"],
                          hypothesis=hyp, confidence=0.7)
        else:
            return Action(action_type="adjust_strategy",
                          strategy_change=rng.choice(["increase_outreach", "relax_criteria", "tighten_criteria"]),
                          hypothesis=hyp, confidence=0.6)

    # Priority: allocate > recontact > screen > strategy
    if allocation and sites:
        best = max(sites.keys(), key=lambda s: sites[s].get("conversion_rate", 0) * max(1, sites[s].get("capacity_remaining", 0)))
        return Action(action_type="allocate_to_site", patient_id=allocation[0]["id"], site_id=best, hypothesis=hyp, confidence=0.8)
    if recontact:
        return Action(action_type="recontact", patient_id=recontact[0]["id"], hypothesis=hyp, confidence=0.75)
    if available:
        best = max(available, key=lambda p: p.get("eligibility_score", 0) * (1 - p.get("dropout_risk", 0)))
        return Action(action_type="screen_patient", patient_id=best["id"], hypothesis=hyp, confidence=0.7)

    # Periodic memory/planning
    if step % 20 == 5:
        return Action(action_type="plan_next_phase", target_phase=rng.choice(["screening", "conversion", "allocation"]),
                      plan_summary="progress the funnel")
    if step % 20 == 15:
        funnel = obs.current_funnel
        return Action(action_type="summarize_and_index", memory_key=f"step_{step}",
                      memory_payload=f"enrolled={funnel.get('enrolled',0)} screened={funnel.get('screened',0)}")

    return Action(action_type="adjust_strategy",
                  strategy_change=["increase_outreach", "relax_criteria", "tighten_criteria"][step % 3],
                  hypothesis=hyp, confidence=0.6)


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

    # Save as list of conversation traces (ready for SFT)
    traces_for_sft = []
    for r in results:
        trace = [{"role": "system", "content": SYSTEM_PROMPT}] + r["messages"]
        traces_for_sft.append(trace)

    output_path.write_text(json.dumps({
        "system_prompt": SYSTEM_PROMPT,
        "num_traces": len(traces_for_sft),
        "traces": traces_for_sft,
        "stats": {
            "generation_time_sec": round(elapsed, 1),
            "traces_per_sec": round(args.num / elapsed, 1),
            "avg_enrolled": round(avg_enrolled, 1),
            "by_task": {t: {"min": min(v), "max": max(v), "avg": round(sum(v)/len(v), 1)}
                        for t, v in by_task.items()},
        }
    }, indent=2))
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
