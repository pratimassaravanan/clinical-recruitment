#!/usr/bin/env python3
"""Test parallel trace generation speed -- LOCAL in-process (no HTTP)."""
import time, json, random, sys, os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import ClinicalRecruitmentEnv
from models import Action

TASKS = ["easy_bench", "medium_bench", "hard_bench"]

def heuristic_action(obs, step):
    available = obs.available_patients
    recontact = obs.recontact_candidates
    allocation = obs.allocation_candidates
    sites = obs.site_performance
    hyp = "noise_dominant"

    if allocation and sites:
        best_site = max(sites.keys(), key=lambda s: sites[s].get("conversion_rate", 0) * max(1, sites[s].get("capacity_remaining", 0)))
        return Action(action_type="allocate_to_site", patient_id=allocation[0]["id"], site_id=best_site, hypothesis=hyp, confidence=0.8)
    if recontact:
        return Action(action_type="recontact", patient_id=recontact[0]["id"], hypothesis=hyp, confidence=0.75)
    if available:
        best = max(available, key=lambda p: p.get("eligibility_score", 0) * (1 - p.get("dropout_risk", 0)))
        return Action(action_type="screen_patient", patient_id=best["id"], hypothesis=hyp, confidence=0.7)
    return Action(action_type="adjust_strategy", strategy_change="increase_outreach", hypothesis=hyp, confidence=0.6)


def generate_one_local(args):
    """Generate one trace using local env (no HTTP). Process-safe."""
    task_id, trace_id, max_steps, seed = args
    rng = random.Random(seed)
    env = ClinicalRecruitmentEnv()
    result = env.reset(task=task_id)
    obs = result.observation
    trace = []

    for step in range(max_steps):
        if result.done:
            break

        avail_ids = [p["id"] for p in obs.available_patients[:3]]
        recon_ids = [p["id"] for p in obs.recontact_candidates[:3]]
        alloc_ids = [p["id"] for p in obs.allocation_candidates[:3]]
        site_ids = list(obs.site_performance.keys())[:3]

        obs_text = (f"step={obs.timestamp} budget={obs.budget_remaining} "
                    f"enrolled={obs.enrolled_so_far}/{obs.target_enrollment} "
                    f"available_patients={avail_ids} recontact_candidates={recon_ids} "
                    f"allocation_candidates={alloc_ids} sites={site_ids}")

        action = heuristic_action(obs, step)

        # Randomize 15% of actions
        if rng.random() < 0.15:
            if obs.allocation_candidates and obs.site_performance:
                action = Action(action_type="allocate_to_site",
                                patient_id=rng.choice(obs.allocation_candidates)["id"],
                                site_id=rng.choice(list(obs.site_performance.keys())),
                                hypothesis="noise_dominant", confidence=0.8)
            elif obs.recontact_candidates:
                action = Action(action_type="recontact",
                                patient_id=rng.choice(obs.recontact_candidates)["id"],
                                hypothesis="noise_dominant", confidence=0.7)

        trace.append({"user": obs_text, "assistant": action.model_dump()})

        result = env.step(action)
        obs = result.observation

    enrolled = obs.enrolled_so_far
    target = obs.target_enrollment
    return {"task": task_id, "id": trace_id, "steps": len(trace), "enrolled": enrolled,
            "target": target, "trace": trace}


if __name__ == "__main__":
    # Test different parallelism levels
    for mode, num_workers, num_traces in [
        ("sequential", 1, 50),
        ("threads-4", 4, 50),
        ("threads-8", 8, 50),
        ("threads-16", 16, 50),
        ("processes-4", 4, 50),
        ("processes-8", 8, 50),
    ]:
        args_list = [(TASKS[i % 3], i, 80, 42 + i) for i in range(num_traces)]

        start = time.time()
        results = []

        if mode == "sequential":
            for a in args_list:
                results.append(generate_one_local(a))
        elif mode.startswith("threads"):
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futs = [ex.submit(generate_one_local, a) for a in args_list]
                results = [f.result() for f in as_completed(futs)]
        elif mode.startswith("processes"):
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                results = list(ex.map(generate_one_local, args_list))

        elapsed = time.time() - start
        total_steps = sum(r["steps"] for r in results)
        avg_enrolled = sum(r["enrolled"] for r in results) / len(results)

        print(f"{mode:>15} | {num_traces} traces | {elapsed:6.1f}s | "
              f"{num_traces/elapsed:5.1f} traces/sec | {total_steps/elapsed:6.0f} steps/sec | "
              f"avg_enrolled={avg_enrolled:.1f}")

    # Now test scaling: how many traces can we generate in 60 seconds?
    print("\n--- Scaling test: max traces in 60 seconds (processes-8) ---")
    start = time.time()
    count = 0
    batch_size = 50
    while time.time() - start < 60:
        args_list = [(TASKS[i % 3], count + i, 80, 42 + count + i) for i in range(batch_size)]
        with ProcessPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(generate_one_local, args_list))
        count += batch_size
        elapsed = time.time() - start
        print(f"  {count} traces in {elapsed:.1f}s ({count/elapsed:.1f}/sec)")

    print(f"\nFINAL: {count} traces in 60 seconds = {count/60:.0f} traces/sec")
