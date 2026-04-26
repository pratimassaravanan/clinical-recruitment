#!/usr/bin/env python3
"""Test parallel trace generation speed against live HF Space."""
import time, json, random, httpx
from concurrent.futures import ThreadPoolExecutor, as_completed

ENV_URL = "https://pratimassaravanan-clinical-recruitment.hf.space"
TASKS = ["easy_bench", "medium_bench", "hard_bench"]

SYSTEM_PROMPT = """You are a clinical trial recruitment agent. Output ONLY JSON actions."""

def heuristic_action(obs, step):
    available = obs.get("available_patients", [])
    recontact = obs.get("recontact_candidates", [])
    allocation = obs.get("allocation_candidates", [])
    sites = obs.get("site_performance", {})
    hyp = "noise_dominant"
    
    if allocation and sites:
        best_site = max(sites.keys(), key=lambda s: sites[s].get("conversion_rate", 0) * max(1, sites[s].get("capacity_remaining", 0)))
        return {"action_type": "allocate_to_site", "patient_id": allocation[0]["id"], "site_id": best_site, "hypothesis": hyp, "confidence": 0.8}
    if recontact:
        return {"action_type": "recontact", "patient_id": recontact[0]["id"], "hypothesis": hyp, "confidence": 0.75}
    if available:
        best = max(available, key=lambda p: p.get("eligibility_score", 0) * (1 - p.get("dropout_risk", 0)))
        return {"action_type": "screen_patient", "patient_id": best["id"], "hypothesis": hyp, "confidence": 0.7}
    strats = ["increase_outreach", "relax_criteria", "tighten_criteria"]
    return {"action_type": "adjust_strategy", "strategy_change": strats[step % 3], "hypothesis": hyp, "confidence": 0.6}


def generate_one_trace(task_id, trace_id, max_steps=50, randomize=True):
    """Generate a single trace. Thread-safe -- each thread gets its own httpx client."""
    client = httpx.Client(timeout=30)
    try:
        r = client.post(f"{ENV_URL}/reset", params={"task_id": task_id})
        result = r.json()
        obs = result.get("observation", {})
        messages = []
        
        for step in range(max_steps):
            if result.get("done", False):
                break
            
            avail_ids = [p["id"] for p in obs.get("available_patients", [])[:3]]
            recon_ids = [p["id"] for p in obs.get("recontact_candidates", [])[:3]]
            alloc_ids = [p["id"] for p in obs.get("allocation_candidates", [])[:3]]
            site_ids = list(obs.get("site_performance", {}).keys())[:3]
            
            obs_text = (f"step={obs.get('timestamp')} budget={obs.get('budget_remaining')} "
                        f"enrolled={obs.get('enrolled_so_far')}/{obs.get('target_enrollment')} "
                        f"available_patients={avail_ids} recontact_candidates={recon_ids} "
                        f"allocation_candidates={alloc_ids} sites={site_ids} "
                        f"funnel={obs.get('current_funnel', {})}")
            
            messages.append({"role": "user", "content": obs_text})
            
            action = heuristic_action(obs, step)
            # Add some randomization
            if randomize and random.random() < 0.15:
                choices = []
                if obs.get("available_patients"): choices.append("screen")
                if obs.get("recontact_candidates"): choices.append("recontact")
                if obs.get("allocation_candidates") and obs.get("site_performance"): choices.append("allocate")
                choices.append("strategy")
                pick = random.choice(choices)
                if pick == "allocate" and obs.get("allocation_candidates") and obs.get("site_performance"):
                    action = {"action_type": "allocate_to_site", "patient_id": obs["allocation_candidates"][0]["id"],
                              "site_id": random.choice(list(obs["site_performance"].keys())), "hypothesis": "noise_dominant", "confidence": 0.8}
                elif pick == "recontact" and obs.get("recontact_candidates"):
                    action = {"action_type": "recontact", "patient_id": obs["recontact_candidates"][0]["id"], "hypothesis": "noise_dominant", "confidence": 0.7}
                elif pick == "screen" and obs.get("available_patients"):
                    action = {"action_type": "screen_patient", "patient_id": random.choice(obs["available_patients"])["id"], "hypothesis": "noise_dominant", "confidence": 0.7}
            
            messages.append({"role": "assistant", "content": json.dumps(action)})
            
            r = client.post(f"{ENV_URL}/step", json=action)
            result = r.json()
            obs = result.get("observation", {})
        
        enrolled = obs.get("enrolled_so_far", 0)
        target = obs.get("target_enrollment", 100)
        client.close()
        return {"task": task_id, "trace_id": trace_id, "messages": len(messages), "enrolled": enrolled, "target": target, "trace": messages}
    except Exception as e:
        client.close()
        return {"task": task_id, "trace_id": trace_id, "error": str(e)}


if __name__ == "__main__":
    # Test different thread counts
    for num_threads in [1, 4, 8, 16]:
        random.seed(42)
        num_traces = 16
        tasks_list = [(TASKS[i % 3], i) for i in range(num_traces)]
        
        start = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(generate_one_trace, task, tid, 50, True): (task, tid) 
                       for task, tid in tasks_list}
            for future in as_completed(futures):
                results.append(future.result())
        
        elapsed = time.time() - start
        successes = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]
        total_msgs = sum(r.get("messages", 0) for r in successes)
        avg_enrolled = sum(r.get("enrolled", 0) for r in successes) / max(1, len(successes))
        
        print(f"\n{'='*50}")
        print(f"Threads: {num_threads} | Traces: {num_traces}")
        print(f"Time: {elapsed:.1f}s | {num_traces/elapsed:.2f} traces/sec")
        print(f"Success: {len(successes)} | Errors: {len(errors)}")
        print(f"Total messages: {total_msgs} | Avg enrolled: {avg_enrolled:.1f}")
        if errors:
            print(f"Error sample: {errors[0].get('error', '')[:100]}")
