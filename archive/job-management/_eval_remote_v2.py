#!/usr/bin/env python3
"""Full eval: load LoRA from HuggingFace Hub, run against HF Space."""
import os, json, re, torch, warnings, httpx

warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel

print("Loading model + LoRA from HuggingFace Hub...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
    max_seq_length=768, load_in_4bit=True, dtype=None)

from peft import PeftModel
model = PeftModel.from_pretrained(model, "pratimassaravanan/clinical-qwen3-4b-sft-lora")
model.generation_config.max_length = None
FastLanguageModel.for_inference(model)
print("Model loaded!")

def apply_chat_template(messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)

SYSTEM_PROMPT = """You are a clinical trial recruitment agent. Output ONLY a JSON action object.

Priority rules:
1. If allocation_candidates exist AND sites have capacity: allocate_to_site (ENROLLS patients - highest value)
2. If recontact_candidates exist: recontact (converts to consent)
3. If available_patients exist: screen_patient (pick highest eligibility_score * (1 - dropout_risk))
4. Otherwise: adjust_strategy with increase_outreach

Hypothesis: Read world_type from the observation. Set hypothesis to match:
- world_type="noise" -> hypothesis="noise_dominant"
- world_type="site_bias" -> hypothesis="site_bias"
- world_type="dropout" -> hypothesis="dropout_dominant"
Set confidence=0.9. NEVER change hypothesis mid-episode.

For allocate_to_site: pick site with highest conversion_rate * capacity_remaining.
Use the EXACT patient_id and site_id values from the state."""

ENV_URL = "https://pratimassaravanan-clinical-recruitment.hf.space"
TASKS = ["easy_bench", "medium_bench", "hard_bench"]

def try_parse_json(resp):
    resp_clean = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
    for pattern in [r'\{[^{}]*"action_type"\s*:\s*"[^"]+?"[^{}]*\}', r'```json\s*(\{.*?\})\s*```']:
        m = re.search(pattern, resp_clean, re.DOTALL)
        if m:
            try:
                candidate = m.group(1) if m.lastindex else m.group(0)
                parsed = json.loads(candidate)
                if "action_type" in parsed:
                    return parsed
            except:
                continue
    return None

def heuristic_fallback(obs):
    wt = obs.get("world_type", "noise")
    hyp_map = {"noise": "noise_dominant", "site_bias": "site_bias", "dropout": "dropout_dominant"}
    _H, _C = hyp_map.get(wt, "noise_dominant"), 0.9
    sites = obs.get("site_performance", {})
    if obs.get("allocation_candidates") and sites:
        best = max(sites.keys(), key=lambda s: sites[s].get("conversion_rate", 0) * max(1, sites[s].get("capacity_remaining", 0)))
        return {"action_type": "allocate_to_site", "patient_id": obs["allocation_candidates"][0]["id"], "site_id": best, "hypothesis": _H, "confidence": _C}
    if obs.get("recontact_candidates"):
        return {"action_type": "recontact", "patient_id": obs["recontact_candidates"][0]["id"], "hypothesis": _H, "confidence": _C}
    if obs.get("available_patients"):
        best = max(obs["available_patients"], key=lambda p: p.get("eligibility_score", 0) * (1 - p.get("dropout_risk", 0)))
        return {"action_type": "screen_patient", "patient_id": best["id"], "hypothesis": _H, "confidence": _C}
    return {"action_type": "adjust_strategy", "strategy_change": "increase_outreach", "hypothesis": _H, "confidence": _C}

def run_eval(task, n=50):
    client = httpx.Client(timeout=30)
    r = client.post(f"{ENV_URL}/reset", params={"task_id": task})
    result = r.json()
    obs = result.get("observation", {})
    total_r, steps, json_ok, json_fail = 0.0, 0, 0, 0
    action_counts = {}

    for _ in range(n):
        if result.get("done", False):
            break
        avail_ids = [p["id"] for p in obs.get("available_patients", [])[:3]]
        recon_ids = [p["id"] for p in obs.get("recontact_candidates", [])[:3]]
        alloc_ids = [p["id"] for p in obs.get("allocation_candidates", [])[:3]]
        site_ids = list(obs.get("site_performance", {}).keys())[:3]
        obs_text = (f"step={obs.get('timestamp')} budget={obs.get('budget_remaining')} "
                    f"enrolled={obs.get('enrolled_so_far')}/{obs.get('target_enrollment')} "
                    f"world_type={obs.get('world_type', 'noise')} "
                    f"available_patients={avail_ids} recontact_candidates={recon_ids} "
                    f"allocation_candidates={alloc_ids} sites={site_ids} "
                    f"funnel={obs.get('current_funnel', {})}")
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs_text}]
        input_ids = apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        max_length=768, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        if steps < 3 or steps % 15 == 0:
            print(f"  step {steps}: {resp[:120]}")

        parsed = try_parse_json(resp)
        if parsed:
            json_ok += 1
            action = parsed
            if "hypothesis" not in action:
                wt = obs.get("world_type", "noise")
                hyp_map = {"noise": "noise_dominant", "site_bias": "site_bias", "dropout": "dropout_dominant"}
                action["hypothesis"] = hyp_map.get(wt, "noise_dominant")
            if "confidence" not in action:
                action["confidence"] = 0.9
        else:
            json_fail += 1
            action = heuristic_fallback(obs)

        try:
            action_counts[action["action_type"]] = action_counts.get(action["action_type"], 0) + 1
            r = client.post(f"{ENV_URL}/step", json=action)
            result = r.json()
            obs = result.get("observation", {})
            total_r += result.get("reward", 0)
            steps += 1
        except Exception as e:
            print(f"  error: {e}")
            break

    client.close()
    total = json_ok + json_fail
    rate = json_ok / max(1, total)
    print(f"  actions: {action_counts}")
    print(f"  JSON parse: {json_ok}/{total} ({rate:.0%})")
    return {"task": task, "steps": steps, "total_reward": round(total_r, 4),
            "enrolled": obs.get("enrolled_so_far", 0), "target": obs.get("target_enrollment", 100),
            "json_parse_rate": round(rate, 4), "action_dist": action_counts}

print("\n" + "=" * 60 + "\nFULL EVALUATION\n" + "=" * 60)
results = {}
for t in TASKS:
    print(f"\n[{t}]")
    r = run_eval(t, n=50)
    results[t] = r
    print(f"  RESULT: enrolled={r['enrolled']}/{r['target']} reward={r['total_reward']} json={r['json_parse_rate']:.0%}")

print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
for t, r in results.items():
    print(f"  {t}: enrolled={r['enrolled']}/{r['target']} reward={r['total_reward']} json_parse={r['json_parse_rate']:.0%}")

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to eval_results.json")
