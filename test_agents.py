"""Tests for real HCAPO, MiRA, KLong, and MemexRL agent implementations."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from training.neural_policy import (
    ActorCritic,
    NeuralNetwork,
    STATE_DIM,
    extract_state_features,
)
from research.methods.hcapo_agent import HCAPOAgent
from research.methods.mira_agent import MiRAAgent, MiRACritic
from research.methods.klong_agent import KLongAgent
from research.methods.memex_agent import MemexRLAgent


checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 60)
print("AGENT IMPLEMENTATION TESTS")
print("=" * 60)

# Test NeuralNetwork
print("\n1. NeuralNetwork tests")
nn = NeuralNetwork(layer_sizes=[10, 8, 4])
check("nn initializes weights", len(nn.weights) == 2)
check("nn weights correct shape", nn.weights[0].shape == (10, 8))
x = np.random.randn(10).astype(np.float32)
out, activations = nn.forward(x)
check("nn forward produces output", out.shape == (4,))
check("nn forward produces activations", len(activations) == 3)

# Test backward
grad = np.random.randn(4).astype(np.float32)
w_grads, b_grads = nn.backward(activations, grad)
check("nn backward produces weight grads", len(w_grads) == 2)
check("nn backward grad shapes match", w_grads[0].shape == nn.weights[0].shape)

# Test ActorCritic
print("\n2. ActorCritic tests")
ac = ActorCritic(state_dim=STATE_DIM, action_dim=8)
check("ac initializes actor", ac.actor is not None)
check("ac initializes critic", ac.critic is not None)
state = np.random.randn(STATE_DIM).astype(np.float32)
probs = ac.get_action_probs(state)
check("ac produces action probs", len(probs) == 8)
check("ac probs sum to 1", abs(probs.sum() - 1.0) < 0.01)
value = ac.get_value(state)
check("ac produces value", isinstance(value, float))

# Test HCAPO
print("\n3. HCAPOAgent tests")
hcapo = HCAPOAgent()
check("hcapo initializes", hcapo is not None)
check("hcapo has executor", hcapo.executor is not None)
check("hcapo has planner", hcapo.planner is not None)
obs = {"enrolled_so_far": 10, "target_enrollment": 100, "budget_remaining": 50000}
action, info = hcapo.select_action(obs)
check("hcapo selects action", 0 <= action < 8)
check("hcapo returns subgoal info", "subgoal_type" in info)

# Test hindsight extraction
trajectory = [
    {"enrolled": 0, "milestone_potential": 0.2, "budget_remaining": 100000},
    {"enrolled": 5, "milestone_potential": 0.4, "budget_remaining": 95000},
    {"enrolled": 10, "milestone_potential": 0.5, "budget_remaining": 90000},
]
goals = hcapo._extract_hindsight_goals(trajectory)
check("hcapo extracts hindsight goals", len(goals) > 0)

# Test MiRA
print("\n4. MiRAAgent tests")
mira = MiRAAgent()
check("mira initializes", mira is not None)
check("mira has policy", mira.policy is not None)
check("mira has potential critic", mira.potential_critic is not None)
action, info = mira.select_action(obs)
check("mira selects action", 0 <= action < 8)
check("mira returns potential", "potential" in info)

# Test potential shaping
obs1 = {"enrolled_so_far": 10, "target_enrollment": 100, "budget_remaining": 90000}
obs2 = {"enrolled_so_far": 15, "target_enrollment": 100, "budget_remaining": 85000}
shaped_reward, breakdown = mira.compute_shaped_reward(obs1, obs2, 0.5, step=10)
check("mira computes shaped reward", isinstance(shaped_reward, float))
check("mira returns breakdown", "original" in breakdown and "shaped" in breakdown)

# Test KLong
print("\n5. KLongAgent tests")
klong = KLongAgent()
check("klong initializes", klong is not None)
check("klong has policy", klong.policy is not None)
check("klong has temporal abstraction", klong.temporal_abstraction is not None)
klong.reset()
action, info = klong.select_action(obs)
check("klong selects action", 0 <= action < 8)
check("klong tracks history", len(klong.state_history) > 0)

# Test segmentation
trajectory = [{"obs": obs, "action": i % 8, "reward": 0.1, "done": False} for i in range(45)]
trajectory[-1]["done"] = True
segments = klong._segment_trajectory(trajectory)
check("klong segments trajectory", len(segments) > 0)
check("klong segments have context", segments[0].context_embedding is not None)

# Test MemexRL
print("\n6. MemexRLAgent tests")
memex = MemexRLAgent()
check("memex initializes", memex is not None)
check("memex has memory", memex.memory is not None)
check("memex has policy", memex.policy is not None)
memex.reset()
action, info = memex.select_action(obs, step=0)
check("memex selects action", 0 <= action < 8)
check("memex returns memory info", "memory_size" in info)

# Test memory write/read
state = extract_state_features(obs)
written = memex.memory.write(state, action, 0.5, step=1, obs=obs, force=True)
check("memex writes to memory", written)
check("memex memory has entry", len(memex.memory.entries) > 0)
retrieved, entries = memex.memory.read(state, top_k=3, current_step=2)
check("memex reads from memory", retrieved.shape[0] == memex.memory.value_dim)

# Test save/load cycle
print("\n7. Save/Load tests")
import tempfile
import json

with tempfile.TemporaryDirectory() as tmpdir:
    # HCAPO save/load
    hcapo_path = os.path.join(tmpdir, "hcapo.json")
    hcapo.save(hcapo_path)
    hcapo_loaded = HCAPOAgent.load(hcapo_path)
    check("hcapo save/load works", hcapo_loaded is not None)

    # MiRA save/load
    mira_path = os.path.join(tmpdir, "mira.json")
    mira.save(mira_path)
    mira_loaded = MiRAAgent.load(mira_path)
    check("mira save/load works", mira_loaded is not None)

    # KLong save/load
    klong_path = os.path.join(tmpdir, "klong.json")
    klong.save(klong_path)
    klong_loaded = KLongAgent.load(klong_path)
    check("klong save/load works", klong_loaded is not None)

    # MemexRL save/load
    memex_path = os.path.join(tmpdir, "memex.json")
    memex.save(memex_path)
    memex_loaded = MemexRLAgent.load(memex_path)
    check("memex save/load works", memex_loaded is not None)

print("\n" + "=" * 60)
passed = sum(1 for _, s in checks if s == "PASS")
failed = sum(1 for _, s in checks if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(checks)} checks")
if failed == 0:
    print("ALL CHECKS PASSED!")
else:
    print("SOME CHECKS FAILED - review above")
print("=" * 60)
