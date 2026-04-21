"""Focused tests for research/training helper modules."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research import FrontierReplayBuffer, discover_goals, infer_skills, rank_preferences
from research.methods import compute_step_advantages, summarize_oversight
from research.privacy import anonymize_patient_rows
from research.world_models import predict_site_value, CounterfactualSimulator, StateSnapshot
from training.async_rl import run_async_training
from training.curriculum import ThompsonCurriculum, confidence_curriculum_schedule
from env import ClinicalRecruitmentEnv


checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 60)
print("RESEARCH MODULE TESTS")
print("=" * 60)

history = [
    {"step": 0, "action": "screen_patient", "reward": 0.2, "milestone_potential": 0.4},
    {"step": 1, "action": "adjust_strategy", "reward": -0.1, "error": "regulatory_hold_active"},
]

buffer = FrontierReplayBuffer(capacity=2)
for item in history:
    buffer.add(item)
check("frontier replay stores items", len(buffer.sample()) == 2)

advantages = compute_step_advantages(history)
check("salt advantage rows returned", len(advantages) == 2)

oversight = summarize_oversight(history)
check("oversight ratio present", "oversight_ratio" in oversight)

skills = infer_skills({"patient_memory_summary": {"followup_due": 1}, "active_constraints": {}})
check("skill inference returns conversion skill", "conversion_rescue" in skills)

goals = discover_goals([{"avg_dropped": 20}])
check("goal discovery finds retention goal", "improve_retention" in goals)

ranked = rank_preferences([
    {"final_score": 0.4, "token_efficiency_score": 0.9},
    {"final_score": 0.7, "token_efficiency_score": 0.7},
])
check("preferences are ranked descending", ranked[0]["final_score"] == 0.7)

anon = anonymize_patient_rows([{"id": "P-1", "age": 30}])
check("privacy anonymizes id", anon[0]["id"].startswith("anon-"))

site_value = predict_site_value(
    {
        "conversion_rate": 0.7,
        "retention_rate": 0.9,
        "avg_wait_days": 3,
        "capacity_remaining": 10,
    }
)
check("site world model returns positive value", site_value > 0)

schedule = confidence_curriculum_schedule("medium_bench")
check("confidence curriculum emits staged tasks", len(schedule) == 3)

curriculum = ThompsonCurriculum(seed=1)
task_id = curriculum.sample_task()
curriculum.update(task_id, 0.8)
check("thompson curriculum updates priors", curriculum.priors[task_id][0] > 1.0)

async_rows = run_async_training([["medium_bench_stage_30"], ["medium_bench_stage_90"]], epochs=1)
check("async training returns rows", len(async_rows) == 2)

# Counterfactual simulator tests
print("\n9. Counterfactual Simulator")
cf_sim = CounterfactualSimulator(lookahead_steps=5, num_rollouts_per_branch=2, seed=42)
check("counterfactual simulator initializes", cf_sim is not None)

# Create a test environment and get a snapshot
test_env = ClinicalRecruitmentEnv()
test_env.reset("easy_bench")
# Take a few steps to build state
from models import Action
test_env.step(Action(action_type="screen_patient"))
test_env.step(Action(action_type="screen_patient"))

snapshot = cf_sim.snapshot_from_env(test_env)
check("snapshot captures enrolled", snapshot.enrolled >= 0)
check("snapshot captures budget", snapshot.budget_remaining > 0)
check("snapshot captures step", snapshot.step == 2)

# Test rollout from snapshot
rollout = cf_sim.rollout_from_snapshot(snapshot, ["screen_patient", "allocate_to_site", "screen_patient"])
check("rollout returns result", rollout is not None)
check("rollout has actions", len(rollout.actions) <= 3)
check("rollout tracks enrolled", rollout.final_enrolled >= 0)

# Test branch generation
branches = cf_sim.generate_action_branches(snapshot)
check("branches generated", len(branches) >= 3)
check("aggressive_screen branch exists", "aggressive_screen" in branches)
check("allocation_focus branch exists", "allocation_focus" in branches)

# Test quick counterfactual
quick_cf = cf_sim.quick_counterfactual(test_env, "screen_patient")
check("quick counterfactual returns dict", isinstance(quick_cf, dict))
check("quick counterfactual has proposed_action", "proposed_action" in quick_cf)
check("quick counterfactual has regret", "regret" in quick_cf)
check("quick counterfactual has confidence", quick_cf["confidence"] in ["high", "medium", "low"])

# Test full analysis
analysis = cf_sim.run_counterfactual_analysis(
    test_env, 
    ["screen_patient", "screen_patient", "allocate_to_site"]
)
check("analysis has actual outcome", analysis.actual_outcome is not None)
check("analysis has counterfactual outcomes", len(analysis.counterfactual_outcomes) > 0)
check("analysis has regret", analysis.regret >= 0)
check("analysis has recommended action", len(analysis.recommended_action) > 0)

# Test Pareto optimal branches
pareto = cf_sim.get_pareto_optimal_branches(test_env)
check("pareto branches returned", len(pareto) > 0)
check("pareto has enrolled field", "enrolled" in pareto[0])
check("pareto has budget field", "budget_remaining" in pareto[0])

# Multi-agent site negotiation tests
print("\n10. Multi-Agent Site Negotiation")
from research.methods.site_agents import (
    SiteAgent, MultiAgentNegotiator, NegotiationOutcome
)

# Test site agent
site_agent = SiteAgent(
    site_id="site_A",
    true_capacity=50,
    true_min_payment=500.0,
    seed=42
)
check("site agent initializes", site_agent is not None)
check("site agent has state", site_agent.state.site_id == "site_A")
check("site agent has capacity", site_agent.state.available_capacity() == 50)

public_info = site_agent.get_public_info()
check("site agent exposes public info", "site_id" in public_info)
check("site agent exposes capacity estimate", "available_capacity_estimate" in public_info)

# Test negotiator
sites_config = {
    "site_A": {"capacity_remaining": 50, "conversion_rate": 0.7, "retention_rate": 0.85},
    "site_B": {"capacity_remaining": 30, "conversion_rate": 0.8, "retention_rate": 0.90},
}
negotiator = MultiAgentNegotiator(sites=sites_config, seed=42)
check("negotiator initializes", negotiator is not None)
check("negotiator has sites", len(negotiator.site_agents) == 2)

# Test offer creation
offer = negotiator.make_offer(
    to_site="site_A",
    capacity_requested=10,
    payment_offered=600.0,
)
check("offer created", offer is not None)
check("offer has correct site", offer.to_agent == "site_A")

# Test offer submission
outcome, counter = negotiator.submit_offer(offer)
check("offer submitted", outcome in [NegotiationOutcome.ACCEPTED, NegotiationOutcome.COUNTEROFFER, NegotiationOutcome.PENDING])

# Test full negotiation
result = negotiator.negotiate_capacity(
    site_id="site_B",
    desired_capacity=10,
    budget_per_enrollment=700.0,
    max_rounds=3,
)
check("negotiation returns result", "success" in result)
check("negotiation has rounds", "rounds" in result)
check("negotiation has history", len(result.get("history", [])) > 0)

# Test recommendations
recommendations = negotiator.get_site_recommendations(
    desired_capacity=20,
    budget=10000.0,
)
check("recommendations returned", len(recommendations) == 2)
check("recommendations have scores", "score" in recommendations[0])

# Test market state
market_state = negotiator.get_market_state()
check("market state has num_sites", market_state["num_sites"] == 2)
check("market state has capacity", "total_available_capacity" in market_state)

# Test serialization
state_dict = negotiator.to_dict()
check("serialization works", "sites" in state_dict)

restored = MultiAgentNegotiator.from_dict(state_dict)
check("deserialization works", len(restored.site_agents) == 2)

# Statistical significance tests
print("\n11. Statistical Significance Tests")
from experiments.reproducibility import (
    bootstrap_ci, cohens_d, paired_t_test, wilcoxon_signed_rank,
    bonferroni_correction, holm_bonferroni_correction, ReproducibilityReport
)
import numpy as np

# Test bootstrap CI
test_data = np.array([0.5, 0.6, 0.55, 0.58, 0.52, 0.61, 0.57])
ci_lower, ci_upper = bootstrap_ci(test_data, n_bootstrap=100, seed=42)
check("bootstrap CI returns tuple", isinstance(ci_lower, float) and isinstance(ci_upper, float))
check("bootstrap CI lower < upper", ci_lower < ci_upper)
check("bootstrap CI contains mean", ci_lower <= np.mean(test_data) <= ci_upper)

# Test Cohen's d
group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
d = cohens_d(group1, group2)
check("cohens_d returns float", isinstance(d, float))
check("cohens_d has correct sign", d < 0)  # group1 mean < group2 mean

# Test paired t-test
t_result = paired_t_test(group1, group2)
check("t_test returns result", t_result.test_name == "paired_t_test")
check("t_test has p_value", 0 <= t_result.p_value <= 1)
check("t_test has significance flags", hasattr(t_result, "significant_at_05"))

# Test Wilcoxon
w_result = wilcoxon_signed_rank(group1, group2)
check("wilcoxon returns result", w_result.test_name == "wilcoxon_signed_rank")
check("wilcoxon has p_value", 0 <= w_result.p_value <= 1)

# Test Bonferroni correction
p_values = [0.01, 0.03, 0.05, 0.10]
bonf_results = bonferroni_correction(p_values, alpha=0.05)
check("bonferroni returns results", len(bonf_results) == 4)
check("bonferroni first is significant", bonf_results[0][1] == True)  # 0.01 < 0.0125
check("bonferroni last not significant", bonf_results[-1][1] == False)

# Test Holm-Bonferroni
holm_results = holm_bonferroni_correction(p_values, alpha=0.05)
check("holm returns results", len(holm_results) == 4)
check("holm has rank", holm_results[0][2] >= 1)

# Test ReproducibilityReport initialization
report = ReproducibilityReport(seeds=[1, 2, 3])
check("report initializes", report is not None)
check("report has seeds", len(report.seeds) == 3)

# 9. Strict subgoal execution + frontier replay (enhanced)
print("\n12. Strict Subgoal Execution + Enhanced Replay")
from research.replay import (
    Subgoal, SubgoalSequence, StrictSubgoalExecutor,
    FrontierReplayBuffer, ReplayDrivenTrainer
)

subgoal = Subgoal("sg_test", "Test Subgoal", "enrollment_progress", 0.5)
check("subgoal initializes", subgoal.subgoal_id == "sg_test")
check("subgoal not achieved initially", not subgoal.achieved)

state_met = {"enrollment_progress": 0.6}
completed = subgoal.check_completion(state_met)
check("subgoal completion detected", completed and subgoal.achieved)

executor = StrictSubgoalExecutor()
check("executor initializes", executor is not None)
check("executor has sequence", len(executor.sequence.subgoals) > 0)

bonus, completed, info = executor.step({"enrollment_progress": 0.3, "screened_count": 15}, "screen_patient", 5)
check("executor step returns info", "current_subgoal" in info)

valid_actions = executor.get_valid_actions({"enrollment_progress": 0.5}, ["screen_patient", "allocate_to_site", "stop"])
check("executor filters actions", len(valid_actions) > 0)

# Enhanced buffer
buffer = FrontierReplayBuffer(capacity=50, priority_keys=["reward", "milestone_potential"])
buffer.add({"reward": 0.5, "milestone_potential": 0.3, "state": {}})
buffer.add({"reward": 0.8, "milestone_potential": 0.5, "state": {}})
check("enhanced buffer stores items", len(buffer) == 2)

sampled = buffer.sample_random(2, seed=42)
check("buffer random sampling works", len(sampled) <= 2)

# 10. Progressive difficulty curriculum (enhanced)
print("\n13. Progressive Difficulty Curriculum")
from training.curriculum import (
    ProgressiveDifficultyCurriculum, EarlyMistakeRecoveryCurriculum,
    AdaptiveCurriculumManager, RecoveryScenario
)

prog_curriculum = ProgressiveDifficultyCurriculum(seed=42)
check("progressive curriculum initializes", prog_curriculum is not None)
check("progressive curriculum has levels", len(prog_curriculum.levels) > 0)

task = prog_curriculum.sample_task()
check("progressive curriculum samples task", task is not None)

result = prog_curriculum.record_episode("easy_bench_stage_30", 0.5)
check("progressive curriculum records episode", "level" in result)

recovery = EarlyMistakeRecoveryCurriculum(seed=42)
check("recovery curriculum initializes", recovery is not None)

scenario = recovery.sample_scenario()
check("recovery scenario sampled", scenario is not None and scenario.scenario_id is not None)

adaptive = AdaptiveCurriculumManager(seed=42)
task, scenario = adaptive.sample_task()
check("adaptive manager samples task", task is not None)

# 13-17. Advanced features
print("\n14. Advanced Long-Horizon Features")
from research.advanced_features import (
    TokenUsageTracker, MultiPhaseRewardDecomposer, PatientMemoryGraph,
    SiteWorldModel, ParetoController, SALTAdvantageComputer, SkillWorldModel,
    AsyncRLCoordinator, RegulatoryEventSimulator, PatientEngagementSimulator,
    CurriculumLogger, HierarchicalOversightSystem, FederatedPrivacySimulator,
    PreferenceAligner, EvolvingSkillLibrary, LongHorizonUncertaintyQuantifier,
    CarbonAwareScaler, CrossDomainTransfer
)

# Token tracking
tracker = TokenUsageTracker(total_budget=10000)
tokens = tracker.record_action("screen_patient")
check("token tracker records usage", tracker.used_tokens == tokens)
check("token tracker calculates efficiency", 0 <= tracker.get_efficiency_score() <= 1)

# Multi-phase rewards
decomposer = MultiPhaseRewardDecomposer()
decomposer.update_objectives({"screened_count": 30, "enrollment_progress": 0.3})
total, breakdown = decomposer.compute_phase_reward()
check("phase decomposer computes rewards", total >= 0)
check("phase decomposer has breakdown", len(breakdown) > 0)

# Patient memory graph
graph = PatientMemoryGraph()
node = graph.add_patient("P1", {"age": 45, "site_id": "A"})
check("patient graph adds node", "P1" in graph.nodes)
graph.add_patient("P2", {"age": 42, "site_id": "A"})
related = graph.get_related_patients("P1")
check("patient graph finds related", len(related) > 0)

# Site world model
site_model = SiteWorldModel(seed=42)
site_model.record_observation("site_A", {"conversion_rate": 0.7, "capacity": 20})
site_model.record_observation("site_A", {"conversion_rate": 0.8, "capacity": 18})
prediction = site_model.predict_performance("site_A", steps_ahead=5)
check("site model predicts", "conversion_rate" in prediction)

# Pareto controller
pareto = ParetoController(objectives=["enrollment", "budget"])
added = pareto.add_point({"enrollment": 50, "budget": 5000}, ["screen", "allocate"])
check("pareto controller adds point", added)
summary = pareto.get_frontier_summary()
check("pareto has frontier", summary["frontier_size"] > 0)

# SALT advantages
salt = SALTAdvantageComputer()
salt.add_transition({"enrollment_progress": 0.2}, "screen", 0.1, {"enrollment_progress": 0.25}, False)
advantages = salt.compute_advantages([{"state": {"enrollment_progress": 0.2}, "reward": 0.1, "done": False}])
check("SALT computes advantages", len(advantages) > 0)

# Skill world model
skill_model = SkillWorldModel(seed=42)
applicable = skill_model.get_applicable_skills({"budget_ratio": 0.5, "screening_backlog": 20})
check("skill model finds applicable skills", len(applicable) > 0)

# Async RL
async_coord = AsyncRLCoordinator(num_workers=2, seed=42)
async_coord.init_workers(["easy_bench", "medium_bench"])
check("async coordinator initializes workers", len(async_coord.worker_states) == 2)

# Regulatory events
reg_sim = RegulatoryEventSimulator(seed=42)
effects = reg_sim.get_active_effects()
check("regulatory simulator initializes", isinstance(effects, dict))

# Patient engagement
engagement = PatientEngagementSimulator(seed=42)
engagement.init_patient("P1")
result = engagement.simulate_contact("P1", current_step=5)
check("engagement simulator contacts patient", "responded" in result)

# Curriculum logger
logger = CurriculumLogger()
logger.log_event("test_event", {"key": "value"}, step=10)
check("curriculum logger logs events", len(logger.logs) > 0)

# Hierarchical oversight
oversight = HierarchicalOversightSystem(seed=42)
decision = oversight.submit_action("allocate_to_site", risk_score=0.5, context={})
check("oversight reviews action", "approved" in decision)

# Federated privacy
federated = FederatedPrivacySimulator(num_sites=3, epsilon=1.0, seed=42)
federated.add_local_data("site_0", {"enrollment_rate": 0.5})
avg = federated.federated_average()
check("federated simulator computes average", isinstance(avg, dict))

# Preference alignment
aligner = PreferenceAligner(seed=42)
aligner.add_preference([{"reward": 0.5}], [{"reward": 0.3}], "a")
aligner.update_reward_model()
check("preference aligner updates model", len(aligner.reward_model) > 0)

# Evolving skills
evolving = EvolvingSkillLibrary(seed=42)
from research.advanced_features import Skill
evolving.add_skill(Skill("test_skill", "Test", ["screen"], {}, {"enrollment": 0.1}))
evolving.record_execution("test_skill", True, {"enrollment": 0.12})
check("evolving library records execution", len(evolving.execution_log) > 0)

# Uncertainty quantification
uncertainty = LongHorizonUncertaintyQuantifier(seed=42)
estimate = uncertainty.estimate_uncertainty({"screening_backlog": 30, "enrolled": 20}, horizon=30)
check("uncertainty quantifier estimates", estimate.total > 0)
check("uncertainty has decomposition", estimate.epistemic >= 0 and estimate.aleatoric >= 0)

# Carbon-aware scaling
carbon = CarbonAwareScaler(carbon_budget_gco2=1000)
co2 = carbon.record_compute("training", duration_seconds=60, gpu_power_watts=200)
check("carbon scaler tracks compute", co2 > 0)
stats = carbon.get_carbon_stats()
check("carbon scaler has stats", "total_gco2" in stats)

# Cross-domain transfer
transfer = CrossDomainTransfer.create_clinical_to_marketing_transfer()
check("cross-domain transfer creates mapping", transfer.target_domain is not None)
transferred_state = transfer.transfer_state({"enrollment_progress": 0.5})
check("cross-domain transfers state", "conversion_rate" in transferred_state)

print("\n" + "=" * 60)
passed = sum(1 for _, s in checks if s == "PASS")
failed = sum(1 for _, s in checks if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(checks)} checks")
if failed == 0:
    print("ALL CHECKS PASSED!")
else:
    print("SOME CHECKS FAILED - review above")
print("=" * 60)
