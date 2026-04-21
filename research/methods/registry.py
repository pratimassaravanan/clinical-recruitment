"""Repository method registry for long-horizon research integrations.

The `paper` field is a lightweight provenance note and should not be treated as a
verified external citation without independent validation.
"""

METHOD_REGISTRY = [
    {
        "id": "hcapo",
        "name": "HCAPO",
        "paper": "repo:hcapo_agent",
        "status": "implemented_in_repo",
        "focus": "hierarchical constrained planning and optimization",
    },
    {
        "id": "mira",
        "name": "MiRA",
        "paper": "repo:mira_agent",
        "status": "implemented_in_repo",
        "focus": "subgoal-driven long-horizon reasoning",
    },
    {
        "id": "klong",
        "name": "KLong",
        "paper": "repo:klong_agent",
        "status": "implemented_in_repo",
        "focus": "long-context memory and temporal credit assignment",
    },
    {
        "id": "plan_and_act",
        "name": "Plan-and-Act",
        "paper": "repo:plan_and_act_runtime",
        "status": "scaffolded",
        "focus": "planner/executor decomposition for long-horizon tasks",
    },
    {
        "id": "memex_rl",
        "name": "MemexRL",
        "paper": "repo:memex_agent",
        "status": "implemented_in_repo",
        "focus": "episodic memory retrieval for long-horizon RL agents",
    },
    {
        "id": "salt",
        "name": "SALT",
        "paper": "internal-step-advantage-scaffold",
        "status": "scaffolded",
        "focus": "step-level trajectory advantage estimation",
    },
    {
        "id": "oversight",
        "name": "Oversight",
        "paper": "internal-oversight-scaffold",
        "status": "scaffolded",
        "focus": "multi-agent hierarchical review and recovery",
    },
]
