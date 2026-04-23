"""Deploy to Hugging Face Spaces via huggingface_hub API."""

import os
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "pratimassaravanan/clinical-recruitment")
HF_REPO_TYPE = "space"

FILES = [
    "app.py",
    "env.py",
    "models.py",
    "graders.py",
    "load_traces.py",
    "openenv_adapter.py",
    "inference.py",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "notebooks/training_grpo_openenv.ipynb",
    ".gitignore",
    ".env.example",
    "validate.py",
    "docs/communication/adaptive_clinical_recruitment_presentation.html",
    "docs/communication/adaptive_clinical_recruitment_posters.pdf",
    "docs/communication/poster_level_1_beginner.png",
    "docs/communication/poster_level_2_mechanics.png",
    "docs/communication/poster_level_3_implementation.png",
    "docs/communication/poster_level_4_results.png",
    "docs/communication/poster_level_5_reviewer.png",
    "docs/communication/protocol_cartography_design_philosophy.md",
    "architecture.png",
    "requirements-research.txt",
    "docs/research.md",
    "docs/theme2_alignment.md",
    "data/sweep_results/neurips_report.md",
    "data/sweep_results/neurips_report.json",
    "data/sweep_results/significance_tests.json",
    "data/sweep_results/sweep_results.csv",
    "data/sweep_results/sweep_results.json",
    "docs/images/benchmark_scores.svg",
    "docs/images/enrollment_budget_tradeoff.svg",
    "docs/images/long_horizon_indicators.svg",
    "docs/images/environment_architecture.png",
    "docs/images/agent_architectures.png",
    "docs/images/training_pipeline.png",
    "docs/images/agent_comparison.png",
    "docs/images/agent_comparison.svg",
    "docs/images/seed_heatmap.png",
    "docs/images/seed_heatmap.svg",
    "docs/images/score_boxplot.png",
    "docs/images/score_boxplot.svg",
    "data/research_runs.csv",
    "data/research_summary.csv",
    "data/leaderboard.csv",
    "research/__init__.py",
    "research/policies.py",
    "research/runner.py",
    "research/methods/__init__.py",
    "research/methods/registry.py",
    "research/methods/README.md",
    "experiments/__init__.py",
    "experiments/run_research.py",
    "scripts/generate_charts.py",
    "scripts/generate_docs_diagrams.py",
    "scripts/generate_communication_posters.py",
    "server/__init__.py",
    "server/app.py",
]


def deploy():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN is required. Set it as an environment variable.")

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            space_sdk="docker",
            exist_ok=True,
        )
        print(f"Repo '{HF_REPO_ID}' ready.")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Upload files
    for filepath in FILES:
        if not os.path.isfile(filepath):
            print(f"  SKIP {filepath} (not found)")
            continue
        try:
            upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filepath,
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                token=HF_TOKEN,
            )
            print(f"  UPLOADED {filepath}")
        except Exception as e:
            print(f"  ERROR {filepath}: {e}")

    print(f"\nDeployed to: https://huggingface.co/spaces/{HF_REPO_ID}")


if __name__ == "__main__":
    deploy()
