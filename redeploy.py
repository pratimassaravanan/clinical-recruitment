"""Quick redeploy: upload only commonly-changed files to HF Spaces."""

import os
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "pratimassaravanan/clinical-recruitment")

QUICK_FILES = [
    "README.md",
    "inference.py",
    "app.py",
    "env.py",
    "openenv_adapter.py",
    "graders.py",
    "openenv.yaml",
    "notebooks/training_grpo_openenv.ipynb",
    "data/sweep_results/benchmark_report.md",
    "data/sweep_results/benchmark_report.json",
    "data/sweep_results/significance_tests.json",
    "docs/theme2_alignment.md",
    "docs/communication/adaptive_clinical_recruitment_presentation.html",
    "docs/communication/adaptive_clinical_recruitment_posters.pdf",
    "docs/communication/poster_level_1_beginner.png",
    "docs/communication/poster_level_2_mechanics.png",
    "docs/communication/poster_level_3_implementation.png",
    "docs/communication/poster_level_4_results.png",
    "docs/communication/poster_level_5_reviewer.png",
    "docs/images/environment_architecture.png",
    "docs/images/agent_architectures.png",
    "docs/images/training_pipeline.png",
    "scripts/generate_docs_diagrams.py",
    "scripts/generate_communication_posters.py",
]


def redeploy():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN required.")
    for f in QUICK_FILES:
        if os.path.isfile(f):
            upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=HF_REPO_ID,
                repo_type="space",
                token=HF_TOKEN,
            )
            print(f"  UPLOADED {f}")
    print(f"Redeployed to: https://huggingface.co/spaces/{HF_REPO_ID}")


if __name__ == "__main__":
    redeploy()
