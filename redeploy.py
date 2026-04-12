"""Quick redeploy: upload only commonly-changed files to HF Spaces."""

import os
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "pratimassaravanan/clinical-recruitment")

QUICK_FILES = ["README.md", "inference.py", "app.py", "env.py", "graders.py"]


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
