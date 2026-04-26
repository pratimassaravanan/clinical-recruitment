#!/usr/bin/env python3
"""AutoResearch Orchestrator for Clinical Recruitment.

Autonomous loop: LLM proposes edits to train.py, deploys to Kaggle T4,
polls for results, keeps or discards based on composite score.

Usage:
    python autoresearch/run_autoresearch.py
    python autoresearch/run_autoresearch.py --model anthropic--claude-4.6-sonnet --max-experiments 50
    python autoresearch/run_autoresearch.py --local  # skip Kaggle, run train.py locally (needs GPU)
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
TRAIN_PY = SCRIPT_DIR / "train.py"
RESULTS_TSV = SCRIPT_DIR / "results.tsv"
PROGRAM_MD = SCRIPT_DIR / "program.md"
EVALUATE_PY = SCRIPT_DIR / "evaluate.py"
KAGGLE_KERNEL_DIR = REPO_DIR / "kaggle_kernel"

# Import evaluate functions
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate import (
    RESULTS_TSV_HEADER,
    compare_experiments,
    compute_composite_score,
    format_results_row,
    parse_train_output,
)


# ── Hyperspace LLM Client ────────────────────────────────────────────

def call_llm(prompt: str, model: str = "anthropic--claude-4.6-sonnet",
             api_key: str = None, base_url: str = None,
             max_tokens: int = 8192, temperature: float = 0.7) -> str:
    """Call Hyperspace LLM proxy for a completion."""
    import urllib.request
    import urllib.error

    base_url = base_url or os.environ.get("HYPERSPACE_URL", "http://localhost:6655")
    api_key = api_key or os.environ.get("HYPERSPACE_API_KEY", "d3d25b98-d27a-4d9c-8f95-5d39731e3a3a")

    url = f"{base_url}/litellm/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  LLM HTTP {e.code}: {body[:300]}", file=sys.stderr)
        raise
    except urllib.error.URLError as e:
        print(f"  LLM connection error: {e.reason}", file=sys.stderr)
        raise

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices in LLM response")
    return choices[0].get("message", {}).get("content", "")


# ── Kaggle Deployment ─────────────────────────────────────────────────

def deploy_to_kaggle(train_py_content: str, kernel_slug: str) -> str:
    """Push a training kernel to Kaggle and return the kernel URL."""
    kernel_dir = SCRIPT_DIR / "_kaggle_tmp"
    kernel_dir.mkdir(exist_ok=True)

    # Write the training script as a notebook-compatible .py
    (kernel_dir / "train_autoresearch.py").write_text(train_py_content, encoding="utf-8")

    # Write kernel metadata
    metadata = {
        "id": kernel_slug,
        "title": f"AutoResearch {datetime.now().strftime('%m%d-%H%M')}",
        "code_file": "train_autoresearch.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["kaushiksarav/clinical-sft-traces-5k"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Push to Kaggle
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle push failed: {result.stderr}")

    print(f"  Pushed to Kaggle: {kernel_slug}")
    return kernel_slug


def poll_kaggle_kernel(kernel_slug: str, timeout_min: int = 45, poll_sec: int = 60) -> str:
    """Poll Kaggle until kernel completes. Returns the log output."""
    deadline = time.time() + timeout_min * 60
    last_status = ""

    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["kaggle", "kernels", "status", kernel_slug],
                capture_output=True, text=True, timeout=30,
            )
            status_text = result.stdout.strip().lower()

            if "complete" in status_text:
                print(f"  Kernel complete!")
                break
            elif "error" in status_text or "cancel" in status_text:
                print(f"  Kernel failed: {status_text}")
                return f"KERNEL_FAILED: {status_text}"

            if status_text != last_status:
                remaining = int((deadline - time.time()) / 60)
                print(f"  Status: {status_text} ({remaining}min remaining)")
                last_status = status_text

        except subprocess.TimeoutExpired:
            pass

        time.sleep(poll_sec)
    else:
        return "KERNEL_TIMEOUT"

    # Fetch the log
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "output", kernel_slug, "-p", str(SCRIPT_DIR / "_kaggle_output")],
            capture_output=True, text=True, timeout=120,
        )
        output_dir = SCRIPT_DIR / "_kaggle_output"
        # Read all log files
        log_text = ""
        if output_dir.exists():
            for f in sorted(output_dir.iterdir()):
                if f.suffix in (".log", ".txt", ".json", ""):
                    try:
                        log_text += f.read_text(encoding="utf-8", errors="replace") + "\n"
                    except Exception:
                        pass
        if not log_text:
            log_text = result.stdout + "\n" + result.stderr
        return log_text
    except Exception as e:
        return f"LOG_FETCH_ERROR: {e}"


# ── Local Execution ───────────────────────────────────────────────────

def run_locally(train_py_path: Path, timeout_min: int = 60) -> str:
    """Run train.py locally and capture output."""
    result = subprocess.run(
        [sys.executable, str(train_py_path)],
        capture_output=True, text=True,
        timeout=timeout_min * 60,
        cwd=str(REPO_DIR),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return result.stdout + "\n" + result.stderr


# ── LLM Researcher Agent ─────────────────────────────────────────────

def build_researcher_prompt(
    current_train_py: str,
    results_history: str,
    program_md: str,
    experiment_num: int,
    best_score: float,
) -> str:
    """Build the prompt for the LLM researcher."""
    return f"""You are an autonomous ML researcher running experiment #{experiment_num}.

## Instructions
{program_md}

## Current best composite_score: {best_score:.6f}

## Experiment History (results.tsv)
{results_history if results_history else "(no experiments yet - this will be the baseline run)"}

## Current train.py
```python
{current_train_py}
```

## Your Task

{"This is the FIRST run. Run train.py as-is to establish the baseline. Output the EXACT same train.py unchanged." if experiment_num == 1 else "Analyze the results history. Propose ONE focused improvement to train.py. Explain your reasoning in 2-3 sentences, then output the complete modified train.py."}

{"" if experiment_num == 1 else "Think about what has worked and what hasn't. Consider:"}
{"" if experiment_num == 1 else "- Which benchmarks are weakest? Focus improvements there."}
{"" if experiment_num == 1 else "- Is the system prompt guiding the model well enough?"}
{"" if experiment_num == 1 else "- Are hyperparameters optimal (LR, epochs, LoRA rank)?"}
{"" if experiment_num == 1 else "- Can observation formatting be improved?"}
{"" if experiment_num == 1 else "- Can parse_action be smarter about site/patient selection?"}

## Output Format

First, write a SHORT description of what you're changing (one line, for results.tsv).
Then output the complete train.py between ```python and ``` markers.

DESCRIPTION: <one-line description>

```python
<complete train.py content>
```"""


def extract_train_py_from_response(response: str) -> tuple:
    """Extract description and train.py content from LLM response.
    
    Returns: (description, train_py_content) or (description, None) on failure.
    """
    # Extract description
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", response)
    description = desc_match.group(1).strip() if desc_match else "no description"

    # Extract python code block
    code_match = re.search(r"```python\s*\n(.+?)```", response, re.DOTALL)
    if code_match:
        return description, code_match.group(1).strip()

    # Fallback: look for any large code block
    code_match = re.search(r"```\s*\n(.+?)```", response, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        if "import" in content and ("SFT" in content or "train" in content.lower()):
            return description, content

    return description, None


# ── Main Loop ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clinical Recruitment AutoResearch")
    parser.add_argument("--model", default="anthropic--claude-4.6-sonnet",
                        help="Hyperspace model for the researcher agent")
    parser.add_argument("--max-experiments", type=int, default=100,
                        help="Maximum number of experiments to run")
    parser.add_argument("--local", action="store_true",
                        help="Run train.py locally instead of on Kaggle")
    parser.add_argument("--kaggle-slug", default="kaushiksarav/clinical-autoresearch",
                        help="Kaggle kernel slug")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between Kaggle status polls")
    parser.add_argument("--timeout-min", type=int, default=45,
                        help="Max minutes to wait for a Kaggle kernel")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just generate the first experiment, don't deploy")
    args = parser.parse_args()

    print("=" * 60)
    print("CLINICAL RECRUITMENT AUTORESEARCH")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Mode:    {'local' if args.local else 'kaggle'}")
    print(f"  Max:     {args.max_experiments} experiments")
    print()

    # Initialize results.tsv if it doesn't exist
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_TSV_HEADER + "\n", encoding="utf-8")

    # Ensure train.py exists (copy from parent if needed)
    if not TRAIN_PY.exists():
        src = REPO_DIR / "train.py"
        if src.exists():
            shutil.copy2(src, TRAIN_PY)
            print(f"  Copied baseline train.py from {src}")
        else:
            print("ERROR: No train.py found!", file=sys.stderr)
            return 1

    # Load program.md
    program_md = PROGRAM_MD.read_text(encoding="utf-8") if PROGRAM_MD.exists() else ""

    # Track best score
    best_score = 0.0
    best_train_py = TRAIN_PY.read_text(encoding="utf-8")

    # Parse existing results to find best score
    if RESULTS_TSV.exists():
        for line in RESULTS_TSV.read_text(encoding="utf-8").strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[-2] == "keep":
                try:
                    score = float(parts[1])
                    if score > best_score:
                        best_score = score
                except ValueError:
                    pass

    experiment_num = len([
        l for l in RESULTS_TSV.read_text(encoding="utf-8").strip().split("\n")[1:]
        if l.strip()
    ]) + 1

    print(f"  Starting from experiment #{experiment_num}")
    print(f"  Current best score: {best_score:.6f}")
    print()

    # ── EXPERIMENT LOOP ───────────────────────────────────────────
    while experiment_num <= args.max_experiments:
        exp_id = f"exp{experiment_num:03d}"
        print(f"\n{'='*60}")
        print(f"EXPERIMENT #{experiment_num} ({exp_id})")
        print(f"{'='*60}")

        # 1. Read current state
        current_train_py = TRAIN_PY.read_text(encoding="utf-8")
        results_history = RESULTS_TSV.read_text(encoding="utf-8") if RESULTS_TSV.exists() else ""

        # 2. Ask LLM for a proposal
        print("  Asking LLM for proposal...")
        prompt = build_researcher_prompt(
            current_train_py, results_history, program_md,
            experiment_num, best_score,
        )

        try:
            response = call_llm(prompt, model=args.model, temperature=0.7 if experiment_num > 1 else 0.0)
        except Exception as e:
            print(f"  LLM call failed: {e}")
            print("  Retrying in 30s...")
            time.sleep(30)
            try:
                response = call_llm(prompt, model=args.model, temperature=0.7)
            except Exception as e2:
                print(f"  LLM retry failed: {e2}. Skipping.")
                experiment_num += 1
                continue

        # 3. Extract the proposed train.py
        description, new_train_py = extract_train_py_from_response(response)
        print(f"  Description: {description}")

        if new_train_py is None:
            print("  Failed to extract train.py from LLM response. Skipping.")
            # Log as crash
            row = format_results_row(exp_id, 0.0, {}, "crash", f"LLM parse failure: {description}")
            with open(RESULTS_TSV, "a", encoding="utf-8") as f:
                f.write(row + "\n")
            experiment_num += 1
            continue

        if args.dry_run:
            print(f"\n  [DRY RUN] Would deploy train.py ({len(new_train_py)} chars)")
            print(f"  First 200 chars: {new_train_py[:200]}...")
            return 0

        # 4. Save the proposed train.py (backup current first)
        backup = TRAIN_PY.with_suffix(".py.bak")
        shutil.copy2(TRAIN_PY, backup)
        TRAIN_PY.write_text(new_train_py, encoding="utf-8")

        # 5. Deploy and run
        print("  Deploying...")
        try:
            if args.local:
                log_text = run_locally(TRAIN_PY, timeout_min=args.timeout_min)
            else:
                kernel_slug = deploy_to_kaggle(new_train_py, args.kaggle_slug)
                print(f"  Polling Kaggle (timeout={args.timeout_min}min)...")
                log_text = poll_kaggle_kernel(
                    kernel_slug,
                    timeout_min=args.timeout_min,
                    poll_sec=args.poll_interval,
                )
        except subprocess.TimeoutExpired:
            log_text = "TIMEOUT"
        except Exception as e:
            log_text = f"DEPLOY_ERROR: {e}"

        # 6. Parse results
        parsed = parse_train_output(log_text)
        composite_score = parsed["composite_score"]
        print(f"  Composite score: {composite_score:.6f} (best: {best_score:.6f})")

        if parsed["error"]:
            print(f"  Error: {parsed['error']}")

        for task in ["easy_bench", "medium_bench", "hard_bench"]:
            r = parsed["results"].get(task, {})
            print(f"    {task}: enrolled={r.get('enrolled',0)}/{r.get('target',0)} reward={r.get('total_reward',0):.2f}")

        # 7. Keep or discard
        if parsed["error"] and not parsed["results"]:
            status = "crash"
            # Revert
            shutil.copy2(backup, TRAIN_PY)
            print(f"  CRASH - reverted to previous train.py")
        elif composite_score > best_score + 0.001:
            status = "keep"
            best_score = composite_score
            best_train_py = new_train_py
            print(f"  KEEP! New best: {best_score:.6f}")
        else:
            status = "discard"
            # Revert
            shutil.copy2(backup, TRAIN_PY)
            print(f"  DISCARD - no improvement (delta={composite_score - best_score:+.6f})")

        # 8. Log to results.tsv
        row = format_results_row(exp_id, composite_score, parsed["results"], status, description)
        with open(RESULTS_TSV, "a", encoding="utf-8") as f:
            f.write(row + "\n")

        # Cleanup backup
        if backup.exists():
            backup.unlink()

        # Save best train.py separately
        best_path = SCRIPT_DIR / "train_best.py"
        best_path.write_text(best_train_py, encoding="utf-8")

        experiment_num += 1
        print(f"\n  Experiment complete. Moving to next...")

    print(f"\n{'='*60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"  Total experiments: {experiment_num - 1}")
    print(f"  Best composite score: {best_score:.6f}")
    print(f"  Best train.py saved to: {SCRIPT_DIR / 'train_best.py'}")
    print(f"  Results log: {RESULTS_TSV}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
