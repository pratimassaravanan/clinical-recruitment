"""Custom Gradio guide tab for the clinical recruitment OpenEnv UI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import gradio as gr

from openenv.core.env_server.types import EnvironmentMetadata


_ACTION_GUIDE_ROWS = [
    (
        "screen_patient",
        "`patient_id`",
        "Pick an ID from `available_patients` to start the funnel.",
    ),
    (
        "recontact",
        "`patient_id`",
        "Pick an ID from `recontact_candidates` to move a screened patient toward consent.",
    ),
    (
        "allocate_to_site",
        "`patient_id`, `site_id`",
        "Pick a consented patient from `allocation_candidates` and a site from `site_performance` to enroll.",
    ),
    (
        "adjust_strategy",
        "`strategy_change`",
        "Use when no direct patient action is available, or when you need to change outreach/site strategy.",
    ),
    (
        "plan_next_phase",
        "`target_phase`",
        "Use for explicit plan updates such as `screening`, `conversion`, `allocation`, or `retention`.",
    ),
    (
        "summarize_and_index",
        "`memory_key`, `memory_payload`",
        "Write a compact memory entry you may want to retrieve later in the episode.",
    ),
    (
        "retrieve_relevant_history",
        "`memory_query`",
        "Search previously indexed memory before choosing the next action.",
    ),
    (
        "stop_recruitment",
        "none",
        "End the episode early when no further useful actions remain.",
    ),
]


def _action_guide_markdown() -> str:
    lines = [
        "## Action Guide",
        "",
        "Only `action_type` is always required. Leave unrelated fields blank.",
        "",
        "Recommended funnel order: `screen_patient` -> `recontact` -> `allocate_to_site`.",
        "",
        "| Action | Fields to fill | When to use |",
        "|---|---|---|",
    ]
    for action, fields, guidance in _ACTION_GUIDE_ROWS:
        lines.append(f"| `{action}` | {fields} | {guidance} |")
    lines.extend(
        [
            "",
            "`hypothesis` is optional but useful for diagnosis. `confidence` should stay between `0.0` and `1.0`.",
        ]
    )
    return "\n".join(lines)


def _field_reference_markdown(action_fields: List[Dict[str, Any]]) -> str:
    lines = [
        "## Parameter Reference",
        "",
        "| Field | Required | Meaning |",
        "|---|---|---|",
    ]
    for field in action_fields:
        description = field.get("description") or ""
        if field.get("choices"):
            choices = ", ".join(f"`{choice}`" for choice in field["choices"])
            description = f"{description} Choices: {choices}".strip()
        if field.get("min_value") is not None or field.get("max_value") is not None:
            low = field.get("min_value", "")
            high = field.get("max_value", "")
            description = f"{description} Range: `{low}` to `{high}`.".strip()
        required = "yes" if field.get("required") else "no"
        lines.append(f"| `{field['name']}` | {required} | {description} |")
    return "\n".join(lines)


def _examples_markdown() -> str:
    return "\n".join(
        [
            "## Example Actions",
            "",
            "```json",
            '{"action_type": "screen_patient", "patient_id": "P-1058", "hypothesis": "noise_dominant", "confidence": 0.7}',
            "```",
            "",
            "```json",
            '{"action_type": "allocate_to_site", "patient_id": "P-2041", "site_id": "site_B", "hypothesis": "site_bias", "confidence": 0.8}',
            "```",
            "",
            "```json",
            '{"action_type": "adjust_strategy", "strategy_change": "increase_outreach", "hypothesis": "dropout_dominant", "confidence": 0.6}',
            "```",
        ]
    )


def build_clinical_recruitment_guide(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    """Build a guide tab that makes the action schema visible in the web UI."""
    del web_manager, is_chat_env

    env_name = metadata.name if metadata else title
    with gr.Blocks(title=f"{title} Guide") as demo:
        gr.Markdown(
            f"# Guided Playground Reference\n\n"
            f"Use this tab alongside the Playground when filling action fields for **{env_name}**."
        )
        if quick_start_md:
            with gr.Accordion("Quick Start", open=True):
                gr.Markdown(quick_start_md)
        with gr.Accordion("Action Guide", open=True):
            gr.Markdown(_action_guide_markdown())
        with gr.Accordion("Parameter Reference", open=True):
            gr.Markdown(_field_reference_markdown(action_fields))
        with gr.Accordion("Example JSON", open=False):
            gr.Markdown(_examples_markdown())
    return demo
