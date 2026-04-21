from __future__ import annotations

import math
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "docs" / "images"
SWEEP_REPORT = ROOT / "data" / "sweep_results" / "neurips_report.json"

BG = "#F7F9FC"
TEXT = "#122033"
MUTED = "#5F6F86"
BORDER = "#D7DEEA"
ACCENT = "#2453FF"
ACCENT_2 = "#16A34A"
ACCENT_3 = "#7C3AED"
ACCENT_4 = "#EA580C"
ACCENT_5 = "#0F766E"
PANEL = "#FFFFFF"
PANEL_ALT = "#EEF4FF"
PANEL_SOFT = "#F4F7FB"


def _matplotlib_font_dir() -> Path | None:
    try:
        import matplotlib

        return Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    except Exception:
        return None


def _font_candidates(bold: bool) -> list[Path]:
    candidates: list[Path] = []
    mpl_dir = _matplotlib_font_dir()
    if mpl_dir is not None:
        candidates.extend(
            [
                mpl_dir / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                mpl_dir / ("DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"),
            ]
        )

    windows = Path("C:/Windows/Fonts")
    candidates.extend(
        [
            windows / ("arialbd.ttf" if bold else "arial.ttf"),
            windows / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        ]
    )
    return candidates


@lru_cache(maxsize=None)
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _font_candidates(bold):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    def split_long_token(token: str) -> list[str]:
        if text_size(draw, token, font)[0] <= max_width:
            return [token]
        chunks: list[str] = []
        current = ""
        for char in token:
            trial = current + char
            if current and text_size(draw, trial, font)[0] > max_width:
                chunks.append(current)
                current = char
            else:
                current = trial
        if current:
            chunks.append(current)
        return chunks

    if not text:
        return [""]

    paragraphs = text.split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        expanded_words: list[str] = []
        for word in words:
            expanded_words.extend(split_long_token(word))

        current = expanded_words[0]
        for word in expanded_words[1:]:
            trial = f"{current} {word}"
            if text_size(draw, trial, font)[0] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int = 7,
    head_len: int = 18,
    head_width: int = 10,
) -> None:
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, math.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    p1 = end
    p2 = (
        int(end[0] - ux * head_len + px * head_width),
        int(end[1] - uy * head_len + py * head_width),
    )
    p3 = (
        int(end[0] - ux * head_len - px * head_width),
        int(end[1] - uy * head_len - py * head_width),
    )
    draw.polygon([p1, p2, p3], fill=color)


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str = PANEL,
    outline: str = BORDER,
    radius: int = 28,
    width: int = 3,
) -> None:
    shadow_box = (box[0] + 10, box[1] + 12, box[2] + 10, box[3] + 12)
    draw.rounded_rectangle(shadow_box, radius=radius, fill="#E4EAF5")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    text_fill: str = TEXT,
    font_size: int = 28,
    padding_x: int = 18,
    padding_y: int = 10,
) -> tuple[int, int, int, int]:
    font = load_font(font_size, bold=True)
    tw, th = text_size(draw, text, font)
    box = (xy[0], xy[1], xy[0] + tw + padding_x * 2, xy[1] + th + padding_y * 2)
    draw.rounded_rectangle(box, radius=18, fill=fill)
    draw.text((xy[0] + padding_x, xy[1] + padding_y - 2), text, fill=text_fill, font=font)
    return box


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    lines: Sequence[str],
    accent: str,
    fill: str = PANEL,
    title_size: int = 34,
    body_size: int = 25,
    inset: int = 28,
    line_gap: int = 10,
) -> None:
    rounded_panel(draw, box, fill=fill)
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1, y1, x2, y1 + 16), radius=28, fill=accent)

    title_font = load_font(title_size, bold=True)
    body_font = load_font(body_size)
    body_width = x2 - x1 - inset * 2
    title_lines = wrap_text(draw, title, title_font, body_width)
    current_title_y = y1 + inset - 4
    for line in title_lines:
        draw.text((x1 + inset, current_title_y), line, fill=TEXT, font=title_font)
        current_title_y += text_size(draw, line, title_font)[1] + 4
    divider_y = current_title_y + 10
    draw.line((x1 + inset, divider_y, x2 - inset, divider_y), fill="#E6ECF5", width=2)

    current_y = divider_y + 18
    for item in lines:
        wrapped = wrap_text(draw, item, body_font, body_width)
        for line in wrapped:
            draw.text((x1 + inset, current_y), line, fill=TEXT, font=body_font)
            current_y += text_size(draw, line, body_font)[1] + 2
        current_y += line_gap


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font_size: int,
    fill: str,
    bold: bool = True,
) -> None:
    font = load_font(font_size, bold=bold)
    lines = wrap_text(draw, text, font, box[2] - box[0])
    line_heights = [text_size(draw, line, font)[1] for line in lines]
    total_h = sum(line_heights) + max(0, len(lines) - 1) * 4
    y = box[1] + (box[3] - box[1] - total_h) / 2 - 2
    for line, line_h in zip(lines, line_heights):
        tw, _ = text_size(draw, line, font)
        x = box[0] + (box[2] - box[0] - tw) / 2
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h + 4


def draw_badge_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    caption: str,
    fill: str,
    accent: str,
    caption_font_size: int = 24,
    chip_font_size: int = 24,
    top_padding: int = 16,
) -> None:
    rounded_panel(draw, box, fill=fill, outline=accent)
    header = (box[0] + 18, box[1] + top_padding)
    chip_box = draw_chip(draw, header, label, fill=accent, text_fill="#FFFFFF", font_size=chip_font_size, padding_x=16, padding_y=8)
    caption_font = load_font(caption_font_size)
    wrapped = wrap_text(draw, caption, caption_font, box[2] - box[0] - 40)
    current_y = chip_box[3] + 14
    for line in wrapped:
        draw.text((box[0] + 20, current_y), line, fill=TEXT, font=caption_font)
        current_y += text_size(draw, line, caption_font)[1] + 4


def canvas(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    return image, draw


def title_block(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    left: int = 72,
    top: int = 52,
) -> int:
    title_font = load_font(54, bold=True)
    subtitle_font = load_font(28)
    draw.text((left, top), title, fill=TEXT, font=title_font)
    th = text_size(draw, title, title_font)[1]
    draw.text((left, top + th + 14), subtitle, fill=MUTED, font=subtitle_font)
    sh = text_size(draw, subtitle, subtitle_font)[1]
    return top + th + sh + 40


def save(image: Image.Image, name: str) -> Path:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGES_DIR / name
    image.save(out_path, format="PNG", optimize=True)
    return out_path


def load_agent_score_map() -> dict[str, str]:
    if not SWEEP_REPORT.exists():
        return {}

    try:
        data = json.loads(SWEEP_REPORT.read_text(encoding="utf-8"))
    except Exception:
        return {}

    labels: dict[str, str] = {}
    for agent, stats in data.get("sweep_results", {}).items():
        mean = stats.get("mean")
        if mean is None:
            continue
        labels[agent.lower()] = f"Mean score {float(mean):.4f}"
    return labels


def generate_environment_architecture() -> Path:
    image, draw = canvas((2200, 1400))
    cursor_y = title_block(
        draw,
        "Clinical Recruitment Environment",
        "Deterministic 180-day benchmark with a 37-feature observation, 8 implemented actions, and multi-site constraints.",
    )

    chip_x = 72
    for text, fill, fg in [
        ("180 steps", "#DBEAFE", "#1D4ED8"),
        ("37 features", "#DCFCE7", "#15803D"),
        ("8 actions", "#F3E8FF", "#7E22CE"),
        ("easy / medium / hard", "#FFEDD5", "#C2410C"),
    ]:
        box = draw_chip(draw, (chip_x, cursor_y), text, fill=fill, text_fill=fg, font_size=28)
        chip_x = box[2] + 18

    agent_box = (72, 250, 620, 615)
    actions_box = (72, 655, 620, 1322)
    funnel_box = (690, 250, 1490, 820)
    state_box = (690, 860, 1490, 1322)
    sites_box = (1560, 250, 2128, 820)
    reward_box = (1560, 860, 2128, 1322)

    draw_card(
        draw,
        agent_box,
        "Agent loop",
        [
            "Chooses one action for each simulated day.",
            "Receives the typed observation from env.py and models.py.",
            "Can plan ahead, write memory, and retrieve indexed history.",
        ],
        accent=ACCENT,
        fill=PANEL,
    )

    rounded_panel(draw, actions_box, fill=PANEL_ALT)
    draw.rounded_rectangle((actions_box[0], actions_box[1], actions_box[2], actions_box[1] + 16), radius=28, fill=ACCENT_3)
    draw.text((actions_box[0] + 28, actions_box[1] + 24), "8 implemented actions", fill=TEXT, font=load_font(34, bold=True))
    subtitle_font = load_font(24)
    subtitle_lines = wrap_text(
        draw,
        "Exact action types exposed by the repo models and policy code.",
        subtitle_font,
        actions_box[2] - actions_box[0] - 56,
    )
    subtitle_y = actions_box[1] + 74
    for line in subtitle_lines:
        draw.text((actions_box[0] + 28, subtitle_y), line, fill=MUTED, font=subtitle_font)
        subtitle_y += text_size(draw, line, subtitle_font)[1] + 4
    action_labels = [
        "screen_patient",
        "recontact",
        "allocate_to_site",
        "adjust_strategy",
        "plan_next_phase",
        "summarize\nand_index",
        "retrieve\nrelevant\nhistory",
        "stop_recruitment",
    ]
    pill_w = 228
    pill_h = 104
    start_x = actions_box[0] + 28
    start_y = subtitle_y + 18
    gap_x = 24
    gap_y = 20
    for idx, label in enumerate(action_labels):
        row = idx // 2
        col = idx % 2
        x1 = start_x + col * (pill_w + gap_x)
        y1 = start_y + row * (pill_h + gap_y)
        x2 = x1 + pill_w
        y2 = y1 + pill_h
        draw.rounded_rectangle((x1, y1, x2, y2), radius=20, fill="#FFFFFF", outline="#C8D5F2", width=2)
        draw_centered_text(draw, (x1 + 14, y1 + 10, x2 - 14, y2 - 10), label, 18, fill=TEXT, bold=False)

    draw_card(
        draw,
        funnel_box,
        "Patient funnel and execution flow",
        [
            "The benchmark models a screening -> enrollment -> retention workflow over 180 simulated days.",
            "Action-specific candidate pools support screening, follow-up, and site allocation decisions.",
        ],
        accent=ACCENT_2,
        fill=PANEL,
    )

    stage_y = funnel_box[1] + 220
    stage_h = 190
    stage_gap = 24
    stage_w = 210
    stage_x = funnel_box[0] + 42
    stage_specs = [
        ("Screening", "Evaluate eligibility and manage backlog.", "#E8F1FF", ACCENT),
        ("Enrollment", "Consent, allocate to sites, and convert patients.", "#ECFDF3", ACCENT_2),
        ("Retention", "Track dropout risk and re-engage patients.", "#F5F3FF", ACCENT_3),
    ]
    stage_boxes: list[tuple[int, int, int, int]] = []
    for i, (title, caption, fill, accent_color) in enumerate(stage_specs):
        x1 = stage_x + i * (stage_w + stage_gap)
        box = (x1, stage_y, x1 + stage_w, stage_y + stage_h)
        stage_boxes.append(box)
        draw_badge_panel(
            draw,
            box,
            title,
            caption,
            fill=fill,
            accent=accent_color,
            caption_font_size=20,
            chip_font_size=22,
        )

    for left_box, right_box, color in zip(stage_boxes, stage_boxes[1:], [ACCENT, ACCENT_2]):
        draw_arrow(
            draw,
            (left_box[2] + 10, (left_box[1] + left_box[3]) // 2),
            (right_box[0] - 10, (right_box[1] + right_box[3]) // 2),
            color=color,
        )

    note_box = (funnel_box[0] + 52, funnel_box[1] + 690, funnel_box[2] - 52, funnel_box[1] + 804)
    draw_badge_panel(
        draw,
        note_box,
        "State signals",
        "Candidate pools, site metrics, and recent events feed each next decision.",
        fill="#F9FBFE",
        accent=ACCENT_5,
        caption_font_size=18,
        chip_font_size=21,
        top_padding=12,
    )

    draw_card(
        draw,
        state_box,
        "State, constraints, and long-horizon signals",
        [
            "Observation includes budget, deadline, enrollment progress, uncertainty, and dropout trend.",
            "Long-horizon fields include milestones, active constraints, delayed effects, and counterfactual hints.",
            "Planning and memory context include current_plan, indexed_memory_summary, and retrieved_memory_context.",
            "Token accounting exposes token_budget_remaining and token_efficiency_score.",
        ],
        accent=ACCENT_5,
        fill=PANEL,
    )

    draw_card(
        draw,
        sites_box,
        "Multi-site layer",
        [
            "Each observation exposes per-site conversion rate, average wait days, and capacity remaining.",
            "Allocation and negotiation operate over site-specific capacity and wait-time signals.",
        ],
        accent=ACCENT_4,
        fill=PANEL,
    )

    site_cards = [
        (sites_box[0] + 30, sites_box[1] + 382, sites_box[2] - 30, sites_box[1] + 462, "Site performance"),
        (sites_box[0] + 30, sites_box[1] + 484, sites_box[2] - 30, sites_box[1] + 564, "Capacity-aware allocation"),
        (sites_box[0] + 30, sites_box[1] + 586, sites_box[2] - 30, sites_box[1] + 666, "Negotiation via adjust_strategy"),
    ]
    for x1, y1, x2, y2, label in site_cards:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill="#FFF7ED", outline="#FDBA74", width=2)
        draw_centered_text(draw, (x1 + 10, y1 + 8, x2 - 10, y2 - 8), label, 24, fill="#9A3412", bold=True)

    draw_card(
        draw,
        reward_box,
        "Reward and benchmark outputs",
        [
            "Per-step reward combines screening success, enrollment gain, dropout penalty, budget efficiency, and timeline bonus.",
            "The environment also tracks milestone progress, hypothesis accuracy, and token efficiency.",
            "Episodes finish with a graded final_score in the 0 to 1 range for benchmark comparison.",
        ],
        accent="#0F766E",
        fill=PANEL,
    )

    draw_arrow(draw, (620, 432), (690, 432), color=ACCENT)
    draw_arrow(draw, (1490, 492), (1560, 492), color=ACCENT_4)
    draw_arrow(draw, (1490, 1090), (1560, 1090), color=ACCENT_5)
    draw_arrow(draw, (1090, 820), (1090, 860), color=ACCENT_2)
    draw_arrow(draw, (892, 860), (560, 860), color="#64748B")

    return save(image, "environment_architecture.png")


def generate_agent_architectures() -> Path:
    image, draw = canvas((2400, 1400))
    score_map = load_agent_score_map()
    cursor_y = title_block(
        draw,
        "Implemented Agent Architectures",
        "Repo-verified architecture summary for the four long-horizon agents used in the sweep report.",
    )
    banner_text = (
        "Scores loaded from data/sweep_results/neurips_report.json"
        if score_map
        else "Architecture summary grounded in repo code paths"
    )
    draw_chip(draw, (72, cursor_y), banner_text, fill="#E0F2FE", text_fill="#075985", font_size=28)

    margin = 72
    gap = 32
    panel_w = 548
    panel_h = 1050
    top = 248
    agent_specs = [
        (
            "HCAPO",
            ACCENT,
            score_map.get("hcapo", "Repo baseline"),
            [
                ("Hierarchical planner", "Infers screening, conversion, allocation, retention, or recovery subgoals."),
                ("Executor over 8 actions", "Selects primitive actions from the repo action space after planner context is attached."),
                ("Hindsight relabeling", "Extracts achieved goals from trajectories and propagates long-horizon credit."),
            ],
        ),
        (
            "MiRA",
            ACCENT_2,
            score_map.get("mira", "Repo baseline"),
            [
                ("Policy network", "Runs the main actor-critic policy over the shared 37-feature state."),
                ("Potential critic", "Learns milestone-aware potential values from progress, budget, time, and funnel health."),
                ("Shaped reward", "Applies potential-based shaping between consecutive states during updates."),
            ],
        ),
        (
            "KLong",
            ACCENT_3,
            score_map.get("klong", "Repo baseline"),
            [
                ("Multi-scale context", "Builds temporal abstractions over 1, 5, 20, and 60 step windows."),
                ("Segmented trajectories", "Processes overlapping trajectory windows for long-context learning."),
                ("TD(lambda) traces", "Updates policy and segment critic with eligibility traces for temporal credit assignment."),
            ],
        ),
        (
            "MemexRL",
            ACCENT_4,
            score_map.get("memex", "Repo baseline"),
            [
                ("Episodic memory store", "Encodes key/value entries from state, action, reward, and concise observation summary."),
                ("Attention retrieval", "Reads memory with cosine-similarity attention over stored episodes."),
                ("Learned write gate", "Predicts write probability and memory importance before storage or eviction."),
            ],
        ),
    ]

    for idx, (name, accent_color, score_text, sections) in enumerate(agent_specs):
        x1 = margin + idx * (panel_w + gap)
        box = (x1, top, x1 + panel_w, top + panel_h)
        rounded_panel(draw, box, fill=PANEL)
        draw.rounded_rectangle((box[0], box[1], box[2], box[1] + 20), radius=30, fill=accent_color)
        draw.text((box[0] + 30, box[1] + 34), name, fill=TEXT, font=load_font(40, bold=True))
        score_box = draw_chip(
            draw,
            (box[0] + 30, box[1] + 96),
            score_text,
            fill="#F8FAFC",
            text_fill=accent_color,
            font_size=26,
            padding_x=16,
            padding_y=8,
        )
        status_box = draw_chip(
            draw,
            (box[0] + 30, score_box[3] + 12),
            "implemented in repo",
            fill="#EFF6FF",
            text_fill="#1D4ED8",
            font_size=22,
            padding_x=16,
            padding_y=8,
        )

        section_y = max(score_box[3], status_box[3]) + 28
        for title, caption in sections:
            section_box = (box[0] + 24, section_y, box[2] - 24, section_y + 214)
            draw_badge_panel(
                draw,
                section_box,
                title,
                caption,
                fill="#FBFDFF",
                accent=accent_color,
            )
            section_y += 238

        footer_box = (box[0] + 24, box[3] - 128, box[2] - 24, box[3] - 24)
        draw.rounded_rectangle(footer_box, radius=20, fill="#F8FAFC", outline="#DCE5F3", width=2)
        footer_font = load_font(24)
        footer_lines = wrap_text(
            draw,
            "All four methods are instantiated directly in experiments/full_sweep.py for multi-seed training and evaluation.",
            footer_font,
            footer_box[2] - footer_box[0] - 30,
        )
        current_y = footer_box[1] + 20
        for line in footer_lines:
            draw.text((footer_box[0] + 16, current_y), line, fill=MUTED, font=footer_font)
            current_y += text_size(draw, line, footer_font)[1] + 4

    return save(image, "agent_architectures.png")


def generate_training_pipeline() -> Path:
    image, draw = canvas((2400, 1500))
    cursor_y = title_block(
        draw,
        "Training and Evaluation Pipeline",
        "Programmatic overview of the deterministic benchmark, training modules, and experiment outputs wired together in the repo.",
    )
    draw_chip(draw, (72, cursor_y), "All labels sourced from repo files", fill="#DCFCE7", text_fill="#166534", font_size=28)

    top = 280
    box_h = 1020
    col_w = 400
    gap = 44
    start_x = 72
    boxes = []
    for i in range(5):
        x1 = start_x + i * (col_w + gap)
        boxes.append((x1, top, x1 + col_w, top + box_h))

    draw_card(
        draw,
        boxes[0],
        "1. Deterministic tasks",
        [
            "load_traces.py supplies public tasks and progressive stage-task horizons.",
            "env.py seeds resets deterministically per task and tracks a 180-step episode horizon.",
            "Task families exposed in the README are easy_bench, medium_bench, and hard_bench.",
        ],
        accent=ACCENT,
        fill=PANEL,
        title_size=30,
        body_size=22,
    )
    draw_badge_panel(
        draw,
        (boxes[0][0] + 26, boxes[0][1] + 600, boxes[0][2] - 26, boxes[0][1] + 740),
        "Trace outputs",
        "Patients, sites, events, budgets, and curriculum state are copied into each reset.",
        fill="#EEF4FF",
        accent=ACCENT,
        caption_font_size=19,
        chip_font_size=21,
    )
    draw_badge_panel(
        draw,
        (boxes[0][0] + 26, boxes[0][1] + 780, boxes[0][2] - 26, boxes[0][1] + 910),
        "Task staging",
        "training/progressive_rl.py iterates stage-task IDs across progressive horizons.",
        fill="#F8FAFC",
        accent="#2563EB",
        caption_font_size=19,
        chip_font_size=21,
    )

    draw_card(
        draw,
        boxes[1],
        "2. Environment interface",
        [
            "models.py defines the typed Observation and Action interfaces.",
            "The observation exposes 37 numeric features through training/neural_policy.py.",
            "The action space contains 8 implemented action types shared by agents and policies.",
        ],
        accent=ACCENT_2,
        fill=PANEL,
        title_size=30,
        body_size=22,
    )
    draw_badge_panel(
        draw,
        (boxes[1][0] + 26, boxes[1][1] + 650, boxes[1][2] - 26, boxes[1][1] + 840),
        "Long-horizon state",
        "Milestones, active constraints, delayed effects, memory summaries, counterfactual hints, and token accounting remain visible during training.",
        fill="#ECFDF3",
        accent=ACCENT_2,
        caption_font_size=19,
        chip_font_size=21,
    )

    draw_card(
        draw,
        boxes[2],
        "3. Trainable agents",
        [
            "research/methods provides HCAPO, MiRA, KLong, and MemexRL implementations.",
            "training/offline_policy.py adds LinearPolicy and MLPPolicy baselines for offline training.",
            "training/neural_policy.py supplies the shared actor-critic and feature extraction core.",
        ],
        accent=ACCENT_3,
        fill=PANEL,
        title_size=30,
        body_size=22,
    )
    draw_badge_panel(
        draw,
        (boxes[2][0] + 26, boxes[2][1] + 620, boxes[2][2] - 26, boxes[2][1] + 760),
        "Action payloads",
        "offline_policy.py normalizes task-aware action payloads before each environment step.",
        fill="#F5F3FF",
        accent=ACCENT_3,
        caption_font_size=18,
        chip_font_size=20,
    )
    draw_badge_panel(
        draw,
        (boxes[2][0] + 26, boxes[2][1] + 790, boxes[2][2] - 26, boxes[2][1] + 930),
        "Shared state vector",
        "training/neural_policy.py fixes STATE_DIM = 37 for the shared feature vector.",
        fill="#FAF5FF",
        accent="#7E22CE",
        caption_font_size=18,
        chip_font_size=20,
    )

    draw_card(
        draw,
        boxes[3],
        "4. Training modules",
        [
            "training/train_offline_policy.py runs epoch-based offline policy training and evaluation.",
            "training/curriculum.py contains progressive, recovery, and Thompson-style curriculum managers.",
            "training/async_rl.py and training/progressive_rl.py scaffold async and staged-horizon runs.",
        ],
        accent=ACCENT_4,
        fill=PANEL,
        title_size=30,
        body_size=22,
    )
    draw_badge_panel(
        draw,
        (boxes[3][0] + 26, boxes[3][1] + 650, boxes[3][2] - 26, boxes[3][1] + 810),
        "Saved artifacts",
        "Offline training persists policy weights, training history CSVs, and evaluation CSVs.",
        fill="#FFF7ED",
        accent=ACCENT_4,
        caption_font_size=19,
        chip_font_size=21,
    )

    draw_card(
        draw,
        boxes[4],
        "5. Reports and outputs",
        [
            "experiments/train_agents.py and experiments/full_sweep.py run multi-agent evaluations.",
            "experiments/reproducibility.py computes bootstrap intervals, paired tests, Wilcoxon, and effect sizes.",
            "Ablation, appendix, Pareto, and plotting scripts turn results into reports and figures.",
        ],
        accent=ACCENT_5,
        fill=PANEL,
        title_size=30,
        body_size=22,
    )
    draw_badge_panel(
        draw,
        (boxes[4][0] + 26, boxes[4][1] + 620, boxes[4][2] - 26, boxes[4][1] + 770),
        "Example report",
        "neurips_report.md stores the mean scores and pairwise comparisons used in the docs.",
        fill="#F0FDFA",
        accent=ACCENT_5,
        caption_font_size=18,
        chip_font_size=20,
    )
    draw_badge_panel(
        draw,
        (boxes[4][0] + 26, boxes[4][1] + 810, boxes[4][2] - 26, boxes[4][1] + 970),
        "Plot scripts",
        "generate_charts.py plus the training and trajectory plot scripts write documentation visuals.",
        fill="#F8FAFC",
        accent="#0F766E",
        caption_font_size=18,
        chip_font_size=20,
    )

    for left_box, right_box, color in zip(boxes, boxes[1:], [ACCENT, ACCENT_2, ACCENT_3, ACCENT_4]):
        arrow_y = top + box_h - 110
        draw_arrow(draw, (left_box[2] + 10, arrow_y), (right_box[0] - 10, arrow_y), color=color, width=8, head_len=20, head_width=11)

    footer = (72, 1342, 2328, 1442)
    rounded_panel(draw, footer, fill="#FFFFFF", outline="#DCE7F6", radius=24)
    footer_text = (
        "Verified flow: deterministic traces -> typed environment interface -> agent or offline policy training -> orchestration modules -> experiment reports and charts."
    )
    footer_font = load_font(28)
    footer_lines = wrap_text(draw, footer_text, footer_font, footer[2] - footer[0] - 40)
    y = footer[1] + 26
    for line in footer_lines:
        draw.text((footer[0] + 20, y), line, fill=TEXT, font=footer_font)
        y += text_size(draw, line, footer_font)[1] + 6

    return save(image, "training_pipeline.png")


def main() -> None:
    outputs = [
        generate_environment_architecture(),
        generate_agent_architectures(),
        generate_training_pipeline(),
    ]
    for path in outputs:
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
