from __future__ import annotations

from pathlib import Path
from typing import Callable
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "communication"
OUT.mkdir(parents=True, exist_ok=True)
FONTS = Path(r"C:\Users\I587436\.agents\skills\canvas-design\canvas-fonts")

W, H = 1800, 2400
PALETTE = {
    "bg_top": (6, 16, 29, 255),
    "bg_bottom": (12, 30, 48, 255),
    "panel": (14, 24, 40, 238),
    "panel_strong": (9, 18, 31, 244),
    "panel_soft": (20, 34, 56, 232),
    "ink": (245, 239, 228, 255),
    "ink_soft": (226, 233, 241, 255),
    "ink_faint": (173, 186, 205, 255),
    "teal": (99, 223, 213, 255),
    "copper": (203, 124, 89, 255),
    "mist": (143, 176, 215, 255),
    "cream": (245, 239, 228, 255),
}

DISPLAY = FONTS / "BricolageGrotesque-Bold.ttf"
BODY = FONTS / "CrimsonPro-Regular.ttf"
ITALIC = FONTS / "CrimsonPro-Italic.ttf"
NUMBER = FONTS / "Boldonse-Regular.ttf"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def blend(a: tuple[int, int, int, int], b: tuple[int, int, int, int], t: float) -> tuple[int, int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(4))


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), trial, font=fnt)
        if box[2] - box[0] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_lines(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: list[str],
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    gap: int = 8,
) -> int:
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        box = draw.textbbox((x, y), line, font=fnt)
        y = box[3] + gap
    return y


def paragraph(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    width: int,
    gap: int = 8,
) -> int:
    return draw_lines(draw, x, y, wrap(draw, text, fnt, width), fnt, fill, gap)


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    radius: int = 34,
    width: int = 2,
    shadow: bool = True,
) -> None:
    x1, y1, x2, y2 = box
    if shadow:
        draw.rounded_rectangle(
            (x1 + 12, y1 + 18, x2 + 12, y2 + 18),
            radius=radius,
            fill=(0, 0, 0, 70),
        )
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def label_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fnt: ImageFont.ImageFont,
    text_fill: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    radius: int = 22,
    pad_x: int = 22,
    pad_y: int = 16,
    line_spacing: int = 4,
) -> None:
    multiline = "\n" in text
    if multiline:
        bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=line_spacing)
    else:
        bbox = draw.textbbox((0, 0), text, font=fnt)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    width = text_width + pad_x * 2
    height = text_height + pad_y * 2
    rounded_panel(draw, (x, y, x + width, y + height), fill=fill, outline=outline, radius=radius, width=2, shadow=False)
    text_x = x + (width - text_width) / 2 - bbox[0]
    text_y = y + (height - text_height) / 2 - bbox[1]
    if multiline:
        draw.multiline_text((text_x, text_y), text, font=fnt, fill=text_fill, spacing=line_spacing, align="center")
    else:
        draw.text((text_x, text_y), text, font=fnt, fill=text_fill)


def base_canvas(accent: tuple[int, int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (W, H), PALETTE["bg_top"])
    draw = ImageDraw.Draw(img, "RGBA")

    for y in range(H):
        t = y / (H - 1)
        draw.line((0, y, W, y), fill=blend(PALETTE["bg_top"], PALETTE["bg_bottom"], t), width=1)

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.ellipse((-180, -120, 700, 740), fill=accent[:3] + (40,))
    gd.ellipse((1100, -120, 1920, 700), fill=PALETTE["mist"][:3] + (28,))
    gd.ellipse((760, 1700, 1760, 2640), fill=PALETTE["copper"][:3] + (26,))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(140)))

    grid = ImageDraw.Draw(img, "RGBA")
    for x in range(84, W, 96):
        grid.line((x, 0, x, H), fill=(255, 255, 255, 12), width=1)
    for y in range(84, H, 96):
        grid.line((0, y, W, y), fill=(255, 255, 255, 12), width=1)

    draw.rounded_rectangle((52, 52, W - 52, H - 52), radius=52, outline=(255, 255, 255, 28), width=2)
    draw.rounded_rectangle((88, 88, W - 88, H - 88), radius=36, outline=(255, 255, 255, 18), width=1)

    noise = Image.effect_noise((W, H), 10).convert("L")
    mask = noise.point(lambda p: 20 if p > 132 else 0)
    grain = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    grain.putalpha(mask)
    img.alpha_composite(grain)
    return img


def header(
    draw: ImageDraw.ImageDraw,
    level: int,
    title: str,
    subtitle: str,
    accent: tuple[int, int, int, int],
    chip_label: str,
) -> None:
    chip_font = font(DISPLAY, 28)
    title_font = font(DISPLAY, 108)
    sub_font = font(ITALIC, 40)
    meta_font = font(BODY, 24)
    num_font = font(NUMBER, 350)

    chip_box = (108, 96, 596, 158)
    rounded_panel(draw, chip_box, fill=PALETTE["panel_strong"], outline=accent[:3] + (150,), radius=28, width=2, shadow=False)
    draw.text((132, 110), chip_label, font=chip_font, fill=PALETTE["cream"])

    title_end = paragraph(draw, 112, 206, title, title_font, PALETTE["ink"], 960, gap=0)
    subtitle_end = paragraph(draw, 116, title_end + 20, subtitle, sub_font, PALETTE["ink_soft"], 980, gap=10)
    rule_y = subtitle_end + 18
    draw.line((112, rule_y, 1092, rule_y), fill=accent[:3] + (160,), width=4)
    draw.line((112, rule_y + 26, 668, rule_y + 26), fill=(255, 255, 255, 42), width=2)
    draw.text((1130, 116), f"LEVEL {level:02d}", font=meta_font, fill=PALETTE["ink_faint"])
    draw.text((1080, 178), "Adaptive Clinical Recruitment", font=meta_font, fill=PALETTE["ink_soft"])
    draw.text((1290, 120), f"0{level}", font=num_font, fill=(245, 239, 228, 16))


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    kicker: str,
    title: str,
    body: list[str],
    accent: tuple[int, int, int, int],
) -> None:
    rounded_panel(draw, box, fill=PALETTE["panel_soft"], outline=accent[:3] + (92,), radius=32, width=2)
    x1, y1, x2, y2 = box
    kick_font = font(DISPLAY, 24)
    title_font = font(DISPLAY, 38)
    body_font = font(BODY, 28)
    draw.text((x1 + 30, y1 + 28), kicker.upper(), font=kick_font, fill=PALETTE["ink_faint"])
    paragraph(draw, x1 + 30, y1 + 74, title, title_font, PALETTE["ink"], x2 - x1 - 60, gap=4)
    y = y1 + 150
    for item in body:
        y = paragraph(draw, x1 + 30, y, item, body_font, PALETTE["ink_soft"], x2 - x1 - 60, gap=8) + 12


def stat_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    label: str,
    accent: tuple[int, int, int, int],
) -> None:
    rounded_panel(draw, box, fill=PALETTE["panel_strong"], outline=accent[:3] + (110,), radius=28, width=2)
    x1, y1, x2, y2 = box
    draw.text((x1 + 26, y1 + 20), value, font=font(DISPLAY, 68), fill=PALETTE["cream"])
    paragraph(draw, x1 + 30, y1 + 104, label, font(BODY, 28), PALETTE["ink_soft"], x2 - x1 - 40, gap=4)


def footer(
    draw: ImageDraw.ImageDraw,
    text_left: str,
    text_mid: str,
    text_right: str,
    accent: tuple[int, int, int, int],
) -> None:
    foot_font = font(BODY, 24)
    y1, y2 = 2176, 2328
    blocks = [
        ((104, y1, 612, y2), text_left),
        ((646, y1, 1166, y2), text_mid),
        ((1200, y1, 1698, y2), text_right),
    ]
    for box, text in blocks:
        rounded_panel(draw, box, fill=PALETTE["panel_strong"], outline=accent[:3] + (84,), radius=24, width=2, shadow=False)
        paragraph(draw, box[0] + 22, box[1] + 24, text, foot_font, PALETTE["ink_soft"], box[2] - box[0] - 44, gap=4)
    draw.line((104, 2148, 1698, 2148), fill=accent[:3] + (140,), width=3)


def funnel_focus(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], accent: tuple[int, int, int, int]) -> None:
    rounded_panel(draw, box, fill=PALETTE["panel_soft"], outline=(255, 255, 255, 40), radius=38, width=2)
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = y1 + 540
    for radius, col, width in [
        (300, PALETTE["mist"][:3] + (96,), 4),
        (224, accent[:3] + (160,), 6),
        (148, PALETTE["copper"][:3] + (118,), 4),
    ]:
        draw.arc((cx - radius, cy - radius, cx + radius, cy + radius), -78, 252, fill=col, width=width)
    for deg in range(-78, 253, 15):
        rad = math.radians(deg)
        ox = cx + math.cos(rad) * 312
        oy = cy + math.sin(rad) * 312
        ix = cx + math.cos(rad) * 292
        iy = cy + math.sin(rad) * 292
        draw.line((ix, iy, ox, oy), fill=(255, 255, 255, 42), width=2)

    core_font = font(DISPLAY, 48)
    draw.ellipse((cx - 120, cy - 120, cx + 120, cy + 120), fill=(10, 18, 33, 255), outline=accent[:3] + (150,), width=3)
    for idx, line in enumerate(["ONE", "EPISODE"]):
        bbox = draw.textbbox((0, 0), line, font=core_font)
        draw.text((cx - (bbox[2] - bbox[0]) / 2, cy - 56 + idx * 56), line, font=core_font, fill=PALETTE["ink"])

    labels = [
        ("SCREEN", x1 + 158, y1 + 86),
        ("RECONTACT", x1 + 354, y1 + 380),
        ("ALLOCATE", x1 + 104, y1 + 824),
        ("RETAIN", x1 + 396, y1 + 824),
    ]
    label_font = font(DISPLAY, 30)
    for text, bx, by in labels:
        label_chip(draw, bx, by, text, label_font, PALETTE["cream"], PALETTE["panel_strong"], accent[:3] + (92,))

    desc_box = (x1 + 36, y2 - 238, x2 - 36, y2 - 38)
    rounded_panel(draw, desc_box, fill=PALETTE["panel_strong"], outline=(255, 255, 255, 36), radius=26, width=2, shadow=False)
    paragraph(
        draw,
        desc_box[0] + 26,
        desc_box[1] + 24,
        "A single run advances through candidate screening, consent pressure, site allocation, and retention risk before the grader emits final_score in (0, 1).",
        font(BODY, 28),
        PALETTE["ink_soft"],
        desc_box[2] - desc_box[0] - 52,
        gap=8,
    )


def phase_focus(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], accent: tuple[int, int, int, int]) -> None:
    rounded_panel(draw, box, fill=PALETTE["panel_soft"], outline=(255, 255, 255, 40), radius=38, width=2)
    x1, y1, x2, y2 = box
    draw.text((x1 + 34, y1 + 28), "PHASE LADDER", font=font(DISPLAY, 34), fill=PALETTE["ink"])
    line_x = x1 + 134
    draw.line((line_x, y1 + 156, line_x, y2 - 150), fill=accent[:3] + (160,), width=6)
    nodes = [
        ("screening", "new candidates appear in available_patients"),
        ("conversion", "follow-up pressure surfaces in recontact_candidates"),
        ("allocation", "consented patients move into allocation_candidates"),
        ("retention", "dropout risk remains active after enrollment"),
        ("recovery", "constraints and bottlenecks alter what should happen next"),
    ]
    for idx, (name, desc) in enumerate(nodes):
        cy = y1 + 210 + idx * 320
        draw.ellipse((line_x - 28, cy - 28, line_x + 28, cy + 28), fill=(10, 18, 33, 255), outline=accent[:3] + (180,), width=4)
        panel = (x1 + 190, cy - 72, x2 - 38, cy + 86)
        rounded_panel(draw, panel, fill=PALETTE["panel_strong"], outline=accent[:3] + (74,), radius=24, width=2, shadow=False)
        draw.text((panel[0] + 24, panel[1] + 18), name.upper(), font=font(DISPLAY, 34), fill=PALETTE["ink"])
        paragraph(draw, panel[0] + 26, panel[1] + 68, desc, font(BODY, 28), PALETTE["ink_soft"], panel[2] - panel[0] - 52, gap=6)


def action_focus(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], accent: tuple[int, int, int, int]) -> None:
    rounded_panel(draw, box, fill=PALETTE["panel_soft"], outline=(255, 255, 255, 40), radius=38, width=2)
    x1, y1, x2, y2 = box
    draw.text((x1 + 34, y1 + 28), "IMPLEMENTED INTERFACE", font=font(DISPLAY, 34), fill=PALETTE["ink"])
    cx = (x1 + x2) // 2
    cy = y1 + 430
    draw.ellipse((cx - 126, cy - 126, cx + 126, cy + 126), fill=(10, 18, 33, 255), outline=accent[:3] + (190,), width=4)
    center_font = font(DISPLAY, 42)
    for idx, line in enumerate(["8", "ACTIONS"]):
        bbox = draw.textbbox((0, 0), line, font=center_font)
        draw.text((cx - (bbox[2] - bbox[0]) / 2, cy - 60 + idx * 56), line, font=center_font, fill=PALETTE["ink"])
    pills = [
        "screen_patient",
        "recontact",
        "allocate_to_site",
        "adjust_strategy",
        "plan_next_phase",
        "summarize_and_index",
        "retrieve_relevant_\nhistory",
        "stop_recruitment",
    ]
    pill_font = font(DISPLAY, 18)
    coords = [
        (x1 + 48, y1 + 180),
        (x1 + 330, y1 + 180),
        (x1 + 36, y1 + 650),
        (x1 + 320, y1 + 650),
        (x1 + 54, y1 + 870),
        (x1 + 302, y1 + 870),
        (x1 + 64, y1 + 1090),
        (x1 + 318, y1 + 1090),
    ]
    for (px, py), text in zip(coords, pills):
        label_chip(draw, px, py, text.upper(), pill_font, PALETTE["cream"], PALETTE["panel_strong"], accent[:3] + (84,), pad_x=14, pad_y=16)
    paragraph(
        draw,
        x1 + 38,
        y2 - 220,
        "The corrected docs and paper use the exact action surface exported by models.py and training/neural_policy.py.",
        font(BODY, 28),
        PALETTE["ink_soft"],
        x2 - x1 - 76,
        gap=8,
    )


def results_focus(
    draw: ImageDraw.ImageDraw,
    left_box: tuple[int, int, int, int],
    right_box: tuple[int, int, int, int],
    accent: tuple[int, int, int, int],
) -> None:
    rounded_panel(draw, left_box, fill=PALETTE["panel_soft"], outline=(255, 255, 255, 40), radius=38, width=2)
    rounded_panel(draw, right_box, fill=PALETTE["panel_soft"], outline=(255, 255, 255, 40), radius=38, width=2)
    draw.text((left_box[0] + 34, left_box[1] + 28), "MEAN SCORE / 95% CI", font=font(DISPLAY, 34), fill=PALETTE["ink"])
    rows = [
        ("HCAPO", 0.2215, "[0.2100, 0.2303]"),
        ("KLong", 0.2152, "[0.1977, 0.2286]"),
        ("MemexRL", 0.2148, "[0.1943, 0.2352]"),
        ("MiRA", 0.2094, "[0.2023, 0.2165]"),
    ]
    base_y = left_box[1] + 170
    max_score = 0.24
    for idx, (name, score, ci) in enumerate(rows):
        y = base_y + idx * 250
        draw.text((left_box[0] + 36, y), name, font=font(DISPLAY, 36), fill=PALETTE["ink"])
        draw.text((left_box[0] + 258, y - 8), f"{score:.4f}", font=font(DISPLAY, 48), fill=PALETTE["cream"])
        track = (left_box[0] + 36, y + 88, left_box[0] + 700, y + 124)
        draw.rounded_rectangle(track, radius=18, fill=(245, 239, 228, 18))
        fill_width = int((score / max_score) * (track[2] - track[0]))
        draw.rounded_rectangle((track[0], track[1], track[0] + fill_width, track[3]), radius=18, fill=accent[:3] + (220,))
        draw.text((left_box[0] + 36, y + 146), ci, font=font(BODY, 28), fill=PALETTE["ink_soft"])

    draw.text((right_box[0] + 34, right_box[1] + 28), "PAIRWISE TESTS", font=font(DISPLAY, 34), fill=PALETTE["ink"])
    pvals = [
        ("HCAPO vs MiRA", "p = 0.1823"),
        ("HCAPO vs KLong", "p = 0.3849"),
        ("HCAPO vs MemexRL", "p = 0.6370"),
        ("MiRA vs KLong", "p = 0.6674"),
        ("MiRA vs MemexRL", "p = 0.6656"),
        ("KLong vs MemexRL", "p = 0.9756"),
    ]
    label_font = font(DISPLAY, 26)
    value_font = font(BODY, 28)
    row_height = 108
    y = right_box[1] + 118
    for label, ptext in pvals:
        draw.text((right_box[0] + 36, y), label, font=label_font, fill=PALETTE["ink"])
        draw.text((right_box[0] + 36, y + 38), ptext, font=value_font, fill=PALETTE["ink_soft"])
        draw.line((right_box[0] + 36, y + row_height - 14, right_box[2] - 36, y + row_height - 14), fill=(255, 255, 255, 30), width=2)
        y += row_height

    verdict = (right_box[0] + 36, y + 18, right_box[2] - 36, right_box[3] - 36)
    rounded_panel(draw, verdict, fill=(11, 63, 69, 198), outline=accent[:3] + (140,), radius=28, width=2, shadow=False)
    draw.text((verdict[0] + 26, verdict[1] + 24), "VERDICT", font=font(DISPLAY, 28), fill=PALETTE["ink_faint"])
    verdict_head_end = paragraph(draw, verdict[0] + 26, verdict[1] + 70, "No pairwise comparison reaches p < 0.05.", font(DISPLAY, 34), PALETTE["ink"], verdict[2] - verdict[0] - 52, gap=4)
    paragraph(
        draw,
        verdict[0] + 26,
        verdict_head_end + 14,
        "The corrected benchmark now supports a conservative read: active, reproducible, and not yet clearly separated by the current baseline suite.",
        font(BODY, 26),
        PALETTE["ink_soft"],
        verdict[2] - verdict[0] - 52,
        gap=8,
    )


def reviewer_focus(
    draw: ImageDraw.ImageDraw,
    left_box: tuple[int, int, int, int],
    right_box: tuple[int, int, int, int],
    accent: tuple[int, int, int, int],
) -> None:
    rounded_panel(draw, right_box, fill=PALETTE["panel_soft"], outline=(255, 255, 255, 40), radius=38, width=2)
    draw.text((right_box[0] + 34, right_box[1] + 28), "VERIFICATION", font=font(DISPLAY, 34), fill=PALETTE["ink"])
    stats = [
        ("30/30", "integration checks"),
        ("76/76", "test_env.py"),
        ("43/43", "test_agents.py"),
        ("109/109", "test_research_modules.py"),
        ("77/77", "test_local_serving.py"),
        ("9 pages", "anonymous paper build"),
    ]
    y = right_box[1] + 122
    for value, label in stats:
        panel = (right_box[0] + 34, y, right_box[2] - 34, y + 122)
        rounded_panel(draw, panel, fill=PALETTE["panel_strong"], outline=accent[:3] + (76,), radius=24, width=2, shadow=False)
        draw.text((panel[0] + 20, panel[1] + 18), value, font=font(DISPLAY, 44), fill=PALETTE["cream"])
        draw.text((panel[0] + 24, panel[1] + 72), label, font=font(BODY, 28), fill=PALETTE["ink_soft"])
        y += 142

    draw.text((right_box[0] + 34, y + 18), "LOOK HERE FIRST", font=font(DISPLAY, 30), fill=PALETTE["ink_faint"])
    path_y = y + 72
    for item in ["README.md", "data/sweep_results/neurips_report.md", "paper/main.pdf", "docs/images/"]:
        panel = (right_box[0] + 34, path_y, right_box[2] - 34, path_y + 84)
        rounded_panel(draw, panel, fill=PALETTE["panel_strong"], outline=accent[:3] + (84,), radius=22, width=2, shadow=False)
        draw.text((panel[0] + 22, panel[1] + 24), item, font=font(DISPLAY, 24), fill=PALETTE["cream"])
        path_y += 102

    titles = [
        ("Strongest claim", "Typed, corrected, and reproducible.", "The environment path, docs, diagrams, sweep outputs, and paper now tell the same conservative story."),
        ("Do not overread", "This is not a finished leaderboard.", "The current means are close, variance remains non-trivial, and stronger training budgets are still needed."),
        ("Scope boundary", "Core path first.", "Center claims on env.py, models.py, the baselines, full_sweep.py, reproducibility.py, tests, and generated artifacts."),
        ("Next work", "Sharper separation, not louder language.", "More seeds, larger budgets, and ablations over state channels or action families should come next."),
    ]
    gap = 24
    card_h = (left_box[3] - left_box[1] - gap) // 2
    card_w = (left_box[2] - left_box[0] - gap) // 2
    positions = [
        (left_box[0], left_box[1], left_box[0] + card_w, left_box[1] + card_h),
        (left_box[0] + card_w + gap, left_box[1], left_box[2], left_box[1] + card_h),
        (left_box[0], left_box[1] + card_h + gap, left_box[0] + card_w, left_box[3]),
        (left_box[0] + card_w + gap, left_box[1] + card_h + gap, left_box[2], left_box[3]),
    ]
    for box, (kicker, title, body) in zip(positions, titles):
        card(draw, box, kicker, title, [body], accent)


def level_one(draw: ImageDraw.ImageDraw, accent: tuple[int, int, int, int]) -> None:
    card(draw, (104, 620, 1030, 980), "Core loop", "A benchmark for sequential trial planning.", ["One run carries an agent through screening, follow-up, site allocation, and retention pressure over 180 simulated steps."], accent)
    stat_box(draw, (104, 1018, 390, 1216), "180", "simulated steps", accent)
    stat_box(draw, (424, 1018, 710, 1216), "3", "public tasks", accent)
    stat_box(draw, (744, 1018, 1030, 1216), "8", "implemented actions", accent)
    card(draw, (104, 1260, 1030, 1810), "What the agent sees", "Typed observations plus a final grade.", ["Action-specific candidate pools, site metrics, milestone state, plan state, memory state, token signals, and counterfactual hints stay visible during the episode.", "Episodes end with final_score in (0, 1), so step-level reward and end-of-run grading can be inspected together."], accent)
    funnel_focus(draw, (1084, 620, 1698, 1810), accent)
    footer(draw, "Core truth: 180 steps / 3 tasks / 8 actions / 37 training features.", "Interpretation: workflow benchmark first, leaderboard second.", "Source: corrected paper, regenerated charts, and fresh 5-seed sweep.", accent)


def level_two(draw: ImageDraw.ImageDraw, accent: tuple[int, int, int, int]) -> None:
    card(draw, (104, 620, 1030, 980), "Candidate routing", "Each action pulls from its own pool.", ["screen_patient uses available_patients.", "recontact uses recontact_candidates.", "allocate_to_site uses allocation_candidates plus a site_id."], accent)
    card(draw, (104, 1016, 1030, 1386), "Long-horizon signals", "The environment keeps delayed state in view.", ["milestones, active_constraints, delayed_effects_pending, uncertainty_components, and token_efficiency_score all remain observable."], accent)
    card(draw, (104, 1422, 1030, 1810), "Planning and memory", "The benchmark exposes internal coordination surfaces.", ["current_plan, indexed_memory_summary, retrieved_memory_context, active_milestone, and counterfactual_hint let agents act beyond one-step reflexes."], accent)
    phase_focus(draw, (1084, 620, 1698, 1810), accent)
    footer(draw, "Level 02 / mechanics: candidate pools, delayed effects, milestones, and explicit recovery.", "Quiet reference: the corrected evaluation path matters because interface truth changes what results mean.", "Read with: models.py, env.py, training/neural_policy.py.", accent)


def level_three(draw: ImageDraw.ImageDraw, accent: tuple[int, int, int, int]) -> None:
    card(draw, (104, 620, 1030, 980), "Core files", "The main benchmark path is compact enough to audit.", ["models.py defines the typed interface.", "env.py implements the dynamics and reward path.", "training/neural_policy.py exports ACTION_SPACE and STATE_DIM = 37."], accent)
    card(draw, (104, 1016, 1030, 1386), "Repo baselines", "Four trainable baselines share one backbone.", ["HCAPO, MiRA, KLong, and MemexRL all sit on the same pure-NumPy actor-critic infrastructure."], accent)
    card(draw, (104, 1422, 1030, 1810), "Scope discipline", "The corrected docs separate core benchmark claims from auxiliary scaffolds.", ["The paper and public docs now avoid treating every research helper as a validated contribution."], accent)
    action_focus(draw, (1084, 620, 1698, 1810), accent)
    footer(draw, "Implementation truth: 8 actions and 37 features are exported by code, not by marketing copy.", "Repo baselines are presented as repo baselines, not as strict external replications.", "Read with: full_sweep.py, reproducibility.py, research/methods/.", accent)


def level_four(draw: ImageDraw.ImageDraw, accent: tuple[int, int, int, int]) -> None:
    results_focus(draw, (104, 620, 1106, 1810), (1140, 620, 1698, 1810), accent)
    footer(draw, "Fresh sweep: 5 seeds / 30 training episodes per seed / 15 evaluation episodes per seed.", "Result: HCAPO is highest mean at 0.2215, but confidence intervals overlap and no pairwise test reaches p < 0.05.", "Use this as a stress test narrative, not as proof of decisive baseline dominance.", accent)


def level_five(draw: ImageDraw.ImageDraw, accent: tuple[int, int, int, int]) -> None:
    reviewer_focus(draw, (104, 620, 1030, 1810), (1084, 620, 1698, 1810), accent)
    footer(draw, "Strongest claim: typed, corrected, regenerated, and reproducible.", "Weakest claim: decisive empirical separation among the current four baselines.", "Next work: more seeds, larger training budgets, and sharper ablations over benchmark surfaces.", accent)


POSTER_SPECS: list[tuple[int, str, str, str, tuple[int, int, int, int], Callable[[ImageDraw.ImageDraw, tuple[int, int, int, int]], None]]] = [
    (1, "beginner", "What This Benchmark Is", "Long-horizon trial planning, reduced to an interface you can test and a score you can inspect.", PALETTE["teal"], level_one),
    (2, "mechanics", "How The Environment Moves", "A patient funnel with delayed consequence, typed state, and action-specific routing.", PALETTE["mist"], level_two),
    (3, "implementation", "What Is Actually Implemented", "The corrected docs, code path, and baseline surface after the benchmark audit.", PALETTE["copper"], level_three),
    (4, "results", "What The Fresh Sweep Says", "The corrected 5-seed run narrows the story: active benchmark, clustered baselines, no clear statistical winner.", PALETTE["teal"], level_four),
    (5, "reviewer", "How To Read It Critically", "Treat the repository as a reproducible evaluation bundle first, and a leaderboard only with caution.", PALETTE["mist"], level_five),
]


def main() -> None:
    pdf_images: list[Image.Image] = []
    for level, slug, title, subtitle, accent, render_fn in POSTER_SPECS:
        image = base_canvas(accent)
        draw = ImageDraw.Draw(image, "RGBA")
        header(draw, level, title, subtitle, accent, f"Protocol Cartography / Level {level:02d}")
        render_fn(draw, accent)
        png_path = OUT / f"poster_level_{level}_{slug}.png"
        image.save(png_path)
        pdf_images.append(image.convert("RGB"))
        print(png_path.relative_to(ROOT).as_posix())

    pdf_path = OUT / "adaptive_clinical_recruitment_posters.pdf"
    pdf_images[0].save(pdf_path, save_all=True, append_images=pdf_images[1:])
    print(pdf_path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
