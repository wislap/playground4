from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from metrics import write_json


DEFAULT_METRICS = [
    "topk_recall",
    "ndcg_at_5",
    "top5_regret",
    "bad_top3_rate",
    "high_value_recall_at_5",
    "mae",
    "spearman",
    "high_conf_precision",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.run_dir / "metrics.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    summary = summarize(rows, split=args.split, metric_names=args.metrics, patience=args.patience)
    write_json(args.run_dir / f"{args.split}_metrics_summary.json", summary)
    plot_paths = render_plots(args.run_dir, rows, split=args.split, metric_names=args.metrics)
    print(json.dumps(summary, indent=2, sort_keys=True))
    for path in plot_paths:
        print(f"[done] wrote {path}")


def summarize(
    rows: list[dict[str, Any]],
    *,
    split: str,
    metric_names: list[str],
    patience: int,
) -> dict[str, Any]:
    summary = {"epochs": len(rows), "split": split, "metrics": {}}
    for metric_name in metric_names:
        values = [float(row[split][metric_name]) for row in rows if metric_name in row[split]]
        if not values:
            continue
        direction = _direction(metric_name)
        best_index = max(range(len(values)), key=lambda idx: direction * values[idx])
        recent = values[-patience:] if len(values) >= patience else values
        recent_delta = direction * (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
        best_to_last_gap = direction * (values[best_index] - values[-1])
        summary["metrics"][metric_name] = {
            "direction": "maximize" if direction > 0 else "minimize",
            "best_epoch": best_index + 1,
            "best": values[best_index],
            "last": values[-1],
            "recent_directional_delta": recent_delta,
            "best_to_last_gap": best_to_last_gap,
            "looks_saturated": abs(recent_delta) < 0.01,
        }
    return summary


def render_plots(
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    split: str,
    metric_names: list[str],
) -> list[Path]:
    records = []
    for row in rows:
        for metric_name in metric_names:
            if metric_name in row[split]:
                records.append(
                    {
                        "epoch": int(row["epoch"]),
                        "metric": metric_name,
                        "value": float(row[split][metric_name]),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    sns.set_theme(style="whitegrid", context="talk")
    metric_count = len(frame["metric"].unique())
    fig_h = max(8.0, metric_count * 2.1)
    grid = sns.relplot(
        data=frame,
        x="epoch",
        y="value",
        col="metric",
        col_wrap=2,
        kind="line",
        marker="o",
        facet_kws={"sharey": False, "sharex": True},
        height=2.35,
        aspect=2.0,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Epoch", "Value")
    grid.figure.set_size_inches(12, fig_h)
    grid.figure.suptitle(f"{run_dir.name} {split} metric trends", y=1.02, fontsize=18)
    for axis in grid.axes.flat:
        axis.xaxis.get_major_locator().set_params(integer=True)
    png_path = run_dir / f"{split}_metrics_trends.png"
    pdf_path = run_dir / f"{split}_metrics_trends.pdf"
    svg_path = run_dir / f"{split}_metrics_trends.svg"
    grid.figure.savefig(png_path, dpi=180, bbox_inches="tight")
    grid.figure.savefig(pdf_path, bbox_inches="tight")
    grid.figure.savefig(svg_path, bbox_inches="tight")
    plt.close(grid.figure)

    dashboard_path = run_dir / f"{split}_metrics_dashboard.png"
    render_dashboard(frame, dashboard_path, run_dir=run_dir, split=split)
    return [png_path, pdf_path, svg_path, dashboard_path]


def render_dashboard(frame: pd.DataFrame, path: Path, *, run_dir: Path, split: str) -> None:
    groups = {
        "Ranking Quality": ["topk_recall", "ndcg_at_5", "high_value_recall_at_5"],
        "Ranking Risk": ["top5_regret", "bad_top3_rate", "agent_bad_top3_rate"],
        "Calibration": ["mae", "spearman", "high_conf_precision"],
    }
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 11), sharex=True)
    for axis, (title, metrics) in zip(axes, groups.items(), strict=True):
        subset = frame[frame["metric"].isin(metrics)]
        sns.lineplot(data=subset, x="epoch", y="value", hue="metric", marker="o", ax=axis)
        axis.set_title(title)
        axis.set_ylabel("Value")
        axis.legend(loc="best", frameon=True)
        axis.xaxis.get_major_locator().set_params(integer=True)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"{run_dir.name} {split} dashboard", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_svg(rows: list[dict[str, Any]], *, split: str, metric_names: list[str]) -> str:
    width = 1100
    panel_h = 150
    left = 58
    right = 24
    top = 34
    bottom = 28
    gap = 20
    panels = []
    for metric_name in metric_names:
        values = [float(row[split][metric_name]) for row in rows if metric_name in row[split]]
        if values:
            panels.append((metric_name, values))
    height = len(panels) * panel_h + max(0, len(panels) - 1) * gap + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;fill:#111827}.axis{stroke:#d1d5db}.line{fill:none;stroke:#2563eb;stroke-width:2}.dot{fill:#2563eb}</style>',
    ]
    plot_w = width - left - right
    for panel_idx, (metric_name, values) in enumerate(panels):
        y0 = 15 + panel_idx * (panel_h + gap)
        plot_h = panel_h - top - bottom
        min_v = min(values)
        max_v = max(values)
        if abs(max_v - min_v) < 1e-9:
            max_v += 1.0
            min_v -= 1.0
        points = []
        for idx, value in enumerate(values):
            x = left + (plot_w * idx / max(1, len(values) - 1))
            y = y0 + top + plot_h * (1.0 - (value - min_v) / (max_v - min_v))
            points.append((x, y))
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.extend(
            [
                f'<text x="{left}" y="{y0 + 16}" font-weight="700">{metric_name}</text>',
                f'<text x="{width - right - 160}" y="{y0 + 16}">min {min_v:.3f} / max {max_v:.3f}</text>',
                f'<line class="axis" x1="{left}" y1="{y0 + top}" x2="{left}" y2="{y0 + top + plot_h}"/>',
                f'<line class="axis" x1="{left}" y1="{y0 + top + plot_h}" x2="{width - right}" y2="{y0 + top + plot_h}"/>',
                f'<polyline class="line" points="{d}"/>',
            ]
        )
        for idx, (x, y) in enumerate(points, start=1):
            parts.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3"><title>epoch {idx}: {values[idx - 1]:.4f}</title></circle>')
        parts.append(f'<text x="{left}" y="{y0 + top + plot_h + 20}">epoch 1</text>')
        parts.append(f'<text x="{width - right - 55}" y="{y0 + top + plot_h + 20}">epoch {len(values)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _direction(metric_name: str) -> int:
    minimize_markers = ("loss", "mae", "mse", "regret", "bad_", "fp_rate")
    return -1 if any(marker in metric_name for marker in minimize_markers) else 1


if __name__ == "__main__":
    main()
