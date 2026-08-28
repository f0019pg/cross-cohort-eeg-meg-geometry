from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "reported" / "multibackbone_adaptation.json"
OUT = REPO / "generated_figures"

EEG = "#009E73"
MEG = "#785EF0"
INK = "#171717"
MID = "#707070"
ZERO = "#8D9398"


def benjamini_hochberg(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def half_density(ax, values: np.ndarray, y: float, color: str, width: float = 0.28) -> None:
    values = np.asarray(values, dtype=float)
    if np.ptp(values) < 1e-12:
        return
    grid = np.linspace(values.min(), values.max(), 180)
    density = gaussian_kde(values)(grid)
    density = width * density / density.max()
    ax.fill_between(grid, y, y + density, color=color, alpha=0.18, linewidth=0)


def add_distribution(ax, metric: dict, p_adjusted: float, y: float, color: str, rng: np.random.Generator) -> None:
    values = np.asarray(metric["values"], dtype=float)
    mean = float(metric["mean"])
    ci = metric["bootstrap_mean_95ci"]
    half_density(ax, values, y, color)
    offsets = rng.uniform(0.025, 0.18, size=len(values))
    ax.vlines(values, y + offsets - 0.025, y + offsets + 0.025, color=color, lw=0.75, alpha=0.85)
    ax.hlines(y - 0.08, ci[0], ci[1], color=INK, lw=1.15)
    ax.vlines([ci[0], ci[1]], y - 0.115, y - 0.045, color=INK, lw=0.8)
    ax.scatter(mean, y - 0.08, marker="D", s=35, facecolor=color, edgecolor=INK, linewidth=0.65, zorder=4)
    ax.text(ax.get_xlim()[1], y + 0.03, f"{mean:.3f}", ha="right", va="center", fontsize=6.6, color=MID)
    ax.text(ax.get_xlim()[1], y + 0.19, significance_label(p_adjusted), ha="right", va="center", fontsize=7.2, fontweight="bold")


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=3)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RESULTS.open(encoding="utf-8-sig") as f:
        results = json.load(f)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    models = [("DINOv3", "DINOv3"), ("CLIP-B32", "CLIP"), ("SigLIP-base", "SigLIP")]
    measurements = [("eeg", "EEG", EEG), ("meg", "MEG", MEG)]
    raw_p = [
        results["individual_backbones"][model][f"{measurement}_alignment_gain"]["exact_two_sided_signflip_p"]
        for measurement, _, _ in measurements
        for model, _ in models
    ]
    adjusted_p = benjamini_hochberg(raw_p)

    fig = plt.figure(figsize=(7.15, 3.35), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 0.9], left=0.085, right=0.985, bottom=0.17, top=0.84, wspace=0.35)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    rng = np.random.default_rng(20260810)

    ax_a.set_xlim(-0.012, 0.094)
    ax_a.axvline(0, color=ZERO, lw=0.7, ls=(0, (2.5, 2.5)))
    y_positions = [5.0, 4.0, 3.0, 1.5, 0.5, -0.5]
    labels = []
    p_index = 0
    for m_index, (measurement, measurement_label, color) in enumerate(measurements):
        for model_index, (result_key, display_name) in enumerate(models):
            metric = results["individual_backbones"][result_key][f"{measurement}_alignment_gain"]
            y = y_positions[m_index * 3 + model_index]
            add_distribution(ax_a, metric, float(adjusted_p[p_index]), y, color, rng)
            labels.append(display_name)
            p_index += 1
        centre = np.mean(y_positions[m_index * 3 : m_index * 3 + 3])
        ax_a.text(-0.011, centre, measurement_label, rotation=90, ha="right", va="center", color=color, fontweight="bold")
    ax_a.axhline(2.25, color="#E5E5E5", lw=0.7)
    ax_a.set_yticks(y_positions, labels)
    ax_a.tick_params(axis="y", length=0, pad=7)
    ax_a.set_ylim(-0.85, 5.55)
    ax_a.set_xlabel(r"Held-out alignment gain, $\Delta\rho$")
    style_axis(ax_a)

    family = results["non_dino_family"]["unique_displacement"]
    family_p = benjamini_hochberg(
        [family["eeg"]["exact_two_sided_signflip_p"], family["meg"]["exact_two_sided_signflip_p"]]
    )
    ax_b.set_xlim(-0.06, 0.17)
    ax_b.axvline(0, color=ZERO, lw=0.7, ls=(0, (2.5, 2.5)))
    for index, (measurement, _, color) in enumerate(measurements):
        add_distribution(ax_b, family[measurement], float(family_p[index]), 1.0 - index, color, rng)
    ax_b.set_yticks([1, 0], ["EEG", "MEG"])
    ax_b.tick_params(axis="y", length=0, pad=7)
    ax_b.set_ylim(-0.4, 1.5)
    ax_b.set_xlabel("Correlation with residual\nneural geometry, r")
    style_axis(ax_b)

    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    fig.text(pos_a.x0 - 0.045, 0.95, "a", fontsize=10, fontweight="bold", ha="left", va="top")
    fig.text(pos_a.x0, 0.95, "Held-out neural-alignment gain", fontsize=8.5, ha="left", va="top")
    fig.text(pos_b.x0 - 0.045, 0.95, "b", fontsize=10, fontweight="bold", ha="left", va="top")
    fig.text(pos_b.x0, 0.95, "Adapter displacement", fontsize=8.5, ha="left", va="top")

    stem = OUT / "Figure_5_adapter_alignment"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
