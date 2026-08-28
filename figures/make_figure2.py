from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "generated_figures"
DISPLAY_DATA = REPO / "source_data" / "main" / "figure2.npz"
RESULTS = REPO / "results" / "reported" / "eeg_temporal_geometry.json"
STIMULUS_INDEX = REPO / "source_data" / "supplementary" / "representative_stimuli.csv"


def read_mapping():
    with STIMULUS_INDEX.open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    categories = [row["category"] for row in rows]
    root = STIMULUS_INDEX.parent
    representative = [root / row["file"] for row in rows]
    return categories, representative


def crop_image(path: Path):
    image = plt.imread(path)
    if image.shape[-1] == 4:
        rgb, alpha = image[..., :3], image[..., 3:4]
        image = rgb * alpha + (1 - alpha)
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def panel_label(ax, label, x=-0.10, y=1.06):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#111111",
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=3)


def add_mean_ci(ax, x, values, color, ci, mean):
    ax.vlines(x, ci[0], ci[1], color="#171717", lw=1.1, zorder=4)
    ax.hlines([ci[0], ci[1]], x - 0.045, x + 0.045, color="#171717", lw=1.0, zorder=4)
    ax.hlines(mean, x - 0.13, x + 0.13, color="#171717", lw=1.5, zorder=5)


def make_figure(rdms, results, categories, representative):
    blue = "#3977A8"
    orange = "#D97926"
    teal = "#198F78"
    gray = "#9AA1A8"
    dark = "#171717"

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.15, 5.35), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.02, 1.30],
        height_ratios=[0.95, 1.0],
        left=0.065,
        right=0.985,
        bottom=0.10,
        top=0.945,
        wspace=0.32,
        hspace=0.38,
    )

    # a | Actual GFP time course with prespecified analysis windows
    ax_a = fig.add_subplot(outer[0, 0])
    ax_a.set_xlim(-120, 520)
    ax_a.set_ylim(-0.18, 1)
    ax_a.axis("off")
    apos = outer[0, 0].get_position(fig)
    fig.text(apos.x0 - 0.031, apos.y1 + 0.013, "a", ha="left", va="bottom", fontsize=10, fontweight="bold", color=dark)
    fig.text(apos.x0, apos.y1 + 0.013, "EEG responses and analysis windows", ha="left", va="bottom", fontsize=8.5, fontweight="bold", color=dark)

    # Actual stimuli are stacked at onset to avoid implying separate early/late image sets.
    image_x = [-92, -66, -40]
    image_y = [0.72, 0.75, 0.78]
    for x, y_img, path in zip(
        image_x,
        image_y,
        [representative[0], representative[2], representative[5]],
    ):
        oi = OffsetImage(crop_image(path), zoom=0.090, interpolation="hanning")
        box = AnnotationBbox(
            oi,
            (x, y_img),
            frameon=True,
            bboxprops={"edgecolor": "#C8CDD1", "linewidth": 0.55, "facecolor": "white"},
            pad=0.04,
            zorder=2 + int((x + 100) / 20),
        )
        ax_a.add_artist(box)
    ax_a.annotate(
        "",
        xy=(0, 0.03),
        xytext=(-22, 0.64),
        arrowprops={"arrowstyle": "-|>", "color": "#777777", "lw": 0.7},
    )

    # The trace is the participant-normalized GFP averaged across the ten EEG
    # participants. It is shown as temporal context and did not define the windows.
    time = rdms["gfp_time"]
    mean = rdms["gfp_mean"]
    sem = rdms["gfp_sem"]
    ax_a.fill_between(
        time,
        np.maximum(0, mean - sem),
        np.minimum(1, mean + sem),
        color="#7E878D",
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax_a.plot(time, mean, color="#50575C", lw=1.25, zorder=2)

    # Light window shading keeps the measured trace visible and avoids turning
    # the exact millisecond limits into prominent figure text.
    ax_a.fill_betweenx([0, 0.84], 64, 144, color=blue, alpha=0.13, lw=0)
    ax_a.fill_betweenx([0, 0.84], 192, 320, color=orange, alpha=0.13, lw=0)
    ax_a.hlines(0, 0, 500, color="#555555", lw=0.75)
    ax_a.vlines(0, 0, 1, color="#555555", lw=0.65)
    for tick in [100, 200, 300, 400, 500]:
        ax_a.vlines(tick, -0.018, 0.018, color="#8C8C8C", lw=0.5)
        ax_a.text(tick, -0.045, str(tick), ha="center", va="top", color="#666666")
    for tick in [0, 0.5, 1.0]:
        ax_a.hlines(tick, -5, 0, color="#8C8C8C", lw=0.5)
        ax_a.text(-10, tick, f"{tick:g}", ha="right", va="center", fontsize=6.2, color="#666666")
    # Keep the onset label below the time axis, away from the stimulus connector,
    # vertical marker and measured EEG trace.
    ax_a.text(6, -0.045, "Onset", ha="left", va="top", fontsize=6.2, color="#4B4B4B")
    ax_a.text(104, 0.89, "Early", ha="center", va="bottom", color=blue, fontweight="bold")
    ax_a.text(256, 0.89, "Late", ha="center", va="bottom", color=orange, fontweight="bold")
    ax_a.text(500, -0.125, "Time (ms)", ha="right", va="top", color="#666666")
    ax_a.text(-62, 0.43, "Normalized GFP", ha="center", va="center", rotation=90, fontsize=6.2, color="#666666")

    # b | Group geometry maps
    bgrid = outer[0, 1].subgridspec(1, 3, wspace=0.16)
    cmap_dist = LinearSegmentedColormap.from_list(
        "dist", ["#295D84", "#7FA9C7", "#D9E4ED", "#F7F8FA"]
    )
    # Keep colour semantics consistent across panels: blue denotes more similar
    # or relatively closer pairs, whereas orange denotes relatively farther pairs.
    cmap_change = LinearSegmentedColormap.from_list(
        "change", ["#3C6FA0", "#E9EFF3", "#FFFFFF", "#F4E6DA", "#C66E2D"]
    )
    dmin, dmax = 0.0, 1.0
    cmax = np.quantile(np.abs(rdms["correction"][np.triu_indices(72, 1)]), 0.98)
    b_axes = []
    titles = ["Early", "Late", "Late residual\nafter early"]
    keys = ["early", "late", "correction"]
    category_centres = np.arange(6) * 12 + 5.5
    category_labels = [
        "Human body parts",
        "Human faces",
        "Non-human body parts",
        "Non-human faces",
        "Natural objects",
        "Artificial objects",
    ]
    for idx, (title, key) in enumerate(zip(titles, keys)):
        ax = fig.add_subplot(bgrid[0, idx])
        b_axes.append(ax)
        if key == "correction":
            im = ax.imshow(
                rdms[key],
                cmap=cmap_change,
                norm=TwoSlopeNorm(vmin=-cmax, vcenter=0, vmax=cmax),
                interpolation="nearest",
                rasterized=True,
            )
        else:
            im = ax.imshow(
                rdms[key],
                cmap=cmap_dist,
                vmin=dmin,
                vmax=dmax,
                interpolation="nearest",
                rasterized=True,
            )
        for boundary in range(12, 72, 12):
            ax.axhline(boundary - 0.5, color="white", lw=0.45, alpha=0.92)
            ax.axvline(boundary - 0.5, color="white", lw=0.45, alpha=0.92)
        ax.set_title(title, pad=4, fontsize=7.3, fontweight="normal")
        ax.set_xticks([])
        if idx == 0:
            ax.set_yticks(category_centres, category_labels)
            ax.tick_params(axis="y", length=0, pad=3, labelsize=5.4, colors="#4B4B4B")
        else:
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    bpos = outer[0, 1].get_position(fig)
    fig.text(
        bpos.x0 - 0.031,
        bpos.y1 + 0.013,
        "b",
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=dark,
    )
    fig.text(
        bpos.x0,
        bpos.y1 + 0.013,
        "EEG object geometry",
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=dark,
    )
    # Compact, explicit color scales directly beneath the heatmaps.
    p0 = b_axes[0].get_position()
    p1 = b_axes[1].get_position()
    p2 = b_axes[2].get_position()
    cb_dist_ax = fig.add_axes([p0.x0, p0.y0 - 0.030, p1.x1 - p0.x0, 0.009])
    cb_dist = mpl.colorbar.ColorbarBase(
        cb_dist_ax,
        cmap=cmap_dist,
        norm=mpl.colors.Normalize(vmin=dmin, vmax=dmax),
        orientation="horizontal",
    )
    cb_dist.set_ticks([])
    cb_dist.outline.set_visible(False)
    cb_dist_ax.invert_xaxis()
    fig.text(p0.x0, p0.y0 - 0.041, "dissimilar", ha="left", va="top", fontsize=6.2, color="#4B4B4B")
    fig.text(p1.x1, p0.y0 - 0.041, "similar", ha="right", va="top", fontsize=6.2, color="#4B4B4B")

    cb_change_ax = fig.add_axes([p2.x0, p2.y0 - 0.030, p2.width, 0.009])
    cb_change = mpl.colorbar.ColorbarBase(
        cb_change_ax,
        cmap=cmap_change,
        norm=TwoSlopeNorm(vmin=-cmax, vcenter=0, vmax=cmax),
        orientation="horizontal",
    )
    cb_change.set_ticks([])
    cb_change.outline.set_visible(False)
    cb_change_ax.invert_xaxis()
    fig.text(p2.x0, p2.y0 - 0.041, "farther (+)", ha="left", va="top", fontsize=6.2, color="#4B4B4B")
    fig.text((p2.x0 + p2.x1) / 2, p2.y0 - 0.041, "0", ha="center", va="top", fontsize=6.2, color="#4B4B4B")
    fig.text(p2.x1, p2.y0 - 0.041, "closer (\N{MINUS SIGN})", ha="right", va="top", fontsize=6.2, color="#4B4B4B")

    # c | Reliability across trial partitions
    ax_c = fig.add_subplot(outer[1, 0])
    cpos = outer[1, 0].get_position(fig)
    fig.text(cpos.x0 - 0.031, cpos.y1 + 0.017, "c", ha="left", va="bottom", fontsize=10, fontweight="bold", color=dark)
    fig.text(cpos.x0, cpos.y1 + 0.017, "Geometry is reliable across trial partitions", ha="left", va="bottom", fontsize=8.5, fontweight="bold", color=dark)
    gates = results["eeg"]["gates"]
    metrics = [
        ("Early", gates["early_reliability"], blue),
        ("Late", gates["late_reliability"], orange),
        ("Late residual\nafter early", gates["correction_reliability"], teal),
    ]
    vals = np.column_stack([m[1]["values"] for m in metrics])
    # These are reliability estimates, not a preregistered paired comparison.
    # Unconnected points avoid implying that differences among windows are the
    # biological endpoint.
    jitter = np.linspace(-0.055, 0.055, vals.shape[0])
    for x, (_, metric, color) in enumerate(metrics):
        y = np.asarray(metric["values"])
        ax_c.scatter(
            x + jitter,
            y,
            s=18,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        add_mean_ci(ax_c, x, y, color, metric["bootstrap_mean_95ci"], metric["mean"])
        ax_c.text(x, 0.735, f"ρ = {metric['mean']:.3f}", ha="center", va="top", color=dark)
    ax_c.axhline(0, color="#8D9398", lw=0.65)
    ax_c.set_xlim(-0.45, 2.45)
    ax_c.set_ylim(-0.03, 0.76)
    ax_c.set_xticks([0, 1, 2], [m[0] for m in metrics])
    ax_c.set_ylabel("Split-half Spearman ρ")
    clean_axis(ax_c)

    # d | Distribution of participant-level leave-one-out correspondence.
    # Each participant is compared with a group estimate from the other nine.
    ax_d = fig.add_subplot(outer[1, 1])
    dpos = outer[1, 1].get_position(fig)
    fig.text(dpos.x0 - 0.035, dpos.y1 + 0.017, "d", ha="left", va="bottom", fontsize=10, fontweight="bold", color=dark)
    fig.text(dpos.x0, dpos.y1 + 0.017, "Late residual geometry is shared across participants", ha="left", va="bottom", fontsize=8.5, fontweight="bold", color=dark)

    shared = gates["shared_correction"]
    values = np.asarray(shared["values"])
    # A narrow deterministic swarm shows the ten participant estimates without
    # assigning meaning to the vertical coordinate.
    order = np.argsort(values)
    offsets = np.linspace(-0.045, 0.045, len(values))
    jitter = np.empty_like(offsets)
    jitter[order] = offsets
    ax_d.scatter(values, jitter, s=31, facecolor=teal, edgecolor="white", linewidth=0.55, zorder=3)
    ci0, ci1 = shared["bootstrap_mean_95ci"]
    summary_y = -0.125
    ax_d.hlines(summary_y, ci0, ci1, color=dark, lw=1.2, zorder=4)
    ax_d.vlines([ci0, ci1], summary_y - 0.025, summary_y + 0.025, color=dark, lw=0.8, zorder=4)
    ax_d.scatter(shared["mean"], summary_y, marker="D", s=28, facecolor=dark, edgecolor="white", linewidth=0.4, zorder=5)
    ax_d.axvline(0, color="#8D9398", lw=0.65)
    ax_d.set_xlim(-0.01, 0.52)
    ax_d.set_ylim(-0.19, 0.13)
    ax_d.set_yticks([])
    ax_d.set_xlabel("Leave-one-participant-out agreement, ρ")
    ax_d.text(0.50, 0.105, "10/10 positive", ha="right", va="top", color="#555555", fontsize=6.4)
    ax_d.text(shared["mean"], summary_y + 0.048, f"mean ρ = {shared['mean']:.3f}", ha="center", va="bottom", color=dark, fontsize=6.4)
    clean_axis(ax_d)
    ax_d.spines["left"].set_visible(False)

    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DISPLAY_DATA, allow_pickle=True) as data:
        rdms = {key: data[key] for key in data.files if key != "categories"}
    with RESULTS.open(encoding="utf-8") as f:
        results = json.load(f)
    categories, representative = read_mapping()

    fig = make_figure(rdms, results, categories, representative)
    stem = OUT / "Figure_2_EEG_temporal_geometry"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
