from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "generated_figures"
SOURCE = REPO / "analysis" / "_shared"
MAPPING = REPO / "source_data" / "supplementary" / "stimulus_mapping.csv"
STIMULUS_INDEX = REPO / "source_data" / "supplementary" / "representative_stimuli.csv"

EEG = "#009E73"
MEG = "#785EF0"
DINO = "#0072B2"
CLIP = "#009E73"
SIGLIP = "#CC79A7"
ACCENT = "#D55E00"
INK = "#111111"
MID = "#6E6E6E"
LIGHT = "#DEDEDE"
PALE = "#F5F5F5"
WHITE = "#FFFFFF"

CATEGORY_COLORS = ["#4E79A7", "#E15759", "#76B7B2", "#B07AA1", "#EDC948", "#9C755F"]

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    }
)


def upper_to_matrix(vector: np.ndarray, n: int = 72) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    matrix = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n, 1)
    matrix[iu] = vector
    matrix[(iu[1], iu[0])] = vector
    np.fill_diagonal(matrix, np.nan)
    return matrix


def build_data_cache() -> dict[str, np.ndarray]:
    cache = REPO / "source_data" / "main" / "figure1.npz"
    if not cache.exists():
        raise FileNotFoundError(f"Missing figure source data: {cache}")
    return dict(np.load(cache, allow_pickle=True))


def arrow(ax, xy0, xy1, color=INK, lw=0.75, rad=0.0, both=False, zorder=3):
    style = "<->" if both else "-|>"
    patch = FancyArrowPatch(
        xy0,
        xy1,
        transform=ax.transAxes,
        arrowstyle=style,
        mutation_scale=6.5,
        lw=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)


def module_box(ax, xy, wh, label, face, edge, fontsize=5.5, bold=False):
    box = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        transform=ax.transAxes,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.65,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=INK,
        zorder=3,
    )


def draw_rdm(fig, parent_ax, bounds, vector, edge, label=None, label_color=None, cmap=None):
    x, y, w, h = bounds
    box = parent_ax.get_position()
    ax = fig.add_axes([box.x0 + x * box.width, box.y0 + y * box.height, w * box.width, h * box.height])
    matrix = upper_to_matrix(vector)
    cm = cmap or LinearSegmentedColormap.from_list("rdm_grey", ["#FFFFFF", "#202020"])
    cm = cm.copy()
    cm.set_bad(WHITE)
    ax.imshow(matrix, cmap=cm, vmin=0, vmax=1, interpolation="nearest", rasterized=True)
    for pos in range(12, 72, 12):
        ax.axhline(pos - 0.5, color=WHITE, lw=0.35)
        ax.axvline(pos - 0.5, color=WHITE, lw=0.35)
    for spine in ax.spines.values():
        spine.set_color(edge)
        spine.set_linewidth(0.75)
    ax.set_xticks([])
    ax.set_yticks([])
    if label:
        ax.text(-0.10, 0.50, label, transform=ax.transAxes, ha="right", va="center", fontsize=5.5, color=label_color or edge, fontweight="bold")
    return ax


def draw_colorbar(fig, parent_ax, bounds):
    x, y, w, h = bounds
    box = parent_ax.get_position()
    cax = fig.add_axes([box.x0 + x * box.width, box.y0 + y * box.height, w * box.width, h * box.height])
    cmap = LinearSegmentedColormap.from_list("rdm_grey_bar", ["#FFFFFF", "#202020"])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 1), orientation="horizontal")
    cax.set_xticks([0, 1], ["similar", "dissimilar"])
    cax.tick_params(axis="x", labelsize=4.8, length=0, pad=1)
    for spine in cax.spines.values():
        spine.set_linewidth(0.35)
        spine.set_edgecolor(MID)
    cax.set_title("ranked dissimilarity", fontsize=4.8, color=MID, pad=1.5)


def draw_timeline(ax):
    x0, x1 = 0.265, 0.535
    y_eeg, y_meg = 0.69, 0.49
    h = 0.095
    def tx(ms):
        return x0 + (x1 - x0) * (ms / 500.0)
    for y in (y_eeg, y_meg):
        ax.add_patch(Rectangle((x0, y), x1 - x0, h, transform=ax.transAxes, facecolor=PALE, edgecolor="none"))
    ax.add_patch(Rectangle((tx(64), y_eeg), tx(144) - tx(64), h, transform=ax.transAxes, facecolor="#A8DCCB", edgecolor="none"))
    ax.add_patch(Rectangle((tx(192), y_eeg), tx(320) - tx(192), h, transform=ax.transAxes, facecolor=EEG, edgecolor="none"))
    ax.add_patch(Rectangle((tx(70), y_meg), tx(130) - tx(70), h, transform=ax.transAxes, facecolor="#C5BCF4", edgecolor="none"))
    ax.add_patch(Rectangle((tx(180), y_meg), tx(300) - tx(180), h, transform=ax.transAxes, facecolor=MEG, edgecolor="none"))
    ax.text((tx(64)+tx(144))/2, y_eeg+h/2, "early", transform=ax.transAxes, ha="center", va="center", fontsize=5.2, color="#275F51")
    ax.text((tx(192)+tx(320))/2, y_eeg+h/2, "late", transform=ax.transAxes, ha="center", va="center", fontsize=5.2, color=WHITE, fontweight="bold")
    ax.text(x0-0.010, y_eeg+h/2, "EEG\nn = 10", transform=ax.transAxes, ha="right", va="center", fontsize=5.4, color=EEG, fontweight="bold", linespacing=0.9)
    ax.text(x0-0.010, y_meg+h/2, "MEG\nn = 16", transform=ax.transAxes, ha="right", va="center", fontsize=5.4, color=MEG, fontweight="bold", linespacing=0.9)
    ax.plot([x0, x1], [0.405, 0.405], transform=ax.transAxes, color=INK, lw=0.45, clip_on=False)
    for ms in (0, 200, 400, 500):
        xx = tx(ms)
        ax.plot([xx, xx], [0.397, 0.414], transform=ax.transAxes, color=INK, lw=0.45, clip_on=False)
        ax.text(xx, 0.354, str(ms), transform=ax.transAxes, ha="center", va="top", fontsize=4.9, color=MID)
    ax.text((x0+x1)/2, 0.282, "time from onset (ms)", transform=ax.transAxes, ha="center", va="top", fontsize=5.0, color=MID)


def draw_stimuli(fig, ax):
    rows = pd.read_csv(STIMULUS_INDEX)
    x0, y0, w, h, gap = 0.024, 0.58, 0.028, 0.155, 0.006
    parent = ax.get_position()
    for i, (_, row) in enumerate(rows.iterrows()):
        x = x0 + i * (w + gap)
        thumb = fig.add_axes([parent.x0 + x * parent.width, parent.y0 + y0 * parent.height, w * parent.width, h * parent.height])
        image = plt.imread(str(STIMULUS_INDEX.parent / row["file"]))
        thumb.imshow(image)
        thumb.set_xticks([])
        thumb.set_yticks([])
        for spine in thumb.spines.values():
            spine.set_color("#B6B6B6")
            spine.set_linewidth(0.4)
        ax.add_patch(Rectangle((x, y0-0.035), w, 0.018, transform=ax.transAxes, facecolor=CATEGORY_COLORS[i], edgecolor="none"))
    ax.text(0.024, 0.465, "72 images  ·  6 categories  ·  12 each", transform=ax.transAxes, fontsize=5.25, color=INK, ha="left")
    ax.text(0.024, 0.385, "human and nonhuman faces / body parts", transform=ax.transAxes, fontsize=4.85, color=MID, ha="left")
    ax.text(0.024, 0.325, "natural and artificial objects", transform=ax.transAxes, fontsize=4.85, color=MID, ha="left")


def draw_participant_fold(ax, x, y, label):
    for j, color in enumerate((EEG, MEG)):
        yy = y + (1-j) * 0.055
        for k in range(3):
            ax.add_patch(Rectangle((x + k*0.010, yy + k*0.006), 0.022, 0.032, transform=ax.transAxes, facecolor=WHITE, edgecolor=color, linewidth=0.55))
    ax.text(x+0.022, y-0.018, label, transform=ax.transAxes, ha="center", va="top", fontsize=4.8, color=MID)


def build_figure() -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_data_cache()
    eeg = np.asarray(data["eeg_rank"], dtype=float)
    meg = np.asarray(data["meg_rank"], dtype=float)
    target = np.asarray(data["combined_rank"], dtype=float)
    frozen = np.asarray(data["frozen_rank"], dtype=float)
    adapted = np.asarray(data["adapted_rank"], dtype=float)

    width_in = 183 / 25.4
    height_in = 90 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor=WHITE)
    ax_a = fig.add_axes([0.015, 0.535, 0.97, 0.44])
    ax_b = fig.add_axes([0.015, 0.045, 0.97, 0.43])
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Panel a
    ax_a.text(-0.018, 1.01, "a", transform=ax_a.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")
    ax_a.text(0.005, 0.995, "Two independent cohorts, the same 72 photographs", transform=ax_a.transAxes, fontsize=6.5, va="top", ha="left")
    draw_stimuli(fig, ax_a)
    draw_timeline(ax_a)

    draw_rdm(fig, ax_a, (0.665, 0.59, 0.066, 0.33), eeg, EEG, "late\nEEG", EEG)
    draw_rdm(fig, ax_a, (0.665, 0.12, 0.066, 0.33), meg, MEG, "late\nMEG", MEG)
    ax_a.plot([0.698, 0.698], [0.485, 0.555], transform=ax_a.transAxes, color=INK, lw=0.65, clip_on=False)
    ax_a.scatter([0.698], [0.563], transform=ax_a.transAxes, marker="^", s=7, color=INK, clip_on=False, zorder=4)
    ax_a.scatter([0.698], [0.477], transform=ax_a.transAxes, marker="v", s=7, color=INK, clip_on=False, zorder=4)
    ax_a.text(0.715, 0.51, "shared geometry", transform=ax_a.transAxes, ha="left", va="center", fontsize=5.2, color=INK)
    draw_rdm(fig, ax_a, (0.888, 0.36, 0.080, 0.40), target, INK)
    ax_a.text(0.928, 0.81, "combined target", transform=ax_a.transAxes, ha="center", va="bottom", fontsize=5.6, color=INK, fontweight="bold")
    ax_a.text(0.928, 0.27, "equal EEG and MEG weight", transform=ax_a.transAxes, ha="center", va="top", fontsize=5.0, color=MID)
    arrow(ax_a, (0.742, 0.77), (0.884, 0.61), color=EEG, rad=-0.16, lw=0.8)
    arrow(ax_a, (0.742, 0.28), (0.884, 0.48), color=MEG, rad=0.16, lw=0.8)
    draw_colorbar(fig, ax_a, (0.900, 0.135, 0.058, 0.025))

    # Panel b
    ax_b.text(-0.018, 1.01, "b", transform=ax_b.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")
    ax_b.text(0.005, 0.995, "A small adapter is trained while the backbone remains frozen", transform=ax_b.transAxes, fontsize=6.5, va="top", ha="left")
    draw_rdm(fig, ax_b, (0.035, 0.26, 0.078, 0.38), target, INK)
    ax_b.text(0.074, 0.69, "neural target", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=5.3, color=INK)

    model_rows = [
        (0.70, "DINOv3", "50,368", DINO),
        (0.48, "CLIP", "50,736", CLIP),
        (0.26, "SigLIP", "51,488", SIGLIP),
    ]
    for y, name, count, color in model_rows:
        arrow(ax_b, (0.12, 0.45), (0.195, y+0.055), color=INK, rad=(y-0.48)*0.18, lw=0.65)
        module_box(ax_b, (0.20, y), (0.12, 0.11), name, PALE, "#9A9A9A", fontsize=5.5)
        module_box(ax_b, (0.325, y+0.018), (0.038, 0.074), "", color, color)
        ax_b.text(0.344, y-0.018, f"{count} trainable", transform=ax_b.transAxes, ha="center", va="top", fontsize=4.6, color=MID)
        arrow(ax_b, (0.365, y+0.055), (0.405, y+0.055), color=INK, lw=0.65)
    ax_b.text(0.26, 0.87, "frozen backbone", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=4.9, color=MID)
    ax_b.text(0.344, 0.87, "adapter", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=4.9, color=MID)

    draw_rdm(fig, ax_b, (0.43, 0.34, 0.072, 0.35), frozen, "#8C8C8C")
    draw_rdm(fig, ax_b, (0.535, 0.34, 0.072, 0.35), adapted, DINO)
    ax_b.text(0.466, 0.73, "frozen", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=5.1, color=MID)
    ax_b.text(0.571, 0.73, "adapted", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=5.1, color=INK, fontweight="bold")
    ax_b.text(0.519, 0.24, "DINOv3 example", transform=ax_b.transAxes, ha="center", va="top", fontsize=4.8, color=MID)
    arrow(ax_b, (0.506, 0.515), (0.531, 0.515), color=INK, lw=0.7)

    arrow(ax_b, (0.61, 0.515), (0.645, 0.515), color=INK, lw=0.7)
    ax_b.text(0.725, 0.80, "all reported values held out", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=5.5, color=INK, fontweight="bold")
    draw_participant_fold(ax_b, 0.655, 0.49, "fold A")
    draw_participant_fold(ax_b, 0.735, 0.49, "fold B")
    ax_b.text(0.825, 0.62, "3 image folds", transform=ax_b.transAxes, ha="left", va="center", fontsize=5.1, color=MID)
    for i in range(3):
        ax_b.add_patch(Rectangle((0.825+i*0.028, 0.50), 0.022, 0.050, transform=ax_b.transAxes, facecolor=WHITE, edgecolor="#8C8C8C", linewidth=0.55))
    ax_b.text(0.825, 0.41, "3 seeds", transform=ax_b.transAxes, ha="left", va="center", fontsize=5.1, color=MID)
    ax_b.scatter([0.873, 0.892, 0.911], [0.41]*3, s=7, color=INK, transform=ax_b.transAxes, clip_on=False)
    ax_b.text(0.725, 0.20, "2 participant folds  ×  3 image folds  ×  3 seeds", transform=ax_b.transAxes, ha="center", va="top", fontsize=5.0, color=INK)
    ax_b.text(0.725, 0.10, "test participants and images excluded from fitting", transform=ax_b.transAxes, ha="center", va="top", fontsize=4.9, color=MID)

    png = OUT / "Figure_1_shared_late_geometry_adapter_v002.png"
    pdf = OUT / "Figure_1_shared_late_geometry_adapter_v002.pdf"
    fig.savefig(png, dpi=300, facecolor=WHITE)
    fig.savefig(pdf, facecolor=WHITE)
    plt.close(fig)

    manifest = {
        "figure": "Figure 1",
        "version": "v002",
        "size_mm": [183, 90],
        "png": str(png),
        "pdf": str(pdf),
        "source_images": [str(MAPPING)],
        "source_neural_data": [str(SOURCE / "run_late_crossmodal_source_gate_v001.py")],
        "source_checkpoints": [str(REPO / "source_data" / "checkpoints" / f"late_consensus_seed_{seed}.pt") for seed in (20260722, 20260723, 20260724)],
        "notes": [
            "Late EEG and MEG RDMs are complete group-level geometries, not controlled residuals.",
            "The combined target gives equal contribution to rank-standardized EEG and MEG geometries.",
            "Frozen and adapted model RDMs use the actual DINOv3 source features and the three final DINOv3 checkpoints.",
        ],
    }
    (OUT / "Figure_1_manifest_v002.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return png, pdf


if __name__ == "__main__":
    png_path, pdf_path = build_figure()
    print(png_path)
    print(pdf_path)
