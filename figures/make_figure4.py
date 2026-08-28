from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "generated_figures"
DATA = np.load(REPO / "source_data" / "main" / "figure4.npz")
RESULTS = json.loads((REPO / "results" / "reported" / "figure4.json").read_text(encoding="utf-8"))

EEG = "#009E73"
MEG = "#785EF0"
INK = "#202124"
MID = "#6E6E6E"
ZERO = "#8A8A8A"
RDM_CMAP = LinearSegmentedColormap.from_list(
    "residual", ["#2166AC", "#F7F7F7", "#D55E00"]
)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.7,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
)


def matrix_from_vec(v: np.ndarray, n: int = 72) -> np.ndarray:
    matrix = np.full((n, n), np.nan, dtype=float)
    iu = np.triu_indices(n, 1)
    matrix[iu] = v
    matrix[(iu[1], iu[0])] = v
    np.fill_diagonal(matrix, 0.0)
    return matrix


def panel_label(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(x, y, label, fontsize=10.0, fontweight="bold", va="baseline", ha="left")


def category_boundaries(ax: plt.Axes) -> None:
    for pos in np.arange(12, 72, 12) - 0.5:
        ax.axhline(pos, color="white", lw=0.45, zorder=3)
        ax.axvline(pos, color="white", lw=0.45, zorder=3)


def clean_axis(ax: plt.Axes, zero: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if zero:
        ax.axhline(0, color=ZERO, lw=0.5, ls=(0, (2.5, 2.5)), zorder=0)


def deterministic_swarm(values: np.ndarray, center: float, y_threshold: float = 0.020) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    offsets = np.zeros(len(values))
    order = np.argsort(values)
    candidates = np.array([0.0, -0.08, 0.08, -0.16, 0.16, -0.24, 0.24])
    placed: list[tuple[float, float]] = []
    for idx in order:
        y = values[idx]
        chosen = candidates[-1]
        for candidate in candidates:
            collision = any(
                abs(y - py) < y_threshold and abs(candidate - px) < 0.075
                for py, px in placed
            )
            if not collision:
                chosen = candidate
                break
        offsets[idx] = chosen
        placed.append((y, chosen))
    return center + offsets


def add_condition(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    ci: tuple[float, float],
    ceiling: tuple[float, float] | None,
    ceiling_offset: float = 0.27,
) -> None:
    values = np.asarray(values, dtype=float)
    if ceiling is not None:
        ax.plot(
            [x + ceiling_offset, x + ceiling_offset],
            ceiling,
            color="#D9D9D9",
            lw=5.8,
            solid_capstyle="butt",
            zorder=0,
        )
    xs = deterministic_swarm(values, x)
    ax.scatter(xs, values, s=19.0, color=color, alpha=0.84, linewidths=0, zorder=3)
    mean = float(values.mean())
    ax.plot([x, x], ci, color=INK, lw=1.0, zorder=4)
    ax.scatter([x], [mean], s=47, facecolor="white", edgecolor=INK, linewidth=0.95, zorder=5)


def main() -> None:
    width_mm, height_mm = 183.0, 84.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.00, 1.34],
        left=0.048,
        right=0.985,
        top=0.865,
        bottom=0.180,
        wspace=0.25,
    )
    panel_y = 0.955

    # a: group-level residual relation.
    ga = outer[0, 0].subgridspec(
        2,
        1,
        height_ratios=[0.90, 1.10],
        hspace=0.31,
    )
    ax_rdm = fig.add_subplot(ga[0, 0])
    ax_pair = fig.add_subplot(ga[1, 0])
    panel_label(fig, 0.012, panel_y, "a")
    fig.text(
        0.046,
        panel_y,
        "Late EEG and MEG residual geometry",
        fontsize=8.4,
        va="baseline",
        ha="left",
        color=INK,
    )

    eeg_teacher = np.asarray(DATA["eeg_teacher"])
    meg_teacher = np.asarray(DATA["meg_teacher"])
    vmax = float(np.max(np.abs(np.concatenate([eeg_teacher, meg_teacher]))))
    eeg_mat = matrix_from_vec(eeg_teacher)
    meg_mat = matrix_from_vec(meg_teacher)
    combined = np.triu(eeg_mat, 1) + np.tril(meg_mat, -1)
    np.fill_diagonal(combined, 0.0)
    im = ax_rdm.imshow(
        combined,
        cmap=RDM_CMAP,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_rdm.set_anchor("N")
    category_boundaries(ax_rdm)
    ax_rdm.set_xticks([])
    ax_rdm.set_yticks([])
    for spine in ax_rdm.spines.values():
        spine.set_visible(True)
        spine.set_color("#A0A0A0")
        spine.set_linewidth(0.6)
    ax_rdm.text(
        0.97,
        0.95,
        "EEG",
        transform=ax_rdm.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=EEG,
        fontweight="bold",
    )
    ax_rdm.text(
        0.03,
        0.05,
        "MEG",
        transform=ax_rdm.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=MEG,
        fontweight="bold",
    )
    cax = ax_rdm.inset_axes([1.075, 0.04, 0.065, 0.92])
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_ticks(
        [-vmax, vmax],
        labels=["closer (−)", "farther (+)"],
    )
    cb.ax.tick_params(labelsize=6.4, length=0, pad=2.0)
    cb.outline.set_visible(False)

    ax_pair.hexbin(
        eeg_teacher,
        meg_teacher,
        gridsize=30,
        mincnt=1,
        bins="log",
        cmap=LinearSegmentedColormap.from_list("hex", ["#E6E6E6", "#2C2C2C"]),
        linewidths=0,
    )
    clean_axis(ax_pair)
    ax_pair.set_box_aspect(1.0)
    ax_pair.set_anchor("N")
    ax_pair.set_xlabel("EEG residual (z)")
    ax_pair.set_ylabel("MEG residual (z)")
    ax_pair.text(
        0.03,
        0.96,
        r"$\rho$ = 0.299",
        transform=ax_pair.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=INK,
    )

    # b: participant-level transfer. The permutation audit is retained in the caption.
    ax_b = fig.add_subplot(outer[0, 1])
    panel_label(fig, 0.440, panel_y, "b")
    fig.text(
        0.475,
        panel_y,
        "Bidirectional participant-level correspondence",
        fontsize=8.4,
        va="baseline",
        ha="left",
        color=INK,
    )
    clean_axis(ax_b, zero=True)
    ax_b.set_ylim(-0.135, 0.565)
    ax_b.set_xlim(-0.55, 5.66)
    ax_b.set_yticks([-0.1, 0.0, 0.2, 0.4])
    ax_b.set_ylabel(r"Partial Spearman $\rho$")
    ax_b.set_xticks(
        [0, 1, 3, 4, 5],
        ["All\npairs", "Within\ncategory", "Prestimulus", "All\npairs", "Within\ncategory"],
    )

    cond = RESULTS["conditions"]
    add_condition(
        ax_b,
        0,
        DATA["eeg_scores"],
        EEG,
        tuple(cond["meg_group_to_eeg_all"]["bootstrap_mean_95ci"]),
        tuple(DATA["eeg_all_ceiling"]),
    )
    add_condition(
        ax_b,
        1,
        DATA["eeg_within_scores"],
        EEG,
        tuple(cond["meg_group_to_eeg_within"]["bootstrap_mean_95ci"]),
        tuple(DATA["eeg_within_ceiling"]),
    )
    add_condition(
        ax_b,
        3,
        DATA["pre_scores"],
        MEG,
        tuple(cond["eeg_group_to_meg_pre"]["bootstrap_mean_95ci"]),
        None,
    )
    add_condition(
        ax_b,
        4,
        DATA["meg_scores"],
        MEG,
        tuple(cond["eeg_group_to_meg_all"]["bootstrap_mean_95ci"]),
        tuple(DATA["meg_all_ceiling"]),
    )
    add_condition(
        ax_b,
        5,
        DATA["meg_within_scores"],
        MEG,
        tuple(cond["eeg_group_to_meg_within"]["bootstrap_mean_95ci"]),
        tuple(DATA["meg_within_ceiling"]),
    )
    for pre, late in zip(np.asarray(DATA["pre_scores"]), np.asarray(DATA["meg_scores"])):
        ax_b.plot([3, 4], [pre, late], color=MEG, alpha=0.18, lw=0.5, zorder=1)
    group_label_transform = ax_b.get_xaxis_transform()
    ax_b.text(
        0.5,
        1.012,
        "MEG → EEG",
        transform=group_label_transform,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK,
        fontweight="bold",
        clip_on=False,
    )
    ax_b.text(
        4.0,
        1.012,
        "EEG → MEG",
        transform=group_label_transform,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK,
        fontweight="bold",
        clip_on=False,
    )
    ax_b.axvline(2.0, color="#D8D8D8", lw=0.55, zorder=0)

    ax_b.text(0, 0.285, "**", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax_b.text(1, 0.305, "**", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax_b.text(3, 0.075, "n.s.", ha="center", va="bottom", fontsize=6.6, color=MID)
    ax_b.text(4, 0.255, "***", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax_b.text(5, 0.285, "***", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    bracket_y = 0.218
    ax_b.plot(
        [3, 3, 4, 4],
        [bracket_y - 0.015, bracket_y, bracket_y, bracket_y - 0.015],
        color=INK,
        lw=0.8,
        clip_on=False,
    )
    ax_b.text(3.5, bracket_y + 0.008, "***", ha="center", va="bottom", fontsize=8.1, fontweight="bold")

    png = OUT / "Figure_4_shared_late_geometry_v009.png"
    pdf = OUT / "Figure_4_shared_late_geometry_v009.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    caption = (
        "**Fig. 4 | Later object geometry is shared across independent EEG and MEG cohorts.** "
        "**a,** The upper matrix combines the group-average late EEG and MEG residual representational dissimilarity matrices in its upper and lower triangles, respectively. Residual dissimilarities were z-standardized after accounting for image similarity captured by DINOv3 and the mean dissimilarity associated with each pairing of the six broad categories. Blue and orange indicate image pairs that were closer and farther apart than expected from these controls. White lines separate categories. The hexbin plot below compares the corresponding residuals for all 2,556 image pairs (Spearman ρ = 0.299). "
        "**b,** Bidirectional participant-level correspondence from the MEG group average to 10 EEG participants and from the EEG group average to 16 MEG participants. Points denote participants, open circles show means and vertical lines show bootstrap 95% confidence intervals. Grey ranges indicate condition-matched noise ceilings. Lines connect prestimulus and poststimulus estimates from the same MEG participant. Double and triple asterisks indicate *P* < 0.01 and *P* < 0.001, respectively, using exact two-sided sign-flip tests. The bracket indicates the paired poststimulus-minus-prestimulus comparison. Category-preserving image-identity permutation results are shown in Supplementary Fig. 4."
    )
    (OUT / "Figure_4_caption_v009.md").write_text(caption, encoding="utf-8")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
