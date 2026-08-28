from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "generated_figures"
RESULTS = REPO / "results" / "reported" / "figure3.json"
DATA = REPO / "source_data" / "main" / "figure3.npz"
STYLE = REPO / "config" / "nature_style.mplstyle"

PURPLE = "#785EF0"
PURPLE_PALE = "#E7E1FF"
GREY = "#6E6E6E"
GREY_MID = "#A8A8A8"
GREY_PALE = "#E6E6E6"
BLACK = "#000000"

# Benjamini-Hochberg adjusted values across the three displayed poststimulus
# participant-level exact two-sided sign-flip tests. The prestimulus negative
# control used a directional positive-effect test and is not in this family.
CONDITION_Q = {
    "prestimulus": 0.942,
    "primary": 4.57763671875e-5,
    "category_controlled": 4.57763671875e-5,
    "within_category": 2.74658203125e-4,
}


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.115,
        1.095,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int | None = None,
    *,
    tight: bool = False,
) -> None:
    # Main figures retain the exact final-size canvas. Compact supplementary
    # panels may still be cropped to their artists.
    kwargs = {"facecolor": "white", "bbox_inches": fig.bbox_inches}
    if tight:
        kwargs.update({"bbox_inches": "tight", "pad_inches": 0.015})
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(path, **kwargs)


def main() -> None:
    plt.style.use(STYLE)
    mpl.rcParams.update({
        "font.size": 8.1,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.2,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
    })

    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    with np.load(DATA) as data:
        time_ms = np.asarray(data["time_ms"])
        timecourse = np.asarray(data["timecourse_participants"])
        time_ci = np.asarray(data["timecourse_ci"])
        condition_names = [str(x) for x in data["condition_names"]]
        condition_values = np.asarray(data["condition_values"])
        permutation_null = np.asarray(data["permutation_null"])

    fig = plt.figure(figsize=(183 / 25.4, 68 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.24, 1.05],
        left=0.072,
        right=0.992,
        bottom=0.190,
        top=0.875,
        wspace=0.36,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    # a. The fixed late EEG residual tested against the MEG RDM at each time.
    mean_time = timecourse.mean(axis=0)
    early_control = (time_ms >= 70) & (time_ms <= 130)
    display_mean = mean_time.copy()
    display_ci = time_ci.copy()
    display_mean[early_control] = np.nan
    display_ci[:, early_control] = np.nan

    ax_a.axvspan(70, 130, color=GREY_PALE, alpha=0.95, linewidth=0, zorder=0)
    ax_a.axvspan(180, 300, color=PURPLE_PALE, alpha=0.75, linewidth=0, zorder=0)
    ax_a.axhline(0, color=GREY_MID, linewidth=0.55, linestyle=(0, (2.2, 2.2)), zorder=0)
    ax_a.axvline(0, color=BLACK, linewidth=0.55, zorder=0)
    ax_a.fill_between(
        time_ms,
        display_ci[0],
        display_ci[1],
        color=PURPLE,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax_a.plot(time_ms, display_mean, color=PURPLE, linewidth=1.15, zorder=2)

    significant = [
        cluster for cluster in result["timecourse"]["clusters"]
        if cluster["p_exact_two_sided"] < 0.05
    ]
    cluster_y = 0.238
    for cluster in significant:
        ax_a.plot(
            [cluster["start_ms"], cluster["end_ms"]],
            [cluster_y, cluster_y],
            color=PURPLE,
            linewidth=2.1,
            solid_capstyle="butt",
            clip_on=False,
            zorder=3,
        )
        ax_a.text(
            (cluster["start_ms"] + cluster["end_ms"]) / 2,
            cluster_y + 0.010,
            stars(cluster["p_exact_two_sided"]),
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=BLACK,
        )

    legend_handles = [
        Patch(facecolor=GREY_PALE, edgecolor="none", label="Early-MEG control interval"),
        Patch(facecolor=PURPLE_PALE, edgecolor="none", label="Primary window"),
    ]
    ax_a.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=False,
        fontsize=7.0,
        handlelength=1.05,
        handleheight=0.72,
        borderaxespad=0.1,
        labelspacing=0.28,
    )
    ax_a.text(
        0.01,
        1.095,
        "Fixed late EEG residual tested across MEG time",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=BLACK,
    )
    ax_a.set_xlim(-100, 900)
    ax_a.set_ylim(-0.075, 0.275)
    ax_a.set_xticks([-100, 0, 200, 400, 600, 800])
    ax_a.set_yticks([0.0, 0.1, 0.2])
    ax_a.set_xlabel("MEG time from image onset (ms)")
    ax_a.set_ylabel(r"Partial Spearman $\rho$")
    panel_letter(ax_a, "a")

    # b. Participant-level estimates. Stars consistently denote BH-FDR-adjusted
    # values across the three poststimulus exact sign-flip tests; permutation P
    # values are reported in the caption rather than mixed into the same glyph.
    # Give the two sensitivity analyses a little more horizontal room so their
    # multi-line labels remain legible at the final two-column print width.
    x = np.array([0.0, 1.0, 2.18, 3.28])
    rng = np.random.default_rng(20260810)
    jitter = rng.uniform(-0.075, 0.075, size=condition_values.shape)

    # Condition-matched ceilings use the same controls and image-pair mask as
    # the corresponding effect. Narrow offset bars keep them visually separate
    # from the participant estimates and their significance markers.
    for idx, name in enumerate(condition_names):
        summary = result["conditions"][name]
        if "noise_ceiling_lower" in summary:
            lower = summary["noise_ceiling_lower"]
            upper = summary["noise_ceiling_upper"]
            ax_b.add_patch(Rectangle(
                (x[idx] + 0.14, lower),
                0.12,
                upper - lower,
                facecolor=GREY_PALE,
                edgecolor="none",
                alpha=0.75,
                zorder=0,
            ))

    for participant in range(condition_values.shape[1]):
        ax_b.plot(
            [x[0] + jitter[0, participant], x[1] + jitter[1, participant]],
            [condition_values[0, participant], condition_values[1, participant]],
            color=PURPLE,
            alpha=0.20,
            linewidth=0.45,
            zorder=1,
        )

    ax_b.scatter(
        (x[:, None] + jitter).ravel(),
        condition_values.ravel(),
        s=14,
        color=PURPLE,
        alpha=0.72,
        edgecolors="none",
        zorder=2,
    )

    for idx, name in enumerate(condition_names):
        summary = result["conditions"][name]
        lo, hi = summary["bootstrap_mean_95ci"]
        ax_b.plot([x[idx], x[idx]], [lo, hi], color=BLACK, linewidth=1.0, zorder=4)
        ax_b.scatter(
            [x[idx]],
            [summary["mean"]],
            s=34,
            facecolors="white",
            edgecolors=BLACK,
            linewidths=1.0,
            zorder=5,
        )
        mark = stars(CONDITION_Q[name])
        if mark:
            star_y = max(float(np.max(condition_values[idx])), hi) + 0.014
            ax_b.text(
                x[idx],
                star_y,
                mark,
                ha="center",
                va="bottom",
                fontsize=9.2,
                color=BLACK,
                zorder=6,
            )

    ax_b.axhline(0, color=GREY_MID, linewidth=0.55, linestyle=(0, (2.2, 2.2)), zorder=0)
    ax_b.set_xlim(-0.40, 3.68)
    ax_b.set_ylim(-0.125, 0.415)
    ax_b.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([
        "Prestimulus",
        "Primary",
        "Category separation\ncontrolled",
        "Within\ncategory",
    ])
    ax_b.set_ylabel(r"Partial Spearman $\rho$")
    ax_b.text(
        0.01,
        1.095,
        "Participant-level correspondence",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=BLACK,
    )
    panel_letter(ax_b, "b")

    # Main Figure 3: no null histogram. Preserve the distribution as a compact
    # supplementary panel so that the permutation remains inspectable.
    png = OUT / "Figure_3_EEG_to_MEG_temporal_transfer_v007.png"
    pdf = OUT / "Figure_3_EEG_to_MEG_temporal_transfer_v007.pdf"
    save_figure(fig, png, dpi=600)
    save_figure(fig, pdf)
    plt.close(fig)

    supp_fig, supp_ax = plt.subplots(figsize=(82 / 25.4, 56 / 25.4), constrained_layout=False)
    supp_fig.subplots_adjust(left=0.23, right=0.96, bottom=0.22, top=0.91)
    observed = result["alignment_permutation"]["observed_mean"]
    bins = np.linspace(float(permutation_null.min()), max(observed + 0.01, float(permutation_null.max())), 52)
    supp_ax.hist(permutation_null, bins=bins, color=GREY_PALE, edgecolor="white", linewidth=0.25)
    supp_ax.axvline(observed, color="#D55E00", linewidth=1.0)
    supp_ax.text(
        observed - 0.002,
        supp_ax.get_ylim()[1] * 0.84,
        "observed",
        ha="right",
        va="top",
        fontsize=5.5,
        color="#D55E00",
    )
    supp_ax.set_xlabel(r"Mean partial Spearman $\rho$")
    supp_ax.set_ylabel("Permutation count")
    supp_ax.text(
        0.03,
        0.98,
        r"$P_{\mathrm{perm}}=1.00\times10^{-4}$",
        transform=supp_ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=BLACK,
    )
    supp_png = OUT / "Supplementary_Figure_image_identity_permutation_v002.png"
    supp_pdf = OUT / "Supplementary_Figure_image_identity_permutation_v002.pdf"
    save_figure(supp_fig, supp_png, dpi=600, tight=True)
    save_figure(supp_fig, supp_pdf, tight=True)
    plt.close(supp_fig)

    print(json.dumps({
        "main_png": str(png),
        "main_pdf": str(pdf),
        "supplementary_png": str(supp_png),
        "supplementary_pdf": str(supp_pdf),
    }, indent=2))


if __name__ == "__main__":
    main()
