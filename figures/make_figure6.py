from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "generated_figures"
STAGE2 = json.loads((REPO / "results" / "reported" / "external_transfer.json").read_text(encoding="utf-8"))
NOD = json.loads((REPO / "results" / "reported" / "nod_multibackbone.json").read_text(encoding="utf-8"))

EEG = "#009E73"
DINO = "#3B7FB2"
CLIP = "#D55E00"
SIGLIP = "#C45A91"
FAMILY = "#8E5D7D"
INK = "#202124"
MID = "#73777C"
LIGHT = "#C8CDD1"
PALE = "#F3F4F5"
ZERO = "#969BA0"
THRESHOLD = "#A94D55"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.7,
        "axes.labelsize": 8.1,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def clean_axis(ax: plt.Axes, *, hide_left: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)


def heading(
    fig: plt.Figure,
    label_x: float,
    title_x: float,
    y: float,
    label: str,
    title: str,
) -> None:
    fig.text(
        label_x,
        y,
        label,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(title_x, y, title, ha="left", va="top", fontsize=9.2, color=INK)


def sorted_jitter(values: np.ndarray, amplitude: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    offsets = np.linspace(-amplitude, amplitude, len(values))
    out = np.empty_like(offsets)
    out[np.argsort(values)] = offsets
    return out


def mean_ci(summary: dict) -> tuple[float, float, float]:
    mean = float(summary["mean"])
    lo, hi = (float(x) for x in summary["bootstrap_mean_95ci"])
    return mean, lo, hi


def main() -> None:
    # Exact Nature double-column width. Axes are positioned manually so the
    # exported canvas remains 183 x 108 mm without bbox-dependent resizing.
    width_mm, height_mm = 183.0, 108.0
    fig = plt.figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        facecolor="white",
    )

    ax_a = fig.add_axes([0.105, 0.585, 0.405, 0.295])
    ax_b = fig.add_axes([0.625, 0.585, 0.325, 0.295])
    ax_c = fig.add_axes([0.135, 0.090, 0.425, 0.255])
    ax_d = fig.add_axes([0.670, 0.090, 0.280, 0.255])

    # Descriptive panel titles are used instead of conclusion sentences.
    heading(fig, 0.014, 0.052, 0.982, "a", "Independent EEG alignment for unseen concepts")
    heading(fig, 0.555, 0.593, 0.982, "b", "Human-similarity alignment")
    heading(fig, 0.014, 0.052, 0.445, "c", "ImageNet transfer across backbones")
    heading(fig, 0.605, 0.643, 0.445, "d", "Poststimulus specificity")

    # a | Participant-level transfer to independent EEG recordings.
    clean_axis(ax_a)
    ax_a.axvline(0, color=ZERO, lw=0.7, ls=(0, (2.6, 2.6)), zorder=0)
    ax_a.set_xlim(-0.0022, 0.0216)
    ax_a.set_ylim(-0.48, 1.48)
    ax_a.set_xticks([0.000, 0.005, 0.010, 0.015, 0.020])
    ax_a.set_yticks([1, 0], ["THINGS-EEG2\n$n=8$", "Alljoined\n$n=20$"])
    ax_a.tick_params(axis="y", length=0, pad=5)
    ax_a.set_xlabel(r"Independent EEG alignment gain, $\Delta\rho$")

    for y, summary, stars in [
        (1.0, STAGE2["things_eeg"], "**"),
        (0.0, STAGE2["alljoined_eeg"], "***"),
    ]:
        values = np.asarray(summary["values"], dtype=float)
        ax_a.scatter(
            values,
            y + sorted_jitter(values, 0.135),
            s=17,
            marker="o",
            facecolor=EEG,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.70,
            zorder=2,
        )
        mean, lo, hi = mean_ci(summary)
        ax_a.plot([lo, hi], [y, y], color=INK, lw=1.25, zorder=4)
        ax_a.scatter(
            [mean],
            [y],
            marker="D",
            s=38,
            facecolor=EEG,
            edgecolor=INK,
            linewidth=0.65,
            zorder=5,
        )
        ax_a.text(
            mean,
            y + 0.255,
            stars,
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=INK,
        )

    # b | Behavioural gain relative to the concept-label permutation null.
    clean_axis(ax_b, hide_left=True)
    behaviour = STAGE2["human_similarity"]
    gain = float(behaviour["gain"])
    null_mean = float(behaviour["label_permutation"]["null_mean"])
    null_95th = float(behaviour["label_permutation"]["null_95th"])
    ax_b.set_xlim(-0.0022, 0.0114)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_yticks([])
    ax_b.set_xticks([0.000, 0.005, 0.010])
    ax_b.set_xlabel(r"Gain in SPoSE alignment, $\Delta\rho$")
    ax_b.axvline(0, color=ZERO, lw=0.7, ls=(0, (2.6, 2.6)), zorder=0)
    ax_b.plot(
        [null_mean, null_95th],
        [0.49, 0.49],
        color=LIGHT,
        lw=4.2,
        solid_capstyle="round",
        zorder=1,
    )
    ax_b.scatter(
        [null_mean],
        [0.49],
        s=30,
        marker="o",
        facecolor="white",
        edgecolor=MID,
        linewidth=0.8,
        zorder=3,
    )
    ax_b.plot([null_95th, null_95th], [0.39, 0.59], color=MID, lw=1.0, zorder=3)
    ax_b.scatter(
        [gain],
        [0.49],
        s=58,
        marker="D",
        facecolor=DINO,
        edgecolor=INK,
        linewidth=0.65,
        zorder=4,
    )
    ax_b.text(
        (null_mean + null_95th) / 2,
        0.34,
        "permutation null",
        ha="center",
        va="top",
        fontsize=7.0,
        color=MID,
    )
    ax_b.text(
        null_95th,
        0.64,
        "95th percentile",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=MID,
    )
    ax_b.text(gain, 0.34, "observed", ha="center", va="top", fontsize=7.1, color=INK)
    ax_b.text(
        gain,
        0.66,
        "**",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )

    # c | Mean and bootstrap interval for each fixed backbone adapter.
    clean_axis(ax_c)
    ax_c.axvline(0.005, color=THRESHOLD, lw=0.9, ls=(0, (4.2, 2.2)), zorder=0)
    ax_c.axhspan(-0.45, 0.45, color=PALE, zorder=-2)
    ax_c.set_xlim(0.0, 0.0122)
    ax_c.set_ylim(-0.58, 3.52)
    ax_c.set_xticks([0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012])
    ax_c.set_yticks([3, 2, 1, 0], ["DINOv3", "CLIP", "SigLIP", "CLIP–SigLIP mean"])
    ax_c.tick_params(axis="y", length=0, pad=5)
    ax_c.set_xlabel(r"Poststimulus NOD-EEG gain, $\Delta\rho$")
    ax_c.text(
        0.00515,
        3.43,
        "prespecified 0.005",
        color=THRESHOLD,
        fontsize=6.9,
        ha="left",
        va="top",
    )

    rows = [
        (3.0, STAGE2["nod"]["poststimulus"], DINO, ""),
        (2.0, NOD["individual_backbones"]["CLIP-B32"]["poststimulus"], CLIP, ""),
        (1.0, NOD["individual_backbones"]["SigLIP-base"]["poststimulus"], SIGLIP, ""),
        (0.0, NOD["non_dino_family"]["poststimulus"], FAMILY, "**"),
    ]
    for y, summary, colour, stars in rows:
        mean, lo, hi = mean_ci(summary)
        ax_c.plot([lo, hi], [y, y], color=INK, lw=1.25, zorder=3)
        ax_c.scatter(
            [mean],
            [y],
            marker="D",
            s=39,
            facecolor=colour,
            edgecolor=INK,
            linewidth=0.65,
            zorder=4,
        )
        if stars:
            ax_c.text(
                mean,
                y + 0.26,
                stars,
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
                color=INK,
            )

    # d | Participant-level poststimulus-minus-prestimulus differences.
    clean_axis(ax_d)
    family = NOD["non_dino_family"]
    difference = np.asarray(family["post_minus_prestimulus"]["values"], dtype=float)
    difference_summary = family["post_minus_prestimulus"]
    ax_d.axvline(0, color=ZERO, lw=0.7, ls=(0, (2.6, 2.6)), zorder=0)
    ax_d.set_xlim(-0.0105, 0.0222)
    ax_d.set_ylim(-0.52, 0.52)
    ax_d.set_xticks([-0.01, 0.00, 0.01, 0.02])
    ax_d.set_yticks([])
    ax_d.set_xlabel(r"Poststimulus $-$ prestimulus gain, $\Delta\rho$")
    ax_d.scatter(
        difference,
        sorted_jitter(difference, 0.16),
        s=18,
        marker="o",
        facecolor=FAMILY,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.72,
        zorder=2,
    )
    diff_mean, diff_lo, diff_hi = mean_ci(difference_summary)
    ax_d.plot([diff_lo, diff_hi], [0, 0], color=INK, lw=1.3, zorder=4)
    ax_d.scatter(
        [diff_mean],
        [0],
        s=45,
        marker="D",
        facecolor=FAMILY,
        edgecolor=INK,
        linewidth=0.65,
        zorder=5,
    )
    ax_d.text(
        diff_mean,
        0.255,
        "**",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )

    stem = "Figure_6_external_generalization_nature_v006"
    png = HERE / f"{stem}.png"
    pdf = HERE / f"{stem}.pdf"
    svg = HERE / f"{stem}.svg"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    caption = (
        "Fig. 6 | Neural adaptation generalizes beyond the source images and recordings. "
        "a, Participant-level DINOv3 alignment gains for 183 concepts absent from source fitting, evaluated in THINGS-EEG2 (n = 8) and the independent Alljoined acquisition (n = 20). Points denote participants; diamonds and horizontal lines show means and bootstrap 95% confidence intervals. Gains were positive in all 8 THINGS-EEG2 participants and 17 of 20 Alljoined participants. Double and triple asterisks indicate exact two-sided sign-flip P < 0.01 and P < 0.001, respectively. "
        "b, Gain in agreement with the Sparse Positive Similarity Embedding (SPoSE) relative to a concept-label permutation null; human judgements were not used during adapter training. The grey circle and vertical line mark the null mean and 95th percentile, respectively, and the blue diamond marks the observed gain. Alignment increased from 0.4243 to 0.4342 (gain, 0.00986; 9,999-draw one-sided permutation P = 0.0012; double asterisks). "
        "c, Mean poststimulus NOD-EEG gains after applying the fixed adapters without refitting. Diamonds and horizontal lines show means and bootstrap 95% confidence intervals. The red dashed line marks the prespecified minimum gain of 0.005. Mean gains were 0.00461 for DINOv3, 0.00579 for CLIP and 0.00572 for SigLIP. The prespecified within-participant CLIP–SigLIP mean was 0.00576 (exact two-sided sign-flip P = 0.00335; double asterisks). "
        "d, Participant-level poststimulus-minus-prestimulus differences for the CLIP–SigLIP mean. Points denote participants; the diamond and horizontal line show the mean and bootstrap 95% confidence interval. The difference was positive in 15 of 19 participants (mean, 0.00579; exact two-sided sign-flip P = 0.00338; double asterisks), whereas the prestimulus gain was not positive (one-sided P = 0.634)."
    )
    caption_path = HERE / "Figure_6_caption_nature_v006.txt"
    caption_path.write_text(caption, encoding="utf-8")

    manifest = {
        "figure": "Figure 6",
        "version": "v006",
        "canvas_mm": [width_mm, height_mm],
        "design": {
            "titles": "descriptive sentence-case titles, regular 9.2 pt",
            "statistics": "participant points, bootstrap 95% CIs and inferential stars",
            "a": "independent EEG transfer for unseen concepts",
            "b": "behavioural gain relative to the label-permutation null",
            "c": "fixed-adapter ImageNet transfer across backbones",
            "d": "participant-level poststimulus specificity",
        },
        "outputs": {
            "png": str(png),
            "pdf": str(pdf),
            "svg": str(svg),
            "caption": str(caption_path),
        },
    }
    (HERE / "Figure_6_manifest_v006.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
