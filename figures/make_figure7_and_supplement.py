from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "reported"
SOURCE = REPO / "source_data" / "supplementary"
CORE_JSON = RESULTS / "core_rsa_robustness.json"
CORE_NPZ = SOURCE / "core_rsa_robustness.npz"
LOIO_JSON = RESULTS / "leave_one_image_out.json"
BASE_JSON = RESULTS / "adapter_baselines.json"
STAB_JSON = RESULTS / "adapter_stability.json"
SEC_JSON = RESULTS / "secondary_controls.json"
PERM_JSON = RESULTS / "target_specificity_permutations.json"
PERM_NPZ = REPO / "source_data" / "main" / "target_specificity_permutations.npz"
NOD_JSON = RESULTS / "paired_nod.json"

OUT = REPO / "generated_figures"
OUT.mkdir(parents=True, exist_ok=True)

EEG = "#009E73"
MEG = "#785EF0"
ACCENT = "#D55E00"
GREY = "#777777"
LIGHT = "#D9D9D9"

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def panel(ax, letter, title):
    ax.text(-0.12, 1.10, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    ax.set_title(title, loc="left", fontsize=8.0, fontweight="normal", pad=8)


def strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


core = load_json(CORE_JSON)
arr = np.load(CORE_NPZ, allow_pickle=True)
loio = load_json(LOIO_JSON)
base = load_json(BASE_JSON)
stab = load_json(STAB_JSON)
sec = load_json(SEC_JSON)
perm = load_json(PERM_JSON)
nod = load_json(NOD_JSON)


# Supplementary robustness figure
fig = plt.figure(figsize=(7.20, 3.55))
ax = fig.add_axes([0.075, 0.17, 0.30, 0.69])
mat = np.asarray(arr["time_by_time_mean"], float)
tmap = np.asarray(arr["time_by_time_t"], float)
eeg_t = np.asarray(arr["eeg_times_ms"], float)
meg_t = np.asarray(arr["meg_times_ms"], float)
thr = float(core["time_by_time_rsa"]["cluster_forming_t"])
labels, nlab = ndimage.label(tmap > thr)
masses = [(tmap[labels == i].sum(), i) for i in range(1, nlab + 1)]
main_lab = max(masses)[1] if masses else 0
mask = labels == main_lab
im = ax.imshow(mat, origin="lower", aspect="auto", extent=[meg_t[0], meg_t[-1], eeg_t[0], eeg_t[-1]], cmap="Purples", vmin=0, vmax=max(0.20, np.nanpercentile(mat, 99)))
if main_lab:
    ax.contour(meg_t, eeg_t, mask.astype(float), levels=[0.5], colors="black", linewidths=0.8)
ax.axvline(0, color=GREY, lw=0.7)
ax.axhline(0, color=GREY, lw=0.7)
ax.add_patch(plt.Rectangle((180, 192), 120, 128, fill=False, ec="white", lw=1.0, ls=(0, (3, 2))))
ax.set(xlabel="MEG time (ms)", ylabel="EEG time (ms)")
panel(ax, "a", "Full time × time correspondence")
cax = fig.add_axes([0.388, 0.535, 0.012, 0.245])
cb = fig.colorbar(im, cax=cax)
cb.set_ticks([0, 0.2]); cb.ax.set_title("$\\rho$", fontsize=6, pad=1)
cb.ax.tick_params(labelsize=6, length=2)

ax = fig.add_axes([0.485, 0.17, 0.265, 0.69])
sens = [
    ("Primary", core["primary_late_geometry"]["eeg_group_to_meg"]["values"]),
    ("All\ncontrols", core["broader_control_sensitivity"]["category_plus_all_visual_and_caption"]["eeg_group_to_meg"]["values"]),
    ("Cross-\nfitted", core["crossfitted_early_sensitivity"]["heldout_group_early"]["meg_participant_scores"]["values"]),
    ("Decoding\nRDM", core["decoding_estimator_sensitivity"]["late_after_early"]["values"]),
]
rng = np.random.default_rng(7)
for i, (lab, vals) in enumerate(sens):
    vals = np.asarray(vals, float)
    ax.scatter(i + rng.normal(0, 0.045, len(vals)), vals, s=13, color=MEG, alpha=0.72, lw=0)
    m = vals.mean()
    ci = np.quantile([np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(5000)], [0.025, 0.975])
    ax.errorbar(i, m, yerr=[[m-ci[0]], [ci[1]-m]], fmt="o", ms=5.5, mfc="white", mec="black", mew=1, ecolor="black", lw=1, capsize=0, zorder=4)
ax.axhline(0, color=GREY, lw=0.7, ls=(0, (3, 3)))
ax.set_xticks(range(len(sens)), [x[0] for x in sens])
ax.set_ylabel("EEG→MEG partial $\\rho$")
panel(ax, "b", "Estimator and control sensitivity")
strip(ax)

ax = fig.add_axes([0.825, 0.17, 0.155, 0.69])
full = core["primary_late_geometry"]["direct_group_rho"]
loco = np.asarray([x["direct_group_rho"] for x in core["leave_one_category_out"]], float)
loio_vals = np.asarray([x["direct_group_rho"] for x in loio["rows"]], float)
rng = np.random.default_rng(11)
ax.scatter(rng.normal(0, .035, len(loco)), loco, color="#4C78A8", s=20, alpha=.85)
ax.scatter(1 + rng.normal(0, .035, len(loio_vals)), loio_vals, color="#72B7B2", s=10, alpha=.60)
ax.axhline(full, color=ACCENT, lw=1.2)
all_c = np.concatenate([loco, loio_vals, np.asarray([full])])
span = max(0.01, float(np.ptp(all_c)))
ax.set_ylim(float(np.min(all_c)) - 0.10 * span, float(np.max(all_c)) + 0.12 * span)
ax.set_xticks([0, 1], ["Leave one\ncategory out", "Leave one\nimage out"])
ax.set_ylabel("Group EEG–MEG partial $\\rho$")
ax.text(0.98, full, "all images", transform=ax.get_yaxis_transform(),
        color=ACCENT, fontsize=6.4, ha="right", va="bottom")
panel(ax, "c", "Leave-one-out influence")
strip(ax)
save(fig, "Supplementary_Fig10_core_robustness_v001")


# Supplementary adapter-specificity figure
fig = plt.figure(figsize=(7.20, 4.40))
gs = fig.add_gridspec(2, 2, left=0.085, right=0.97, bottom=0.11, top=0.80,
                      hspace=0.68, wspace=0.40)
targets = ["raw_consensus", "controlled_residual", "category_only", "dino_self"]
labels = ["Late EEG–MEG", "Controlled residual", "Category only", "Frozen-model self"]
x = np.arange(len(targets))

def target_panel(ax, metric, ylabel, letter, title):
    for off, meas, color in [(-0.12, "eeg", EEG), (0.12, "meg", MEG)]:
        means, lo, hi = [], [], []
        for t in targets:
            key = f"heldout_{meas}_{metric}"
            d = base["target_baselines"][t][key]
            means.append(d["mean"])
            lo.append(d["bootstrap_mean_95ci"][0]); hi.append(d["bootstrap_mean_95ci"][1])
        means = np.asarray(means); lo=np.asarray(lo); hi=np.asarray(hi)
        ax.errorbar(x+off, means, yerr=[means-lo, hi-means], fmt="o", color=color, mfc="white", mec=color, mew=1.2, lw=1, ms=5, label=meas.upper())
    ax.axhline(0, color=GREY, lw=.7, ls=(0,(3,3)))
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    values = []
    for t in targets:
        for meas in ("eeg", "meg"):
            d = base["target_baselines"][t][f"heldout_{meas}_{metric}"]
            values.extend([d["bootstrap_mean_95ci"][0], d["bootstrap_mean_95ci"][1]])
    lo_lim = min(0.0, float(np.min(values)))
    hi_lim = max(0.0, float(np.max(values)))
    pad = max(0.002, 0.10 * (hi_lim - lo_lim))
    ax.set_ylim(lo_lim - pad, hi_lim + pad)
    panel(ax, letter, title)
    strip(ax)

target_panel(fig.add_subplot(gs[0,0]), "gain", "Held-out alignment gain, $\\Delta\\rho$", "a", "Matched target baselines")
target_panel(fig.add_subplot(gs[0,1]), "unique_movement", "Unique neural movement, $r$", "b", "Image-specific adapter movement")
handles, legend_labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles[:2], legend_labels[:2], frameon=False, ncol=2,
           loc="upper center", bbox_to_anchor=(0.51, 0.965),
           fontsize=7.0, handletextpad=0.4, columnspacing=1.1)

ax = fig.add_subplot(gs[1,0])
perm_npz = np.load(PERM_NPZ)
null = np.asarray(perm_npz["ensemble_null"], float)
obs = float(perm_npz["observed_ensemble"])
ax.hist(null, bins=45, color=LIGHT, edgecolor="none")
ax.axvline(obs, color=ACCENT, lw=1.5)
ax.text(obs, ax.get_ylim()[1]*.88, f" observed {obs:.4f}\n $P$ = 0.0219", color=ACCENT, ha="left", va="top", fontsize=7)
ax.set_xlabel("Gain after within-category target shuffling")
ax.set_ylabel("Permutation count")
panel(ax, "c", "Category-preserving target permutation")
strip(ax)

ax = fig.add_subplot(gs[1,1])
cells = base["hyperparameter_grid"]["cells"]
bvals = [32,64,128]; lvals=[10,100,1000]
hm=np.full((3,3),np.nan)
for cell in cells:
    i=bvals.index(int(cell["bottleneck"])); j=lvals.index(int(cell["lambda_anchor"]))
    hm[i,j]=float(cell["macro_gain"])
im=ax.imshow(hm, cmap="Blues", aspect="auto", vmin=0)
for i in range(3):
    for j in range(3):
        ax.text(j,i,f"{hm[i,j]:.3f}",ha="center",va="center",fontsize=6.5,color="white" if hm[i,j]>.018 else "black")
ax.set_xticks(range(3),lvals); ax.set_yticks(range(3),bvals)
ax.set_xlabel("Anchor weight"); ax.set_ylabel("Adapter bottleneck")
panel(ax, "d", "Fixed hyperparameter-grid sensitivity")
save(fig, "Supplementary_Fig11_adapter_specificity_v001")


# Main paired NOD boundary figure
fig = plt.figure(figsize=(7.20, 2.65))
gs = fig.add_gridspec(1, 2, wspace=0.42)

ax = fig.add_subplot(gs[0,0])
eeg_vals = np.asarray(nod["endpoints"]["native_eeg_delta_post"]["values"], float)
meg_vals = np.asarray(nod["endpoints"]["native_meg_delta_post"]["values"], float)
for i in range(min(len(eeg_vals),len(meg_vals))):
    ax.plot([0,1],[eeg_vals[i],meg_vals[i]],color=LIGHT,lw=.6,zorder=1)
rng=np.random.default_rng(4)
for i,(vals,color) in enumerate([(eeg_vals,EEG),(meg_vals,MEG)]):
    ax.scatter(i+rng.normal(0,.035,len(vals)),vals,s=15,color=color,alpha=.75,lw=0,zorder=2)
    m=vals.mean(); boot=np.asarray([np.mean(rng.choice(vals,len(vals),replace=True)) for _ in range(10000)]); ci=np.quantile(boot,[.025,.975])
    ax.errorbar(i,m,yerr=[[m-ci[0]],[ci[1]-m]],fmt="o",ms=6,mfc="white",mec="black",mew=1.1,ecolor="black",lw=1,zorder=4)
ax.axhline(0,color=GREY,lw=.7,ls=(0,(3,3)))
ax.set_xticks([0,1],["EEG","MEG"]); ax.set_ylabel("Frozen-to-adapted gain, $\\Delta\\rho$")
panel(ax,"a","Paired transfer to NOD recordings")
strip(ax)

ax=fig.add_subplot(gs[0,1])
rel=sec["nod_group_reliability"]
for i,(key,color) in enumerate([("eeg_native_post",EEG),("meg_native_post",MEG)]):
    d=rel[key]; vals=np.asarray(d["leave_one_participant_to_group_values"],float)
    ax.scatter(i+rng.normal(0,.035,len(vals)),vals,s=15,color=color,alpha=.75,lw=0)
    m=vals.mean(); ci=np.asarray(d["leave_one_participant_to_group_95ci"],float)
    ax.errorbar(i,m,yerr=[[m-ci[0]],[ci[1]-m]],fmt="o",ms=6,mfc="white",mec="black",mew=1.1,ecolor="black",lw=1)
    ax.vlines(i, d["noise_ceiling_lower_mean"], d["noise_ceiling_upper_mean"], color=LIGHT, lw=8, zorder=0)
ax.axhline(0,color=GREY,lw=.7,ls=(0,(3,3)))
ax.set_xticks([0,1],["EEG","MEG"]); ax.set_ylabel("Participant-to-group reliability, $\\rho$")
panel(ax,"b","Reliability of the external neural geometry")
strip(ax)
save(fig, "Fig7_paired_NOD_boundary_v001")

print(OUT)
