from __future__ import annotations

from pathlib import Path
import hashlib
import itertools
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
from scipy.io import loadmat
from scipy.stats import rankdata, spearmanr
import h5py

OUT = Path(os.environ.get("EEG_MEG_OUTPUT_DIR", ROOT / "derived" / "temporal_stage0"))
EEG_DIR = Path(os.environ.get("KANESHIRO_EEG_DIR", ROOT / "data" / "kaneshiro_eeg"))
MEG_FILE = Path(os.environ.get("CICHY_MEG_FILE", ROOT / "data" / "cichy_meg_rdms.mat"))
MAPPING = ROOT / "source_data" / "supplementary" / "stimulus_mapping.csv"
DINO_FEATURES = ROOT / "source_data" / "model_features" / "dinov3_72.npy"
RNG = np.random.default_rng(20260722)
N_PERM = 10_000
N_BOOT = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def upper_vector(x: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(x.shape[0], 1)
    return np.asarray(x[i, j], dtype=np.float64)


def cross_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = x.shape[1]
    g = (x @ y.T) / p
    d = np.diag(g)[:, None] + np.diag(g)[None, :] - g - g.T
    np.fill_diagonal(d, 0.0)
    return d


def residualize_rank(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    yr = rankdata(y).astype(np.float64)
    cols = [np.ones(len(yr), dtype=np.float64)]
    for c in controls:
        cols.append(rankdata(c).astype(np.float64))
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, yr, rcond=None)[0]
    return yr - X @ beta


def rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def exact_signflip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    n = len(values)
    if n <= 20:
        means = []
        for bits in itertools.product((-1.0, 1.0), repeat=n):
            means.append(float(np.mean(values * np.asarray(bits))))
        means = np.asarray(means)
        return float(np.mean(np.abs(means) >= observed - 1e-15))
    signs = RNG.choice((-1.0, 1.0), size=(100_000, n))
    null = np.mean(signs * values[None, :], axis=1)
    return float((1 + np.sum(np.abs(null) >= observed)) / (len(null) + 1))


def bootstrap_ci(values: np.ndarray) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
    return [float(x) for x in np.quantile(v[idx].mean(axis=1), [0.025, 0.975])]


def summarize(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "n_positive": int(np.sum(v > 0)),
        "n_negative": int(np.sum(v < 0)),
        "n_total": int(len(v)),
        "exact_signflip_p_two_sided": exact_signflip(v),
        "bootstrap_mean_95ci": bootstrap_ci(v),
        "values": v.tolist(),
    }


def fold_patterns(X: np.ndarray, labels: np.ndarray, sample_idx: np.ndarray) -> np.ndarray:
    # X is channels x time x trials. Occurrence-based interleaving preserves all
    # exemplars in every fold and separates repeated measurements.
    labels = labels.astype(int)
    n_img = 72
    fold_id = np.empty(len(labels), dtype=int)
    counts = np.zeros(n_img + 1, dtype=int)
    for t, lab in enumerate(labels):
        fold_id[t] = counts[lab] % 4
        counts[lab] += 1
    if counts[1:].min() < 68:
        raise RuntimeError(f"Insufficient repetitions: {counts[1:].min()}")

    raw = np.transpose(X[:, sample_idx, :], (2, 0, 1)).reshape(len(labels), -1).astype(np.float64)
    out = np.empty((4, n_img, raw.shape[1]), dtype=np.float64)
    for f in range(4):
        train = fold_id != f
        mu = raw[train].mean(axis=0)
        sd = raw[train].std(axis=0, ddof=1)
        sd[sd < 1e-8] = 1.0
        z = (raw - mu) / sd
        for lab in range(1, n_img + 1):
            take = (fold_id == f) & (labels == lab)
            if take.sum() < 17:
                raise RuntimeError(f"Fold count too small f={f} label={lab}: {take.sum()}")
            out[f, lab - 1] = z[take].mean(axis=0)
    return out


def load_eeg_metrics() -> tuple[dict, np.ndarray]:
    early_idx = np.arange(4, 10)   # 64,80,...,144 ms
    late_idx = np.arange(12, 21)   # 192,208,...,320 ms
    participant = []
    correction_a, correction_b = [], []

    category = np.repeat(np.arange(6), 12)
    category_rdm = upper_vector((category[:, None] != category[None, :]).astype(float))

    for s in range(1, 11):
        d = loadmat(
            EEG_DIR / f"S{s}.mat",
            variable_names=["X_3D", "exemplarLabels", "categoryLabels", "Fs", "N"],
            squeeze_me=True,
        )
        X = np.asarray(d["X_3D"])
        labels = np.asarray(d["exemplarLabels"]).ravel().astype(int)
        cats = np.asarray(d["categoryLabels"]).ravel().astype(int)
        if X.shape[:2] != (124, 32) or X.shape[2] != len(labels):
            raise RuntimeError(f"S{s} structural mismatch {X.shape}")
        for lab in range(1, 73):
            observed = np.unique(cats[labels == lab])
            if len(observed) != 1 or observed[0] != (lab - 1) // 12 + 1:
                raise RuntimeError(f"S{s} label/category mismatch {lab}: {observed}")

        pe = fold_patterns(X, labels, early_idx)
        pl = fold_patterns(X, labels, late_idx)
        ea = upper_vector(cross_distance(pe[0], pe[1]))
        eb = upper_vector(cross_distance(pe[2], pe[3]))
        la = upper_vector(cross_distance(pl[0], pl[1]))
        lb = upper_vector(cross_distance(pl[2], pl[3]))
        ca = residualize_rank(la, [ea])
        cb = residualize_rank(lb, [eb])
        correction_a.append(ca); correction_b.append(cb)

        cat_a = rho(residualize_rank(la, [ea]), residualize_rank(category_rdm, [ea]))
        cat_b = rho(residualize_rank(lb, [eb]), residualize_rank(category_rdm, [eb]))
        participant.append({
            "participant": s,
            "early_reliability": rho(ea, eb),
            "late_reliability": rho(la, lb),
            "correction_reliability": rho(ca, cb),
            "category_partial": float(np.mean([cat_a, cat_b])),
        })
        del X, pe, pl, d

    A = np.stack(correction_a); B = np.stack(correction_b)
    shared = []
    for i in range(10):
        others = np.arange(10) != i
        shared.append(float(np.mean([
            rho(A[i], B[others].mean(axis=0)),
            rho(B[i], A[others].mean(axis=0)),
        ])))
        participant[i]["shared_correction_alignment"] = shared[-1]

    gates = {
        "early_reliability": summarize([x["early_reliability"] for x in participant]),
        "late_reliability": summarize([x["late_reliability"] for x in participant]),
        "correction_reliability": summarize([x["correction_reliability"] for x in participant]),
        "shared_correction": summarize(shared),
        "category_positive_control": summarize([x["category_partial"] for x in participant]),
    }
    gates["early_reliability"]["pass"] = gates["early_reliability"]["mean"] > 0 and gates["early_reliability"]["n_positive"] >= 8
    gates["late_reliability"]["pass"] = gates["late_reliability"]["mean"] > 0 and gates["late_reliability"]["n_positive"] >= 8
    gates["correction_reliability"]["pass"] = (gates["correction_reliability"]["mean"] > .05 and gates["correction_reliability"]["n_positive"] >= 8 and gates["correction_reliability"]["exact_signflip_p_two_sided"] < .05)
    gates["shared_correction"]["pass"] = (gates["shared_correction"]["mean"] > .03 and gates["shared_correction"]["n_positive"] >= 8 and gates["shared_correction"]["exact_signflip_p_two_sided"] < .05)
    gates["category_positive_control"]["pass"] = (gates["category_positive_control"]["n_positive"] >= 8 and gates["category_positive_control"]["exact_signflip_p_two_sided"] < .05)

    # Average standardized independent correction estimates only after all
    # participant metrics are fixed.
    C = (A + B) / 2
    C = (C - C.mean(axis=1, keepdims=True)) / C.std(axis=1, keepdims=True)
    group_correction = C.mean(axis=0)
    return {"participants": participant, "gates": gates}, group_correction


def meg_confirmation(group_correction: np.ndarray) -> dict:
    mapping = np.genfromtxt(MAPPING, delimiter=",", names=True, dtype=None, encoding="utf-8")
    idx = np.asarray(mapping["cichy_index"], dtype=int) - 1
    if len(idx) != 72 or len(np.unique(idx)) != 72:
        raise RuntimeError("Mapping invalid")
    z = np.load(DINO_FEATURES)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    dino = upper_vector(1 - z @ z.T)

    varname = "MEG_decoding_RDMs"
    # MATLAB v7.3 stores reversed dimension order in HDF5:
    # image x image x time x session x participant.
    windows = {"early": (70, 130), "late": (180, 300), "pre": (-100, -1)}
    vectors = {name: [] for name in windows}
    with h5py.File(MEG_FILE, "r") as f:
        if list(f.keys()) != [varname] or f[varname].shape != (92, 92, 1301, 2, 16):
            raise RuntimeError(f"MEG HDF5 structure mismatch: {[(k, f[k].shape) for k in f.keys()]}")
        ds = f[varname]
        for p in range(16):
            for name, (lo_ms, hi_ms) in windows.items():
                lo, hi = lo_ms + 100, hi_ms + 100
                block = np.asarray(ds[:, :, lo:hi + 1, :, p])
                m = np.nanmean(block, axis=(2, 3))
                m = m[np.ix_(idx, idx)]
                vectors[name].append(upper_vector(m))

    primary, pre, post_minus_pre = [], [], []
    for p in range(16):
        e = vectors["early"][p]
        l = vectors["late"][p]
        pr = vectors["pre"][p]
        pred_r = residualize_rank(group_correction, [e, dino])
        late_r = residualize_rank(l, [e, dino])
        pre_r = residualize_rank(pr, [e, dino])
        primary.append(rho(pred_r, late_r))
        pre.append(rho(pred_r, pre_r))
        post_minus_pre.append(primary[-1] - pre[-1])

    primary_s = summarize(primary)
    pre_s = summarize(pre)
    diff_s = summarize(post_minus_pre)

    # Image-label permutation of the frozen EEG correction.
    n = 72
    iu = np.triu_indices(n, 1)
    C = np.zeros((n, n), dtype=float)
    C[iu] = group_correction; C[(iu[1], iu[0])] = group_correction
    # Cache participant residualized targets and their controls.
    targets, ctrls = [], []
    for p in range(16):
        e = vectors["early"][p]
        l = vectors["late"][p]
        targets.append(residualize_rank(l, [e, dino]))
        ctrls.append(e)
    observed = float(np.mean(primary))
    null = np.empty(N_PERM, dtype=float)
    for q in range(N_PERM):
        perm = RNG.permutation(n)
        pv = upper_vector(C[np.ix_(perm, perm)])
        vals = []
        for p in range(16):
            prd = residualize_rank(pv, [ctrls[p], dino])
            vals.append(rho(prd, targets[p]))
        null[q] = np.mean(vals)
    perm_p = float((1 + np.sum(null >= observed)) / (N_PERM + 1))

    # Pre is a falsification: a positive one-sided test must be non-significant.
    pre_v = np.asarray(pre)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=16)))
    pre_null = np.mean(signs * pre_v[None, :], axis=1)
    pre_one_sided_p = float(np.mean(pre_null >= pre_v.mean() - 1e-15))

    primary_s["alignment_permutation_p_one_sided"] = perm_p
    primary_s["pass"] = (primary_s["mean"] > .02 and primary_s["n_positive"] >= 12 and primary_s["exact_signflip_p_two_sided"] < .05 and perm_p < .05)
    pre_s["one_sided_signflip_p_positive"] = pre_one_sided_p
    pre_s["pass"] = pre_one_sided_p >= .05
    diff_s["pass"] = diff_s["n_positive"] >= 12 and diff_s["exact_signflip_p_two_sided"] < .05
    return {
        "meg_variable": varname,
        "meg_file_sha256": sha256(MEG_FILE),
        "primary": primary_s,
        "prestimulus_falsification": pre_s,
        "post_minus_pre": diff_s,
        "joint_pass": bool(primary_s["pass"] and pre_s["pass"] and diff_s["pass"]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "analysis": "cross-modal early-to-late relational correction Stage 0",
        "script_sha256": sha256(Path(__file__)),
        "mapping_sha256": sha256(MAPPING),
        "dino_feature_sha256": sha256(DINO_FEATURES),
    }
    eeg, group_correction = load_eeg_metrics()
    result["eeg"] = eeg
    eeg_pass = all(v["pass"] for v in eeg["gates"].values())
    result["eeg_joint_pass"] = bool(eeg_pass)
    if not eeg_pass:
        result["decision"] = "STOP_EEG_MEASUREMENT_GATE"
    else:
        result["meg"] = meg_confirmation(group_correction)
        result["decision"] = "GO_STAGE1" if result["meg"]["joint_pass"] else "STOP_EXTERNAL_MEG_GATE"

    (OUT / "STAGE0_RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUT / "STAGE0_PARTICIPANTS.tsv").open("w", encoding="utf-8") as f:
        keys = list(eeg["participants"][0])
        f.write("\t".join(keys) + "\n")
        for row in eeg["participants"]:
            f.write("\t".join(str(row[k]) for k in keys) + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "eeg_gates": {k: {q: v[q] for q in ("mean", "n_positive", "exact_signflip_p_two_sided", "pass")} for k, v in eeg["gates"].items()},
        "meg": None if "meg" not in result else {
            "primary": result["meg"]["primary"],
            "prestimulus": result["meg"]["prestimulus_falsification"],
            "post_minus_pre": result["meg"]["post_minus_pre"],
            "joint_pass": result["meg"]["joint_pass"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
