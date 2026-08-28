from __future__ import annotations

"""Cross-recording RSA controls reported in Supplementary Figs. 10–11."""

import csv
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import label as connected_components
from scipy.stats import rankdata, spearmanr, t as student_t


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "derived" / "core_rsa_robustness"
OUT.mkdir(parents=True, exist_ok=True)

EEG_DIR = Path(os.environ.get("KANESHIRO_EEG_DIR", ROOT / "data" / "kaneshiro_eeg"))
MEG_FILE = Path(os.environ.get("CICHY_MEG_FILE", ROOT / "data" / "cichy_meg_rdms.mat"))
MAPPING = ROOT / "source_data" / "supplementary" / "stimulus_mapping.csv"
STIM_DIR = Path(os.environ.get("KANESHIRO_STIMULI_DIR", ROOT / "data" / "kaneshiro_stimuli"))

FEATURE_ROOT = ROOT / "source_data" / "model_features"
FEATURE_FILES = {
    "dinov3": FEATURE_ROOT / "features_dinov3_92x384.npy",
    "clip": FEATURE_ROOT / "features_clip_b32_image_92.npy",
    "siglip": FEATURE_ROOT / "features_siglip_base_image_92.npy",
    "caption_text": FEATURE_ROOT / "features_gte_blip_caption_92.npy",
}

SEED = 20260827
N_BOOT = 10_000
N_LABEL_PERM = 9_999
N_CLUSTER_PERM = 5_000
EEG_EARLY_IDX = np.arange(4, 10)   # 64, 80, ..., 144 ms
EEG_LATE_IDX = np.arange(12, 21)   # 192, 208, ..., 320 ms
MEG_EARLY_MS = (70, 130)
MEG_LATE_MS = (180, 300)
MEG_TIME_MIN_MS = -100
MEG_TIME_MAX_MS = 700
MEG_TIME_BIN_MS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def upper(matrix: np.ndarray, iu: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    if iu is None:
        iu = np.triu_indices(matrix.shape[0], 1)
    return np.asarray(matrix[iu], dtype=np.float64)


def vec_to_matrix(vector: np.ndarray, n: int = 72) -> np.ndarray:
    iu = np.triu_indices(n, 1)
    out = np.zeros((n, n), dtype=np.float64)
    out[iu] = vector
    out[(iu[1], iu[0])] = vector
    return out


def zscore(vector: np.ndarray) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float64)
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-12:
        raise RuntimeError("Degenerate vector")
    return (x - float(np.mean(x))) / sd


def zr(vector: np.ndarray) -> np.ndarray:
    return zscore(rankdata(np.asarray(vector), method="average"))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(zscore(a) * zscore(b)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)


def exact_signflip(values: np.ndarray, one_sided_positive: bool = False) -> float:
    v = np.asarray(values, dtype=np.float64)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(v))), dtype=np.float64)
    null = np.mean(signs * v[None, :], axis=1)
    observed = float(np.mean(v))
    if one_sided_positive:
        return float(np.mean(null >= observed - 1e-15))
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))


def bootstrap_mean_ci(values: np.ndarray, seed_offset: int = 0) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED + seed_offset)
    take = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    return [float(x) for x in np.quantile(v[take].mean(axis=1), [0.025, 0.975])]


def summarize(values: np.ndarray, seed_offset: int = 0) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "n": int(len(v)),
        "positive_n": int(np.sum(v > 0)),
        "exact_two_sided_signflip_p": exact_signflip(v),
        "bootstrap_mean_95ci": bootstrap_mean_ci(v, seed_offset),
        "values": [float(x) for x in v],
    }


def cross_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = x.shape[1]
    g = (x @ y.T) / p
    d = np.diag(g)[:, None] + np.diag(g)[None, :] - g - g.T
    np.fill_diagonal(d, 0.0)
    return d


def load_mapping() -> tuple[np.ndarray, np.ndarray, list[str]]:
    with MAPPING.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = sorted(rows, key=lambda row: int(row["kaneshiro_exemplar"]))
    if len(rows) != 72:
        raise RuntimeError(f"Expected 72 rows, got {len(rows)}")
    cichy_idx = np.asarray([int(row["cichy_index"]) - 1 for row in rows], dtype=int)
    text = [row["category"] for row in rows]
    names = list(dict.fromkeys(text))
    category = np.asarray([names.index(x) for x in text], dtype=int)
    if len(names) != 6 or not np.all(np.bincount(category) == 12):
        raise RuntimeError("Frozen category mapping is invalid")
    return cichy_idx, category, names


def occurrence_folds(labels: np.ndarray) -> np.ndarray:
    fold_id = np.empty(len(labels), dtype=int)
    counts = np.zeros(73, dtype=int)
    for trial, label_value in enumerate(labels.astype(int)):
        fold_id[trial] = counts[label_value] % 4
        counts[label_value] += 1
    if counts[1:].min() < 68:
        raise RuntimeError(f"Insufficient EEG repetitions: {counts[1:].min()}")
    return fold_id


def fold_patterns(
    x: np.ndarray,
    labels: np.ndarray,
    sample_idx: np.ndarray,
    fold_id: np.ndarray,
) -> np.ndarray:
    raw = np.transpose(x[:, sample_idx, :], (2, 0, 1)).reshape(len(labels), -1).astype(np.float64)
    out = np.empty((4, 72, raw.shape[1]), dtype=np.float64)
    for fold in range(4):
        train = fold_id != fold
        mu = raw[train].mean(axis=0)
        sd = raw[train].std(axis=0, ddof=1)
        sd[sd < 1e-8] = 1.0
        standardized = (raw - mu) / sd
        for image in range(1, 73):
            take = (fold_id == fold) & (labels == image)
            out[fold, image - 1] = standardized[take].mean(axis=0)
    return out


def nearest_centroid_decoding_rdm(patterns: np.ndarray) -> np.ndarray:
    """Balanced pairwise decoding from independent repetition folds.

    Two folds form class centroids and the other two provide four held-out class
    patterns; roles are then reversed.  The resulting eight decisions produce a
    decoding-accuracy dissimilarity for every image pair.
    """
    iu = np.triu_indices(72, 1)
    out = np.empty(len(iu[0]), dtype=np.float64)
    centroids = [patterns[[0, 1]].mean(axis=0), patterns[[2, 3]].mean(axis=0)]
    tests = [patterns[[2, 3]], patterns[[0, 1]]]
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        correct = 0
        total = 0
        for center, test in zip(centroids, tests):
            ci, cj = center[i], center[j]
            for fold in range(2):
                xi, xj = test[fold, i], test[fold, j]
                correct += int(np.sum((xi - ci) ** 2) < np.sum((xi - cj) ** 2))
                correct += int(np.sum((xj - cj) ** 2) < np.sum((xj - ci) ** 2))
                total += 2
        out[k] = correct / total
    return out


def load_eeg() -> dict:
    early_a, early_b, late_a, late_b = [], [], [], []
    early_decode, late_decode = [], []
    time_rdms = []
    eeg_times_ms = None
    for participant in range(1, 11):
        data = loadmat(
            EEG_DIR / f"S{participant}.mat",
            variable_names=["X_3D", "exemplarLabels", "categoryLabels", "Fs"],
            squeeze_me=True,
        )
        x = np.asarray(data["X_3D"])
        labels = np.asarray(data["exemplarLabels"]).ravel().astype(int)
        cats = np.asarray(data["categoryLabels"]).ravel().astype(int)
        fs = float(np.asarray(data["Fs"]).squeeze())
        if x.shape[:2] != (124, 32) or x.shape[2] != len(labels):
            raise RuntimeError(f"EEG structural mismatch in S{participant}: {x.shape}")
        for image in range(1, 73):
            observed = np.unique(cats[labels == image])
            expected = (image - 1) // 12 + 1
            if len(observed) != 1 or int(observed[0]) != expected:
                raise RuntimeError(f"EEG category mismatch S{participant}, image {image}")
        fold_id = occurrence_folds(labels)
        pe = fold_patterns(x, labels, EEG_EARLY_IDX, fold_id)
        pl = fold_patterns(x, labels, EEG_LATE_IDX, fold_id)
        early_a.append(upper(cross_distance(pe[0], pe[1])))
        early_b.append(upper(cross_distance(pe[2], pe[3])))
        late_a.append(upper(cross_distance(pl[0], pl[1])))
        late_b.append(upper(cross_distance(pl[2], pl[3])))
        early_decode.append(nearest_centroid_decoding_rdm(pe))
        late_decode.append(nearest_centroid_decoding_rdm(pl))

        participant_time = []
        for sample in range(x.shape[1]):
            pt = fold_patterns(x, labels, np.asarray([sample]), fold_id)
            va = upper(cross_distance(pt[0], pt[1]))
            vb = upper(cross_distance(pt[2], pt[3]))
            participant_time.append((zr(va) + zr(vb)) / 2.0)
        time_rdms.append(np.stack(participant_time))
        if eeg_times_ms is None:
            eeg_times_ms = np.arange(x.shape[1], dtype=float) * 1000.0 / fs
        print(f"EEG core RDMs: S{participant}/10", flush=True)

    early_a = np.stack(early_a)
    early_b = np.stack(early_b)
    late_a = np.stack(late_a)
    late_b = np.stack(late_b)
    mean_early = np.stack([(zr(a) + zr(b)) / 2.0 for a, b in zip(early_a, early_b)])
    mean_late = np.stack([(zr(a) + zr(b)) / 2.0 for a, b in zip(late_a, late_b)])
    return {
        "early_a": early_a,
        "early_b": early_b,
        "late_a": late_a,
        "late_b": late_b,
        "mean_early": mean_early,
        "mean_late": mean_late,
        "early_decode": np.stack(early_decode),
        "late_decode": np.stack(late_decode),
        "time_rdms": np.stack(time_rdms),
        "times_ms": np.asarray(eeg_times_ms),
    }


def load_meg(cichy_idx: np.ndarray) -> dict:
    windows = {"early": MEG_EARLY_MS, "late": MEG_LATE_MS}
    result = {name: [[], []] for name in windows}
    starts = np.arange(MEG_TIME_MIN_MS, MEG_TIME_MAX_MS + 1, MEG_TIME_BIN_MS)
    time_rdms = []
    with h5py.File(MEG_FILE, "r") as f:
        ds = f["MEG_decoding_RDMs"]
        if ds.shape != (92, 92, 1301, 2, 16):
            raise RuntimeError(f"Unexpected MEG shape: {ds.shape}")
        for participant in range(16):
            for name, (lo_ms, hi_ms) in windows.items():
                lo, hi = lo_ms + 100, hi_ms + 100
                for session in range(2):
                    matrix = np.nanmean(np.asarray(ds[:, :, lo : hi + 1, session, participant]), axis=2)
                    matrix = matrix[np.ix_(cichy_idx, cichy_idx)]
                    result[name][session].append(upper(matrix))
            participant_time = []
            for start in starts:
                stop = min(start + MEG_TIME_BIN_MS - 1, MEG_TIME_MAX_MS)
                lo, hi = start + 100, stop + 100
                matrix = np.nanmean(np.asarray(ds[:, :, lo : hi + 1, :, participant]), axis=(2, 3))
                matrix = matrix[np.ix_(cichy_idx, cichy_idx)]
                participant_time.append(upper(matrix))
            time_rdms.append(np.stack(participant_time))
            print(f"MEG core RDMs: P{participant + 1}/16", flush=True)
    out: dict[str, object] = {}
    for name in windows:
        a = np.stack(result[name][0])
        b = np.stack(result[name][1])
        out[name] = {
            "a": a,
            "b": b,
            "mean": np.stack([(zr(x) + zr(y)) / 2.0 for x, y in zip(a, b)]),
        }
    out["time_rdms"] = np.stack(time_rdms)
    out["times_ms"] = starts.astype(float) + (MEG_TIME_BIN_MS - 1) / 2.0
    return out


def cosine_rdm(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    matrix = 1.0 - x @ x.T
    np.fill_diagonal(matrix, 0.0)
    return matrix


def low_level_features() -> dict[str, np.ndarray]:
    files = sorted(STIM_DIR.glob("K*_C*.png"))
    if len(files) != 72:
        raise RuntimeError(f"Expected 72 stimuli, found {len(files)}")
    pixel, color, spatial = [], [], []
    for file in files:
        image = Image.open(file).convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.float64) / 255.0
        gray = np.dot(rgb[..., :3], np.asarray([0.299, 0.587, 0.114]))
        pixel.append(np.asarray(Image.fromarray(np.uint8(gray * 255)).resize((24, 24)), dtype=np.float64).ravel())

        color_row = []
        for channel in range(3):
            hist, _ = np.histogram(rgb[..., channel], bins=16, range=(0.0, 1.0), density=True)
            color_row.extend(hist.tolist())
            color_row.extend([float(rgb[..., channel].mean()), float(rgb[..., channel].std())])
        color.append(color_row)

        fft = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
        yy, xx = np.indices(gray.shape)
        center = (np.asarray(gray.shape) - 1) / 2.0
        radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
        radial = []
        bounds = np.linspace(0, radius.max() + 1e-9, 17)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            take = (radius >= lo) & (radius < hi)
            radial.append(float(np.log1p(fft[take]).mean()))
        gy, gx = np.gradient(gray)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = (np.arctan2(gy, gx) + np.pi) % np.pi
        orient = []
        for lo, hi in zip(np.linspace(0, np.pi, 9)[:-1], np.linspace(0, np.pi, 9)[1:]):
            orient.append(float(magnitude[(angle >= lo) & (angle < hi)].sum()))
        spatial.append(radial + orient)
    return {
        "pixel": np.asarray(pixel),
        "color": np.asarray(color),
        "spatial_frequency_edge": np.asarray(spatial),
    }


def category_design(category: np.ndarray, iu: tuple[np.ndarray, np.ndarray]) -> list[np.ndarray]:
    a = np.minimum(category[iu[0]], category[iu[1]])
    b = np.maximum(category[iu[0]], category[iu[1]])
    codes = a * 6 + b
    levels = sorted(np.unique(codes).tolist())
    return [(codes == level).astype(float) for level in levels[1:]]


def control_q(
    control_vectors: list[np.ndarray],
    category: np.ndarray,
    iu: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if iu is None:
        iu = np.triu_indices(len(category), 1)
    cols = [np.ones(len(iu[0]), dtype=np.float64)]
    cols.extend(zr(v) for v in control_vectors)
    cols.extend(category_design(category, iu))
    matrix = np.column_stack(cols)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    tol = np.finfo(float).eps * max(matrix.shape) * s[0]
    return u[:, s > tol]


def residual_rank(vector: np.ndarray, q: np.ndarray) -> np.ndarray:
    y = rankdata(np.asarray(vector), method="average").astype(np.float64)
    return zscore(y - q @ (q.T @ y))


def residual_pair(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    cols = [np.ones(len(y), dtype=np.float64)] + [rankdata(np.asarray(c), method="average") for c in controls]
    x = np.column_stack(cols)
    beta = np.linalg.lstsq(x, rankdata(np.asarray(y), method="average"), rcond=None)[0]
    return zscore(rankdata(np.asarray(y), method="average") - x @ beta)


def crossmodal_scores(
    eeg_vectors: np.ndarray,
    meg_vectors: np.ndarray,
    q: np.ndarray,
) -> dict:
    eeg_res = np.stack([residual_rank(v, q) for v in eeg_vectors])
    meg_res = np.stack([residual_rank(v, q) for v in meg_vectors])
    eeg_group = zscore(eeg_res.mean(axis=0))
    meg_group = zscore(meg_res.mean(axis=0))
    meg_to_eeg = eeg_res @ meg_group / eeg_res.shape[1]
    eeg_to_meg = meg_res @ eeg_group / meg_res.shape[1]
    return {
        "direct_group_rho": corr(eeg_group, meg_group),
        "meg_group_to_eeg": summarize(meg_to_eeg, 110),
        "eeg_group_to_meg": summarize(eeg_to_meg, 111),
        "equal_direction_mean": float(0.5 * (meg_to_eeg.mean() + eeg_to_meg.mean())),
        "eeg_residuals": eeg_res,
        "meg_residuals": meg_res,
        "eeg_group": eeg_group,
        "meg_group": meg_group,
    }


def crossfitted_early_sensitivity(
    eeg: dict,
    meg: dict,
    base_q: np.ndarray,
    dino_vector: np.ndarray,
    category_vectors: list[np.ndarray],
) -> dict:
    n_eeg = eeg["late_a"].shape[0]
    loo_residuals = []
    iv_residuals = []
    iv_betas = []
    for participant in range(n_eeg):
        others = np.arange(n_eeg) != participant
        early_group = zscore(eeg["mean_early"][others].mean(axis=0))
        ra = residual_pair(eeg["late_a"][participant], [early_group])
        rb = residual_pair(eeg["late_b"][participant], [early_group])
        loo_residuals.append(zscore(0.5 * (ra + rb)))

        ea = rankdata(eeg["early_a"][participant], method="average").astype(float)
        eb = rankdata(eeg["early_b"][participant], method="average").astype(float)
        la = rankdata(eeg["late_a"][participant], method="average").astype(float)
        lb = rankdata(eeg["late_b"][participant], method="average").astype(float)
        ea -= ea.mean(); eb -= eb.mean(); la -= la.mean(); lb -= lb.mean()
        denom = float(np.dot(ea, eb))
        if abs(denom) < 1e-12:
            raise RuntimeError("Instrumental-variable denominator was degenerate")
        beta_a = float(np.dot(eb, la) / denom)
        beta_b = float(np.dot(ea, lb) / denom)
        iv_betas.append([beta_a, beta_b])
        iv_residuals.append(zscore(0.5 * (zscore(la - beta_a * ea) + zscore(lb - beta_b * eb))))

    def evaluate_teacher(participant_residuals: np.ndarray, label: str) -> dict:
        teacher = zscore(np.mean(participant_residuals, axis=0))
        values = []
        for participant in range(16):
            controls = [meg["early"]["mean"][participant], dino_vector] + category_vectors
            pred = residual_pair(teacher, controls)
            target = residual_pair(meg["late"]["mean"][participant], controls)
            values.append(corr(pred, target))
        return {"label": label, "meg_participant_scores": summarize(np.asarray(values), 120)}

    conventional = []
    conventional_participant_residuals = []
    for participant in range(n_eeg):
        ra = residual_pair(eeg["late_a"][participant], [eeg["early_a"][participant]])
        rb = residual_pair(eeg["late_b"][participant], [eeg["early_b"][participant]])
        conventional_participant_residuals.append(zscore(0.5 * (ra + rb)))
    conventional = evaluate_teacher(np.stack(conventional_participant_residuals), "same-half conventional residual")
    loo = evaluate_teacher(np.stack(loo_residuals), "held-out group-early residual")
    iv = evaluate_teacher(np.stack(iv_residuals), "split-half instrumental-variable residual")
    return {
        "conventional": conventional,
        "heldout_group_early": loo,
        "instrumental_variable": iv,
        "instrumental_variable_betas": {
            "mean": [float(x) for x in np.mean(iv_betas, axis=0)],
            "range": [float(np.min(iv_betas)), float(np.max(iv_betas))],
            "participant_values": [[float(x) for x in row] for row in iv_betas],
        },
    }


def decoding_estimator_sensitivity(
    eeg: dict,
    meg: dict,
    q: np.ndarray,
    category: np.ndarray,
) -> dict:
    late = crossmodal_scores(eeg["late_decode"], meg["late"]["mean"], q)
    eeg_group_late = zscore(np.mean([residual_rank(v, q) for v in eeg["late_decode"]], axis=0))
    eeg_group_early = zscore(np.mean([residual_rank(v, q) for v in eeg["early_decode"]], axis=0))
    teacher = residual_pair(eeg_group_late, [eeg_group_early])
    values = []
    for participant in range(16):
        target_late = residual_rank(meg["late"]["mean"][participant], q)
        target_early = residual_rank(meg["early"]["mean"][participant], q)
        target = residual_pair(target_late, [target_early])
        values.append(corr(teacher, target))
    values = np.asarray(values)

    rng = np.random.default_rng(SEED + 201)
    n = 72
    teacher_matrix = vec_to_matrix(teacher)
    target_residuals = np.stack([
        residual_pair(
            residual_rank(meg["late"]["mean"][participant], q),
            [residual_rank(meg["early"]["mean"][participant], q)],
        )
        for participant in range(16)
    ])
    null = np.empty(N_LABEL_PERM, dtype=float)
    for permutation in range(N_LABEL_PERM):
        perm = np.arange(n)
        for category_value in range(6):
            indices = np.flatnonzero(category == category_value)
            perm[indices] = rng.permutation(indices)
        shuffled = upper(teacher_matrix[np.ix_(perm, perm)])
        shuffled = zscore(shuffled)
        null[permutation] = float(np.mean(target_residuals @ shuffled / len(shuffled)))
    observed = float(np.mean(values))
    p = float((1 + np.sum(null >= observed)) / (N_LABEL_PERM + 1))
    return {
        "late_geometry": _strip_arrays(late),
        "late_after_early": summarize(values, 202),
        "late_after_early_category_preserving_permutation": {
            "n": N_LABEL_PERM,
            "observed": observed,
            "exceedances": int(np.sum(null >= observed)),
            "one_sided_add_one_p": p,
            "null_mean": float(null.mean()),
            "null_95th": float(np.quantile(null, 0.95)),
        },
        "null": null,
    }


def full_time_by_time_rsa(
    eeg: dict,
    meg: dict,
    q: np.ndarray,
) -> dict:
    eeg_group = []
    for time_index in range(eeg["time_rdms"].shape[1]):
        residuals = np.stack([residual_rank(v, q) for v in eeg["time_rdms"][:, time_index]])
        eeg_group.append(zscore(residuals.mean(axis=0)))
    eeg_group = np.stack(eeg_group)

    participant_maps = []
    for participant in range(meg["time_rdms"].shape[0]):
        meg_residuals = np.stack([residual_rank(v, q) for v in meg["time_rdms"][participant]])
        participant_maps.append(eeg_group @ meg_residuals.T / eeg_group.shape[1])
    participant_maps = np.stack(participant_maps)
    mean_map = participant_maps.mean(axis=0)

    n = participant_maps.shape[0]
    mean = participant_maps.mean(axis=0)
    sd = participant_maps.std(axis=0, ddof=1)
    t_map = mean / np.maximum(sd / math.sqrt(n), 1e-12)
    threshold = float(student_t.ppf(0.95, df=n - 1))
    observed_labels, observed_n = connected_components(t_map > threshold)
    clusters = []
    for cluster_id in range(1, observed_n + 1):
        take = observed_labels == cluster_id
        clusters.append({
            "id": cluster_id,
            "mass": float(t_map[take].sum()),
            "eeg_index_min": int(np.where(take)[0].min()),
            "eeg_index_max": int(np.where(take)[0].max()),
            "meg_index_min": int(np.where(take)[1].min()),
            "meg_index_max": int(np.where(take)[1].max()),
            "n_cells": int(take.sum()),
        })

    rng = np.random.default_rng(SEED + 301)
    max_mass = np.zeros(N_CLUSTER_PERM, dtype=float)
    for permutation in range(N_CLUSTER_PERM):
        signs = rng.choice((-1.0, 1.0), size=(n, 1, 1))
        signed = participant_maps * signs
        pm = signed.mean(axis=0)
        ps = signed.std(axis=0, ddof=1)
        pt = pm / np.maximum(ps / math.sqrt(n), 1e-12)
        labels, count = connected_components(pt > threshold)
        if count:
            max_mass[permutation] = max(float(pt[labels == cluster_id].sum()) for cluster_id in range(1, count + 1))
    for cluster in clusters:
        cluster["one_sided_cluster_p"] = float((1 + np.sum(max_mass >= cluster["mass"])) / (N_CLUSTER_PERM + 1))
        cluster["eeg_ms"] = [
            float(eeg["times_ms"][cluster["eeg_index_min"]]),
            float(eeg["times_ms"][cluster["eeg_index_max"]]),
        ]
        cluster["meg_ms"] = [
            float(meg["times_ms"][cluster["meg_index_min"]]),
            float(meg["times_ms"][cluster["meg_index_max"]]),
        ]
    return {
        "participant_maps": participant_maps,
        "mean_map": mean_map,
        "t_map": t_map,
        "cluster_forming_t": threshold,
        "n_cluster_permutations": N_CLUSTER_PERM,
        "clusters": clusters,
        "max_cluster_mass_null": max_mass,
        "eeg_times_ms": eeg["times_ms"],
        "meg_times_ms": meg["times_ms"],
    }


def crossed_participant_image_bootstrap(
    eeg_residuals: np.ndarray,
    meg_residuals: np.ndarray,
) -> dict:
    eeg_matrices = np.stack([vec_to_matrix(v) for v in eeg_residuals])
    meg_matrices = np.stack([vec_to_matrix(v) for v in meg_residuals])
    rng = np.random.default_rng(SEED + 401)
    values = np.empty(N_BOOT, dtype=float)
    positions = np.triu_indices(72, 1)
    for bootstrap in range(N_BOOT):
        eeg_take = rng.integers(0, len(eeg_matrices), size=len(eeg_matrices))
        meg_take = rng.integers(0, len(meg_matrices), size=len(meg_matrices))
        images = rng.integers(0, 72, size=72)
        keep = images[positions[0]] != images[positions[1]]
        rows = images[positions[0][keep]]
        cols = images[positions[1][keep]]
        eeg_group = eeg_matrices[eeg_take].mean(axis=0)[rows, cols]
        meg_group = meg_matrices[meg_take].mean(axis=0)[rows, cols]
        values[bootstrap] = spearman(eeg_group, meg_group)
    return {
        "n": N_BOOT,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "95ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "proportion_at_or_below_zero": float((1 + np.sum(values <= 0)) / (N_BOOT + 1)),
        "definition": "EEG participants, MEG participants and image identities resampled; duplicate-image self-pairs omitted; nuisance coefficients fixed from the full sample",
        "values": values,
    }


def leave_one_category_out(
    eeg_vectors: np.ndarray,
    meg_vectors: np.ndarray,
    control_matrices: list[np.ndarray],
    category: np.ndarray,
    category_names: list[str],
) -> list[dict]:
    rows = []
    for left_out in range(6):
        images = np.flatnonzero(category != left_out)
        iu_local = np.triu_indices(len(images), 1)
        eeg_sub = np.stack([upper(vec_to_matrix(v)[np.ix_(images, images)], iu_local) for v in eeg_vectors])
        meg_sub = np.stack([upper(vec_to_matrix(v)[np.ix_(images, images)], iu_local) for v in meg_vectors])
        controls_sub = [upper(m[np.ix_(images, images)], iu_local) for m in control_matrices]
        q = control_q(controls_sub, category[images], iu_local)
        result = crossmodal_scores(eeg_sub, meg_sub, q)
        rows.append({
            "left_out_category": category_names[left_out],
            "direct_group_rho": result["direct_group_rho"],
            "meg_group_to_eeg_mean": result["meg_group_to_eeg"]["mean"],
            "eeg_group_to_meg_mean": result["eeg_group_to_meg"]["mean"],
        })
    return rows


def _strip_arrays(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {"eeg_residuals", "meg_residuals", "eeg_group", "meg_group"}}


def main() -> None:
    started = time.time()
    cichy_idx, category, category_names = load_mapping()
    eeg = load_eeg()
    meg = load_meg(cichy_idx)

    model_features = {}
    for name, path in FEATURE_FILES.items():
        array = np.load(path).astype(np.float64)[cichy_idx]
        model_features[name] = array
    low_features = low_level_features()
    feature_rdms = {name: cosine_rdm(array) for name, array in {**model_features, **low_features}.items()}
    feature_vectors = {name: upper(matrix) for name, matrix in feature_rdms.items()}

    primary_controls = [feature_vectors["dinov3"]]
    primary_q = control_q(primary_controls, category)
    primary = crossmodal_scores(eeg["mean_late"], meg["late"]["mean"], primary_q)

    control_sets = {
        "category_plus_dinov3": ["dinov3"],
        "category_plus_low_level": ["pixel", "color", "spatial_frequency_edge"],
        "category_plus_vision_models": ["dinov3", "clip", "siglip"],
        "category_plus_all_visual_and_caption": [
            "pixel", "color", "spatial_frequency_edge", "dinov3", "clip", "siglip", "caption_text"
        ],
    }
    broader_controls = {}
    for name, members in control_sets.items():
        q = control_q([feature_vectors[x] for x in members], category)
        broader_controls[name] = _strip_arrays(crossmodal_scores(eeg["mean_late"], meg["late"]["mean"], q))

    category_vectors = category_design(category, np.triu_indices(72, 1))
    crossfit = crossfitted_early_sensitivity(
        eeg,
        meg,
        primary_q,
        feature_vectors["dinov3"],
        category_vectors,
    )
    decoding = decoding_estimator_sensitivity(eeg, meg, primary_q, category)
    time_by_time = full_time_by_time_rsa(eeg, meg, primary_q)
    crossed_bootstrap = crossed_participant_image_bootstrap(primary["eeg_residuals"], primary["meg_residuals"])
    loco = leave_one_category_out(
        eeg["mean_late"],
        meg["late"]["mean"],
        [feature_rdms["dinov3"]],
        category,
        category_names,
    )

    result = {
        "analysis": "post-review core EEG-MEG robustness audit",
        "status": "COMPLETE",
        "interpretation_status": "POST_OUTCOME_SENSITIVITY_NOT_PREREGISTERED",
        "sample": {"eeg_participants": 10, "meg_participants": 16, "images": 72},
        "windows_ms": {
            "eeg_early": [64, 144],
            "eeg_late": [192, 320],
            "meg_early": list(MEG_EARLY_MS),
            "meg_late": list(MEG_LATE_MS),
            "time_by_time_meg": [MEG_TIME_MIN_MS, MEG_TIME_MAX_MS, MEG_TIME_BIN_MS],
        },
        "primary_late_geometry": _strip_arrays(primary),
        "broader_control_sensitivity": broader_controls,
        "crossfitted_early_sensitivity": crossfit,
        "decoding_estimator_sensitivity": {key: value for key, value in decoding.items() if key != "null"},
        "time_by_time_rsa": {
            key: value for key, value in time_by_time.items()
            if key not in {"participant_maps", "mean_map", "t_map", "max_cluster_mass_null", "eeg_times_ms", "meg_times_ms"}
        },
        "crossed_participant_image_bootstrap": {
            key: value for key, value in crossed_bootstrap.items() if key != "values"
        },
        "leave_one_category_out": loco,
        "runtime_seconds": float(time.time() - started),
        "provenance": {
            "script_sha256": sha256(Path(__file__)),
            "mapping_sha256": sha256(MAPPING),
            "meg_sha256": sha256(MEG_FILE),
            "feature_sha256": {name: sha256(path) for name, path in FEATURE_FILES.items()},
        },
    }
    np.savez_compressed(
        OUT / "CORE_RSA_ARRAYS_v001.npz",
        eeg_times_ms=time_by_time["eeg_times_ms"],
        meg_times_ms=time_by_time["meg_times_ms"],
        time_by_time_participant_maps=time_by_time["participant_maps"],
        time_by_time_mean=time_by_time["mean_map"],
        time_by_time_t=time_by_time["t_map"],
        time_by_time_max_cluster_mass_null=time_by_time["max_cluster_mass_null"],
        decoding_label_permutation_null=decoding["null"],
        crossed_bootstrap=crossed_bootstrap["values"],
        primary_eeg_residuals=primary["eeg_residuals"],
        primary_meg_residuals=primary["meg_residuals"],
    )
    (OUT / "CORE_RSA_RESULTS_v001.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
