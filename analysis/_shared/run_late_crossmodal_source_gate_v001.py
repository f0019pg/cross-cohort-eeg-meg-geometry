from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[2]

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.stats import rankdata, spearmanr


OUT = Path(os.environ.get("EEG_MEG_OUTPUT_DIR", ROOT_DEFAULT / "derived" / "source_geometry"))
PROTOCOL = OUT / "PROTOCOL_LOCK_v001_KR.md"
MAPPING_NAME = "source_data/supplementary/stimulus_mapping.csv"
DINO_NAME = "source_data/model_features/dinov3_72.npy"
EEG_WINDOWS = {"late": np.arange(12, 21)}  # 192, 208, ..., 320 ms
MEG_WINDOWS = {"late": (180, 300), "pre": (-100, -1)}
N_PERM = 9_999
N_BOOT = 10_000
SEED = 20260806


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_sources(root: Path, eeg_dir_arg: str | None, meg_file_arg: str | None) -> tuple[Path, Path]:
    eeg_candidates = [
        Path(eeg_dir_arg) if eeg_dir_arg else None,
        Path(os.environ["KANESHIRO_EEG_DIR"]) if os.environ.get("KANESHIRO_EEG_DIR") else None,
    ]
    meg_candidates = [
        Path(meg_file_arg) if meg_file_arg else None,
        Path(os.environ["CICHY_MEG_FILE"]) if os.environ.get("CICHY_MEG_FILE") else None,
    ]
    eeg = next((p for p in eeg_candidates if p is not None and p.exists()), None)
    meg = next((p for p in meg_candidates if p is not None and p.exists()), None)
    if eeg is None:
        raise FileNotFoundError("Set --eeg-dir or KANESHIRO_EEG_DIR")
    if meg is None:
        raise FileNotFoundError("Set --meg-file or CICHY_MEG_FILE")
    return eeg, meg


def upper(x: np.ndarray, indices: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    iu = indices if indices is not None else np.triu_indices(x.shape[0], 1)
    return np.asarray(x[iu], dtype=np.float64)


def vec_to_matrix(v: np.ndarray, n: int = 72, indices: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    iu = indices if indices is not None else np.triu_indices(n, 1)
    m = np.zeros((n, n), dtype=np.float64)
    m[iu] = v
    m[(iu[1], iu[0])] = v
    return m


def zscore(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    sd = x.std()
    if not np.isfinite(sd) or sd < 1e-12:
        raise RuntimeError("Degenerate vector")
    return (x - x.mean()) / sd


def zr(v: np.ndarray) -> np.ndarray:
    return zscore(rankdata(np.asarray(v), method="average"))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(zscore(a) * zscore(b)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)


def exact_signflip(values: list[float] | np.ndarray, one_sided_positive: bool = False) -> float:
    v = np.asarray(values, dtype=np.float64)
    observed = float(v.mean())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(v))), dtype=np.float64)
    null = np.mean(signs * v[None, :], axis=1)
    if one_sided_positive:
        return float(np.mean(null >= observed - 1e-15))
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))


def bootstrap_ci(values: list[float] | np.ndarray, seed_offset: int) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED + seed_offset)
    take = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    return [float(x) for x in np.quantile(v[take].mean(axis=1), [0.025, 0.975])]


def summarize(values: list[float] | np.ndarray, seed_offset: int) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "positive_n": int(np.sum(v > 0)),
        "n": int(len(v)),
        "exact_two_sided_signflip_p": exact_signflip(v),
        "bootstrap_mean_95ci": bootstrap_ci(v, seed_offset),
        "values": [float(x) for x in v],
    }


def cross_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = x.shape[1]
    g = (x @ y.T) / p
    d = np.diag(g)[:, None] + np.diag(g)[None, :] - g - g.T
    np.fill_diagonal(d, 0.0)
    return d


def fold_patterns(x: np.ndarray, labels: np.ndarray, sample_idx: np.ndarray) -> np.ndarray:
    labels = labels.astype(int)
    fold_id = np.empty(len(labels), dtype=int)
    counts = np.zeros(73, dtype=int)
    for trial, label in enumerate(labels):
        fold_id[trial] = counts[label] % 4
        counts[label] += 1
    if counts[1:].min() < 68:
        raise RuntimeError(f"Insufficient EEG repetitions: {counts[1:].min()}")

    raw = np.transpose(x[:, sample_idx, :], (2, 0, 1)).reshape(len(labels), -1).astype(np.float64)
    out = np.empty((4, 72, raw.shape[1]), dtype=np.float64)
    for fold in range(4):
        train = fold_id != fold
        mu = raw[train].mean(axis=0)
        sd = raw[train].std(axis=0, ddof=1)
        sd[sd < 1e-8] = 1.0
        standardized = (raw - mu) / sd
        for label in range(1, 73):
            take = (fold_id == fold) & (labels == label)
            if take.sum() < 17:
                raise RuntimeError(f"EEG fold count too small: fold={fold}, label={label}, n={take.sum()}")
            out[fold, label - 1] = standardized[take].mean(axis=0)
    return out


def load_mapping(mapping_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 72:
        raise RuntimeError(f"Expected 72 mapping rows, got {len(rows)}")
    rows = sorted(rows, key=lambda r: int(r["kaneshiro_exemplar"]))
    cichy_idx = np.asarray([int(r["cichy_index"]) - 1 for r in rows], dtype=int)
    categories_text = [r["category"] for r in rows]
    category_names = list(dict.fromkeys(categories_text))
    category = np.asarray([category_names.index(x) for x in categories_text], dtype=int)
    counts = np.bincount(category, minlength=len(category_names))
    if len(category_names) != 6 or not np.all(counts == 12) or len(np.unique(cichy_idx)) != 72:
        raise RuntimeError(f"Invalid mapping categories/counts: {category_names}, {counts.tolist()}")
    return cichy_idx, category, category_names


def load_eeg(eeg_dir: Path) -> dict:
    half_a, half_b, reliabilities = [], [], []
    for participant in range(1, 11):
        d = loadmat(
            eeg_dir / f"S{participant}.mat",
            variable_names=["X_3D", "exemplarLabels", "categoryLabels"],
            squeeze_me=True,
        )
        x = np.asarray(d["X_3D"])
        labels = np.asarray(d["exemplarLabels"]).ravel().astype(int)
        cats = np.asarray(d["categoryLabels"]).ravel().astype(int)
        if x.shape[:2] != (124, 32) or x.shape[2] != len(labels):
            raise RuntimeError(f"S{participant} EEG structural mismatch: {x.shape}")
        for label in range(1, 73):
            observed = np.unique(cats[labels == label])
            expected = (label - 1) // 12 + 1
            if len(observed) != 1 or int(observed[0]) != expected:
                raise RuntimeError(f"S{participant} label/category mismatch at image {label}")
        patterns = fold_patterns(x, labels, EEG_WINDOWS["late"])
        a = upper(cross_distance(patterns[0], patterns[1]))
        b = upper(cross_distance(patterns[2], patterns[3]))
        half_a.append(a)
        half_b.append(b)
        reliabilities.append(spearman(a, b))
        print(f"EEG late RDM: S{participant}/10", flush=True)
    a = np.stack(half_a)
    b = np.stack(half_b)
    mean = np.stack([(zr(x) + zr(y)) / 2.0 for x, y in zip(a, b)])
    return {"a": a, "b": b, "mean": mean, "reliability": np.asarray(reliabilities)}


def load_meg(meg_file: Path, cichy_idx: np.ndarray) -> dict:
    vectors = {name: [[], []] for name in MEG_WINDOWS}
    with h5py.File(meg_file, "r") as f:
        if "MEG_decoding_RDMs" not in f:
            raise RuntimeError(f"Missing MEG_decoding_RDMs in {meg_file}")
        ds = f["MEG_decoding_RDMs"]
        if ds.shape != (92, 92, 1301, 2, 16):
            raise RuntimeError(f"Unexpected MEG shape: {ds.shape}")
        for participant in range(16):
            for name, (lo_ms, hi_ms) in MEG_WINDOWS.items():
                lo, hi = lo_ms + 100, hi_ms + 100
                for session in range(2):
                    matrix = np.nanmean(np.asarray(ds[:, :, lo : hi + 1, session, participant]), axis=2)
                    matrix = matrix[np.ix_(cichy_idx, cichy_idx)]
                    vectors[name][session].append(upper(matrix))
            print(f"MEG RDMs: P{participant + 1}/16", flush=True)
    out = {}
    for name in MEG_WINDOWS:
        a = np.stack(vectors[name][0])
        b = np.stack(vectors[name][1])
        out[name] = {"a": a, "b": b, "mean": np.stack([(zr(x) + zr(y)) / 2.0 for x, y in zip(a, b)])}
    out["reliability"] = np.asarray([spearman(a, b) for a, b in zip(out["late"]["a"], out["late"]["b"])])
    return out


def control_basis(dino: np.ndarray, category: np.ndarray, within: bool) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    full_iu = np.triu_indices(72, 1)
    if within:
        keep = category[full_iu[0]] == category[full_iu[1]]
        iu = (full_iu[0][keep], full_iu[1][keep])
        codes = category[iu[0]]
    else:
        iu = full_iu
        a = np.minimum(category[iu[0]], category[iu[1]])
        b = np.maximum(category[iu[0]], category[iu[1]])
        codes = a * 6 + b
    dino_v = upper(dino, iu)
    levels = sorted(np.unique(codes).tolist())
    dummies = np.column_stack([(codes == level).astype(float) for level in levels[1:]])
    x = np.column_stack([np.ones(len(dino_v)), zr(dino_v), dummies])
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    tol = np.finfo(float).eps * max(x.shape) * s[0]
    q = u[:, s > tol]
    return q, iu


def residual_rank(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    y = rankdata(np.asarray(v), method="average").astype(np.float64)
    return zscore(y - q @ (q.T @ y))


def participant_targets(vectors: np.ndarray, q: np.ndarray, iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    full_iu = np.triu_indices(72, 1)
    if len(iu[0]) == len(full_iu[0]):
        take = slice(None)
    else:
        mask = np.zeros((72, 72), dtype=bool)
        mask[iu] = True
        take = mask[full_iu]
    return np.stack([residual_rank(v[take], q) for v in vectors])


def group_teacher(vectors: np.ndarray, indices: np.ndarray, q: np.ndarray, iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    full_iu = np.triu_indices(72, 1)
    if len(iu[0]) == len(full_iu[0]):
        take = slice(None)
    else:
        mask = np.zeros((72, 72), dtype=bool)
        mask[iu] = True
        take = mask[full_iu]
    participant_residuals = [residual_rank(vectors[i][take], q) for i in indices]
    return zscore(np.mean(participant_residuals, axis=0))


def permuted_teacher(teacher: np.ndarray, perm: np.ndarray, q: np.ndarray, iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    matrix = vec_to_matrix(teacher, 72, iu)
    vector = upper(matrix[np.ix_(perm, perm)], iu)
    return residual_rank(vector, q)


def generate_within_category_permutation(rng: np.random.Generator, category: np.ndarray) -> np.ndarray:
    perm = np.arange(72)
    for cat in range(6):
        idx = np.flatnonzero(category == cat)
        perm[idx] = rng.permutation(idx)
    return perm


def check_inputs(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    mapping = root / MAPPING_NAME
    dino_path = root / DINO_NAME
    cichy_idx, category, category_names = load_mapping(mapping)
    dino_features = np.load(dino_path, mmap_mode="r")
    if dino_features.shape != (72, 384) or not np.isfinite(np.asarray(dino_features)).all():
        raise RuntimeError(f"Unexpected DINO shape/content: {dino_features.shape}")
    eeg_files = [eeg_dir / f"S{i}.mat" for i in range(1, 11)]
    if not all(p.exists() for p in eeg_files):
        raise RuntimeError("One or more Kaneshiro EEG files are missing")
    with h5py.File(meg_file, "r") as f:
        meg_shape = tuple(f["MEG_decoding_RDMs"].shape)
    if meg_shape != (92, 92, 1301, 2, 16):
        raise RuntimeError(f"Unexpected MEG shape: {meg_shape}")
    audit = {
        "status": "INPUT_CHECK_PASSED",
        "root": str(root),
        "eeg_dir": str(eeg_dir),
        "meg_file": str(meg_file),
        "eeg_participants": 10,
        "meg_participants": 16,
        "images": 72,
        "category_names": category_names,
        "category_counts": np.bincount(category).tolist(),
        "cichy_indices_unique": int(len(np.unique(cichy_idx))),
        "dino_shape": list(dino_features.shape),
        "meg_shape": list(meg_shape),
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "mapping": sha256(mapping),
            "dino": sha256(dino_path),
            "script": sha256(Path(__file__)),
        },
    }
    (OUT / "INPUT_AUDIT_v001.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def run_stage0(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    audit = check_inputs(root, eeg_dir, meg_file)
    mapping = root / MAPPING_NAME
    dino_path = root / DINO_NAME
    cichy_idx, category, category_names = load_mapping(mapping)
    features = np.load(dino_path).astype(np.float64)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    dino = 1.0 - features @ features.T
    np.fill_diagonal(dino, 0.0)

    q_all, iu_all = control_basis(dino, category, within=False)
    q_within, iu_within = control_basis(dino, category, within=True)
    eeg = load_eeg(eeg_dir)
    meg = load_meg(meg_file, cichy_idx)

    eeg_all_targets = participant_targets(eeg["mean"], q_all, iu_all)
    meg_all_targets = participant_targets(meg["late"]["mean"], q_all, iu_all)
    meg_pre_targets = participant_targets(meg["pre"]["mean"], q_all, iu_all)
    eeg_within_targets = participant_targets(eeg["mean"], q_within, iu_within)
    meg_within_targets = participant_targets(meg["late"]["mean"], q_within, iu_within)

    eeg_teacher = group_teacher(eeg["mean"], np.arange(10), q_all, iu_all)
    meg_teacher = group_teacher(meg["late"]["mean"], np.arange(16), q_all, iu_all)
    eeg_teacher_within = group_teacher(eeg["mean"], np.arange(10), q_within, iu_within)
    meg_teacher_within = group_teacher(meg["late"]["mean"], np.arange(16), q_within, iu_within)

    eeg_scores = np.asarray([corr(meg_teacher, target) for target in eeg_all_targets])
    meg_scores = np.asarray([corr(eeg_teacher, target) for target in meg_all_targets])
    meg_pre_scores = np.asarray([corr(eeg_teacher, target) for target in meg_pre_targets])
    eeg_within_scores = np.asarray([corr(meg_teacher_within, target) for target in eeg_within_targets])
    meg_within_scores = np.asarray([corr(eeg_teacher_within, target) for target in meg_within_targets])

    crossmodal_teacher = {
        "eeg_teacher_participants": list(range(1, 11)),
        "eeg_evaluation_participants": "independent MEG participants 1-16",
        "meg_teacher_participants": list(range(1, 17)),
        "meg_evaluation_participants": "independent EEG participants 1-10",
        "direct_crossmodal_teacher_rho": corr(eeg_teacher, meg_teacher),
        "direct_crossmodal_teacher_within_category_rho": corr(eeg_teacher_within, meg_teacher_within),
    }

    observed_combined = 0.5 * (eeg_scores.mean() + meg_scores.mean())
    observed_within_combined = 0.5 * (eeg_within_scores.mean() + meg_within_scores.mean())
    rng = np.random.default_rng(SEED)
    null_all = np.empty(N_PERM, dtype=np.float64)
    null_within = np.empty(N_PERM, dtype=np.float64)
    for p in range(N_PERM):
        perm = generate_within_category_permutation(rng, category)
        perm_meg = permuted_teacher(meg_teacher, perm, q_all, iu_all)
        perm_eeg = permuted_teacher(eeg_teacher, perm, q_all, iu_all)
        perm_meg_within = permuted_teacher(meg_teacher_within, perm, q_within, iu_within)
        perm_eeg_within = permuted_teacher(eeg_teacher_within, perm, q_within, iu_within)
        null_all[p] = 0.5 * (
            np.mean([corr(perm_meg, target) for target in eeg_all_targets])
            + np.mean([corr(perm_eeg, target) for target in meg_all_targets])
        )
        null_within[p] = 0.5 * (
            np.mean([corr(perm_meg_within, target) for target in eeg_within_targets])
            + np.mean([corr(perm_eeg_within, target) for target in meg_within_targets])
        )
        if (p + 1) % 1000 == 0:
            print(f"within-category label permutation: {p + 1}/{N_PERM}", flush=True)

    permutation_all_p = float((1 + np.sum(null_all >= observed_combined)) / (N_PERM + 1))
    permutation_within_p = float((1 + np.sum(null_within >= observed_within_combined)) / (N_PERM + 1))

    r1 = summarize(eeg["reliability"], 1)
    r2 = summarize(meg["reliability"], 2)
    s1 = summarize(eeg_scores, 3)
    s2 = summarize(meg_scores, 4)
    s5_pre = summarize(meg_pre_scores, 5)
    s5_pre["exact_one_sided_positive_signflip_p"] = exact_signflip(meg_pre_scores, one_sided_positive=True)
    s5_diff = summarize(meg_scores - meg_pre_scores, 6)
    s6_eeg = summarize(eeg_within_scores, 7)
    s6_meg = summarize(meg_within_scores, 8)
    direct = float(crossmodal_teacher["direct_crossmodal_teacher_rho"])
    direct_within = float(crossmodal_teacher["direct_crossmodal_teacher_within_category_rho"])

    gates = {
        "R1_eeg_late_reliability": bool(r1["mean"] > 0.10 and r1["positive_n"] >= 8 and r1["exact_two_sided_signflip_p"] < 0.05),
        "R2_meg_late_session_reliability": bool(r2["mean"] > 0.02 and r2["positive_n"] >= 12 and r2["exact_two_sided_signflip_p"] < 0.05),
        "S1_meg_teacher_to_heldout_eeg": bool(s1["mean"] > 0.03 and s1["positive_n"] >= 8 and s1["exact_two_sided_signflip_p"] < 0.05),
        "S2_eeg_teacher_to_heldout_meg": bool(s2["mean"] > 0.03 and s2["positive_n"] >= 12 and s2["exact_two_sided_signflip_p"] < 0.05),
        "S3_direct_teacher_agreement": bool(direct > 0.05),
        "S4_label_permutation": bool(permutation_all_p < 0.05),
        "S5_late_specificity": bool(s5_diff["mean"] > 0.02 and s5_diff["positive_n"] >= 12 and s5_diff["exact_two_sided_signflip_p"] < 0.05 and s5_pre["exact_one_sided_positive_signflip_p"] >= 0.05),
        "S6_within_category": bool(s6_eeg["mean"] > 0.02 and s6_eeg["positive_n"] >= 8 and s6_eeg["exact_two_sided_signflip_p"] < 0.05 and s6_meg["mean"] > 0.02 and s6_meg["positive_n"] >= 12 and s6_meg["exact_two_sided_signflip_p"] < 0.05 and permutation_within_p < 0.05),
    }
    decision = "GO_STAGE1_ADAPTER" if all(gates.values()) else "STOP_OR_LIMITED_SOURCE"

    result = {
        "analysis": "prospectively locked late EEG-MEG shared relational geometry source gate",
        "decision": decision,
        "input_audit": audit,
        "windows_ms": {"eeg_late": [192, 320], "meg_late": [180, 300], "meg_pre": [-100, -1]},
        "controls": "rank(DINOv3 distance) + complete six-category pair-type dummy design",
        "crossmodal_teacher": crossmodal_teacher,
        "reliability": {"eeg": r1, "meg": r2},
        "crossmodal_transfer": {
            "meg_teacher_to_heldout_eeg": s1,
            "eeg_teacher_to_heldout_meg": s2,
            "equal_modality_weighted_mean": float(observed_combined),
            "within_category_label_permutation": {
                "n": N_PERM,
                "seed": SEED,
                "null_mean": float(null_all.mean()),
                "null_sd": float(null_all.std()),
                "null_95th": float(np.quantile(null_all, 0.95)),
                "one_sided_p": permutation_all_p,
            },
        },
        "late_specificity": {"meg_prestimulus": s5_pre, "meg_late_minus_prestimulus": s5_diff},
        "within_category": {
            "meg_teacher_to_heldout_eeg": s6_eeg,
            "eeg_teacher_to_heldout_meg": s6_meg,
            "direct_teacher_rho": direct_within,
            "equal_modality_weighted_mean": float(observed_within_combined),
            "label_permutation": {
                "n": N_PERM,
                "seed": SEED,
                "null_mean": float(null_within.mean()),
                "null_sd": float(null_within.std()),
                "null_95th": float(np.quantile(null_within, 0.95)),
                "one_sided_p": permutation_within_p,
            },
        },
        "gates": gates,
        "category_names": category_names,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "script": sha256(Path(__file__)),
            "mapping": sha256(mapping),
            "dino": sha256(dino_path),
            "meg": sha256(meg_file),
        },
    }
    np.savez_compressed(OUT / "STAGE0_NULLS_v001.npz", all_pairs=null_all, within_category=null_within)
    (OUT / "STAGE0_RESULTS_v001.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "decision": decision,
        "reliability": {"eeg": r1, "meg": r2},
        "crossmodal": {"meg_to_eeg": s1, "eeg_to_meg": s2, "permutation_p": permutation_all_p},
        "direct_teacher_rho": direct,
        "late_specificity": {"pre": s5_pre, "late_minus_pre": s5_diff},
        "within_category": {"meg_to_eeg": s6_eeg, "eeg_to_meg": s6_meg, "permutation_p": permutation_within_p},
        "gates": gates,
    }, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--eeg-dir", type=str)
    parser.add_argument("--meg-file", type=str)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-inputs", action="store_true")
    action.add_argument("--run-stage0", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    eeg_dir, meg_file = find_sources(root, args.eeg_dir, args.meg_file)
    if args.check_inputs:
        print(json.dumps(check_inputs(root, eeg_dir, meg_file), indent=2, ensure_ascii=False))
    else:
        run_stage0(root, eeg_dir, meg_file)


if __name__ == "__main__":
    main()
