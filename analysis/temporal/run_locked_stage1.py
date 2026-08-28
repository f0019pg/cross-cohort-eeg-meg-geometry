from __future__ import annotations

from pathlib import Path
import hashlib
import itertools
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.stats import rankdata, spearmanr

import run_locked_stage0 as s0


OUT = ROOT / "derived" / "temporal_stage1"
EEG_DIR = s0.EEG_DIR
MEG_FILE = s0.MEG_FILE
MAPPING = s0.MAPPING
DINO_FEATURES = s0.DINO_FEATURES
STAGE0_RESULTS = OUT / "STAGE0_RESULTS.json"
SPEC = OUT / "PROSPECTIVE_STAGE1_DISTILLATION_SPEC.md"

RANKS = (1, 2, 4, 8)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
WEIGHTS = (0.03, 0.10, 0.30)
N_SHUFFLES = 20
SEED = 20260722
RNG = np.random.default_rng(SEED)
N_BOOT = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)


def upper_vector(x: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(x.shape[0], 1)
    return np.asarray(x[i, j], dtype=np.float64)


def vector_to_symmetric(v: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.float64)
    i, j = np.triu_indices(n, 1)
    m[i, j] = v
    m[j, i] = v
    return m


def zscore_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    if sd < 1e-12:
        raise RuntimeError("Zero-variance geometry")
    return (v - v.mean()) / sd


def euclidean_distance(x: np.ndarray) -> np.ndarray:
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (x @ x.T), 0.0)
    return np.sqrt(d2)


def cosine_distance(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    z = x / n
    return 1.0 - z @ z.T


def subset_vector(m: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return upper_vector(m[np.ix_(ids, ids)])


def exact_signflip(values: np.ndarray, two_sided: bool = True) -> float:
    v = np.asarray(values, dtype=np.float64)
    obs = float(v.mean())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(v))))
    null = np.mean(signs * v[None, :], axis=1)
    if two_sided:
        return float(np.mean(np.abs(null) >= abs(obs) - 1e-15))
    return float(np.mean(null >= obs - 1e-15))


def bootstrap_ci(values: np.ndarray) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED + 17)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    return [float(x) for x in np.quantile(v[idx].mean(axis=1), [0.025, 0.975])]


def summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "n_positive": int(np.sum(v > 0)),
        "n_negative": int(np.sum(v < 0)),
        "n_total": int(len(v)),
        "exact_signflip_p_two_sided": exact_signflip(v, True),
        "bootstrap_mean_95ci": bootstrap_ci(v),
        "values": v.tolist(),
    }


def residualize_rank(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    yr = rankdata(y).astype(np.float64)
    cols = [np.ones(len(yr), dtype=np.float64)]
    cols.extend(rankdata(c).astype(np.float64) for c in controls)
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, yr, rcond=None)[0]
    return yr - X @ beta


def partial_rho(y: np.ndarray, x: np.ndarray, controls: list[np.ndarray]) -> float:
    return rho(residualize_rank(y, controls), residualize_rank(x, controls))


def load_source_geometries() -> tuple[np.ndarray, np.ndarray, dict]:
    early_idx = np.arange(4, 10)
    late_idx = np.arange(12, 21)
    early_halves, late_halves, correction_halves = [], [], []
    audit = []

    for subject in range(1, 11):
        d = loadmat(
            EEG_DIR / f"S{subject}.mat",
            variable_names=["X_3D", "exemplarLabels"],
            squeeze_me=True,
        )
        X = np.asarray(d["X_3D"])
        labels = np.asarray(d["exemplarLabels"]).ravel().astype(int)
        pe = s0.fold_patterns(X, labels, early_idx)
        pl = s0.fold_patterns(X, labels, late_idx)
        ea = upper_vector(s0.cross_distance(pe[0], pe[1]))
        eb = upper_vector(s0.cross_distance(pe[2], pe[3]))
        la = upper_vector(s0.cross_distance(pl[0], pl[1]))
        lb = upper_vector(s0.cross_distance(pl[2], pl[3]))
        ca = residualize_rank(la, [ea])
        cb = residualize_rank(lb, [eb])
        early_halves.extend((zscore_vector(ea), zscore_vector(eb)))
        late_halves.extend((zscore_vector(la), zscore_vector(lb)))
        correction_halves.append(zscore_vector((ca + cb) / 2.0))
        audit.append({
            "participant": subject,
            "early_reliability": rho(ea, eb),
            "late_reliability": rho(la, lb),
            "correction_reliability": rho(ca, cb),
        })

    # Match the Stage-0 definition exactly: standardize each participant's
    # averaged pair of independent correction estimates before group averaging.
    correction = np.mean(np.stack(correction_halves), axis=0)
    late = np.mean(np.stack(late_halves), axis=0)

    locked = json.loads(STAGE0_RESULTS.read_text(encoding="utf-8"))
    for row, old in zip(audit, locked["eeg"]["participants"]):
        for key in ("early_reliability", "late_reliability", "correction_reliability"):
            if not np.isclose(row[key], old[key], atol=1e-12, rtol=0):
                raise RuntimeError(f"Stage-0 reproducibility mismatch S{row['participant']} {key}")

    return vector_to_symmetric(correction, 72), vector_to_symmetric(late, 72), {
        "stage0_recalculation_verified": True,
        "participants": audit,
    }


def rank_distance_for_mds(d: np.ndarray) -> np.ndarray:
    # The Stage-0 correction is a signed residualized rank geometry, not a
    # metric distance. Classical MDS therefore receives its affine rank map to
    # [0,1], fixed before Stage-1 outcomes and invariant to signed scale.
    n = d.shape[0]
    v = upper_vector(d)
    r = rankdata(v, method="average")
    if len(r) > 1:
        r = (r - 1.0) / (len(r) - 1.0)
    return vector_to_symmetric(r, n)


def spectral_coordinates(d: np.ndarray, requested_rank: int) -> np.ndarray:
    dm = rank_distance_for_mds(d)
    n = dm.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dm * dm) @ J
    vals, vecs = np.linalg.eigh((B + B.T) / 2.0)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = np.flatnonzero(vals > max(1e-12, vals[0] * 1e-10))[:requested_rank]
    if len(keep) == 0:
        raise RuntimeError("No positive spectral dimensions")
    return vecs[:, keep] * np.sqrt(vals[keep])[None, :]


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray,
                  x_test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    xm = x_train.mean(axis=0, keepdims=True)
    ym = y_train.mean(axis=0, keepdims=True)
    X = x_train - xm
    Y = y_train - ym
    # Dual ridge is stable for p >> n and keeps the frozen backbone untouched.
    coef_dual = np.linalg.solve(X @ X.T + alpha * np.eye(len(X)), Y)
    pred_train = X @ (X.T @ coef_dual) + ym
    pred_test = (x_test - xm) @ (X.T @ coef_dual) + ym
    return pred_train, pred_test


def adapted_features(z_train: np.ndarray, q_train: np.ndarray,
                     z_test: np.ndarray, q_test: np.ndarray,
                     weight: float) -> tuple[np.ndarray, np.ndarray]:
    qm = q_train.mean(axis=0, keepdims=True)
    qs = q_train.std(axis=0, keepdims=True)
    qs[qs < 1e-8] = 1.0
    qtr = (q_train - qm) / qs
    qte = (q_test - qm) / qs
    atr = np.concatenate((z_train, weight * qtr), axis=1)
    ate = np.concatenate((z_test, weight * qte), axis=1)
    atr /= np.linalg.norm(atr, axis=1, keepdims=True)
    ate /= np.linalg.norm(ate, axis=1, keepdims=True)
    return atr, ate


def fit_adapter(teacher: np.ndarray, z: np.ndarray, train: np.ndarray,
                test: np.ndarray, hp: tuple[int, float, float]) -> dict:
    rank, alpha, weight = hp
    q_target = spectral_coordinates(teacher[np.ix_(train, train)], rank)
    q_train, q_test = ridge_predict(z[train], q_target, z[test], alpha)
    a_train, a_test = adapted_features(z[train], q_train, z[test], q_test, weight)
    return {
        "adapted_train": a_train,
        "adapted_test": a_test,
        "predicted_neural_train": q_train,
        "predicted_neural_test": q_test,
        "effective_rank": int(q_target.shape[1]),
    }


def make_outer_folds() -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    all_ids = np.arange(72)
    within = np.tile(np.arange(12), 6)
    for f in range(3):
        test = all_ids[within % 3 == f]
        train = all_ids[within % 3 != f]
        if len(test) != 24 or len(train) != 48:
            raise RuntimeError("Outer fold mismatch")
        folds.append((train, test))
    return folds


def make_inner_folds(outer_train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    # Parity is recomputed within each category's ordered outer-training list,
    # yielding balanced 24/24 inner splits.
    folds = []
    for parity in (0, 1):
        val = []
        for c in range(6):
            ids = outer_train[(outer_train // 12) == c]
            val.extend(ids[np.arange(len(ids)) % 2 == parity])
        val = np.asarray(sorted(val), dtype=int)
        train = np.asarray(sorted(set(outer_train.tolist()) - set(val.tolist())), dtype=int)
        if len(train) != 24 or len(val) != 24:
            raise RuntimeError("Inner fold mismatch")
        folds.append((train, val))
    return folds


def choose_hyperparameters(teacher: np.ndarray, late: np.ndarray,
                           z: np.ndarray, outer_train: np.ndarray) -> tuple[tuple[int, float, float], list[dict]]:
    records = []
    for rank in RANKS:
        for alpha in ALPHAS:
            for weight in WEIGHTS:
                hp = (rank, alpha, weight)
                scores = []
                for inner_train, inner_val in make_inner_folds(outer_train):
                    fit = fit_adapter(teacher, z, inner_train, inner_val, hp)
                    av = upper_vector(cosine_distance(fit["adapted_test"]))
                    dv = subset_vector(cosine_distance(z), inner_val)
                    lv = subset_vector(late, inner_val)
                    scores.append(rho(av, lv) - rho(dv, lv))
                records.append({
                    "rank": rank, "alpha": alpha, "weight": weight,
                    "inner_delta_mean": float(np.mean(scores)),
                    "inner_delta_values": [float(x) for x in scores],
                })
    best = max(records, key=lambda x: (
        x["inner_delta_mean"], -x["rank"], x["alpha"], -x["weight"]
    ))
    return (best["rank"], best["alpha"], best["weight"]), records


def source_validation(teacher: np.ndarray, late: np.ndarray, z: np.ndarray) -> dict:
    folds_out, all_records = [], []
    for fold, (train, test) in enumerate(make_outer_folds()):
        hp, records = choose_hyperparameters(teacher, late, z, train)
        fit = fit_adapter(teacher, z, train, test, hp)
        adapted = upper_vector(cosine_distance(fit["adapted_test"]))
        dino = subset_vector(cosine_distance(z), test)
        late_v = subset_vector(late, test)
        correction_v = subset_vector(teacher, test)
        pred_correction = upper_vector(euclidean_distance(fit["predicted_neural_test"]))
        fold_row = {
            "fold": fold,
            "n_train_images": len(train),
            "n_test_images": len(test),
            "selected_rank": hp[0],
            "selected_alpha": hp[1],
            "selected_weight": hp[2],
            "effective_rank": fit["effective_rank"],
            "adapted_late_rho": rho(adapted, late_v),
            "dino_late_rho": rho(dino, late_v),
            "late_delta_rho": rho(adapted, late_v) - rho(dino, late_v),
            "predicted_correction_rho": rho(pred_correction, correction_v),
            "dino_preservation_rho": rho(adapted, dino),
            "test_indices_zero_based": test.tolist(),
        }
        folds_out.append(fold_row)
        all_records.append({"fold": fold, "candidates": records})

    deltas = np.asarray([x["late_delta_rho"] for x in folds_out])
    correction = np.asarray([x["predicted_correction_rho"] for x in folds_out])
    preservation = np.asarray([x["dino_preservation_rho"] for x in folds_out])
    gates = {
        "late_improvement": {
            "mean": float(deltas.mean()), "n_positive": int(np.sum(deltas > 0)),
            "threshold": 0.01,
            "pass": bool(deltas.mean() > .01 and np.sum(deltas > 0) == 3),
        },
        "predicted_correction": {
            "mean": float(correction.mean()), "n_positive": int(np.sum(correction > 0)),
            "threshold": 0.05,
            "pass": bool(correction.mean() > .05 and np.sum(correction > 0) == 3),
        },
        "geometry_preservation": {
            "mean": float(preservation.mean()), "threshold": 0.95,
            "pass": bool(preservation.mean() > .95),
        },
    }
    return {
        "folds": folds_out,
        "hyperparameter_audit": all_records,
        "gates": gates,
        "joint_pass": bool(all(x["pass"] for x in gates.values())),
    }


def build_outer_adapters(teacher: np.ndarray, z: np.ndarray,
                         selected: list[tuple[int, float, float]]) -> list[dict]:
    out = []
    for (train, test), hp in zip(make_outer_folds(), selected):
        fit = fit_adapter(teacher, z, train, test, hp)
        out.append({"train": train, "test": test, "fit": fit})
    return out


def stitched_model_vectors(adapters: list[dict], z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dino, adapted = [], []
    full_dino = cosine_distance(z)
    for item in adapters:
        test = item["test"]
        dino.append(subset_vector(full_dino, test))
        adapted.append(upper_vector(cosine_distance(item["fit"]["adapted_test"])))
    return np.concatenate(dino), np.concatenate(adapted)


def load_meg_stitched() -> dict[str, list[np.ndarray]]:
    mapping = np.genfromtxt(MAPPING, delimiter=",", names=True, dtype=None, encoding="utf-8")
    idx = np.asarray(mapping["cichy_index"], dtype=int) - 1
    windows = {"early": (70, 130), "late": (180, 300), "pre": (-100, -1)}
    out = {k: [] for k in windows}
    folds = make_outer_folds()
    with h5py.File(MEG_FILE, "r") as f:
        ds = f["MEG_decoding_RDMs"]
        if ds.shape != (92, 92, 1301, 2, 16):
            raise RuntimeError(f"Unexpected MEG shape {ds.shape}")
        for p in range(16):
            participant = {k: [] for k in windows}
            for name, (lo_ms, hi_ms) in windows.items():
                lo, hi = lo_ms + 100, hi_ms + 100
                block = np.asarray(ds[:, :, lo:hi + 1, :, p])
                matrix = np.nanmean(block, axis=(2, 3))[np.ix_(idx, idx)]
                for _, test in folds:
                    participant[name].append(subset_vector(matrix, test))
                out[name].append(np.concatenate(participant[name]))
    return out


def external_metrics(dino: np.ndarray, adapted: np.ndarray,
                     meg: dict[str, list[np.ndarray]]) -> dict:
    delta, pre_delta, early_dino = [], [], []
    for p in range(16):
        early = meg["early"][p]
        late = meg["late"][p]
        pre = meg["pre"][p]
        base = partial_rho(late, dino, [early])
        new = partial_rho(late, adapted, [early])
        delta.append(new - base)
        base_pre = partial_rho(pre, dino, [early])
        new_pre = partial_rho(pre, adapted, [early])
        pre_delta.append(new_pre - base_pre)
        early_dino.append(rho(early, dino))
    delta_s = summary(np.asarray(delta))
    pre_s = summary(np.asarray(pre_delta))
    early_s = summary(np.asarray(early_dino))
    pre_s["one_sided_positive_p"] = exact_signflip(np.asarray(pre_delta), False)
    preservation = rho(adapted, dino)
    return {
        "primary_delta": delta_s,
        "prestimulus_delta": pre_s,
        "early_dino_positive_control": early_s,
        "adapted_to_dino_geometry_rho": preservation,
    }


def shuffled_teacher_null(teacher: np.ndarray, z: np.ndarray,
                          selected: list[tuple[int, float, float]],
                          meg: dict[str, list[np.ndarray]], observed: float) -> dict:
    rng = np.random.default_rng(SEED)
    null_means = []
    permutations = []
    for _ in range(N_SHUFFLES):
        perm = rng.permutation(72)
        shuffled = teacher[np.ix_(perm, perm)]
        adapters = build_outer_adapters(shuffled, z, selected)
        dino, adapted = stitched_model_vectors(adapters, z)
        metrics = external_metrics(dino, adapted, meg)
        null_means.append(metrics["primary_delta"]["mean"])
        permutations.append(perm.tolist())
    null = np.asarray(null_means)
    return {
        "n_fixed_shuffles": N_SHUFFLES,
        "seed": SEED,
        "null_mean_deltas": null.tolist(),
        "null_max": float(null.max()),
        "null_mean": float(null.mean()),
        "observed_mean": float(observed),
        "observed_exceeds_all": bool(observed > null.max()),
        "permutations_zero_based": permutations,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    locked0 = json.loads(STAGE0_RESULTS.read_text(encoding="utf-8"))
    if locked0.get("decision") != "GO_STAGE1":
        raise RuntimeError("Stage 0 did not authorize Stage 1")

    correction, late, source_audit = load_source_geometries()
    z = np.load(DINO_FEATURES).astype(np.float64)
    z /= np.linalg.norm(z, axis=1, keepdims=True)

    result = {
        "analysis": "locked cross-modal EEG temporal-correction distillation Stage 1",
        "script_sha256": sha256(Path(__file__)),
        "spec_sha256": sha256(SPEC),
        "stage0_results_sha256": sha256(STAGE0_RESULTS),
        "mapping_sha256": sha256(MAPPING),
        "dino_feature_sha256": sha256(DINO_FEATURES),
        "meg_file_sha256": sha256(MEG_FILE),
        "source_recalculation_audit": source_audit,
    }

    source = source_validation(correction, late, z)
    result["source_validation"] = source
    if not source["joint_pass"]:
        result["decision"] = "STOP_SOURCE_GENERALIZATION"
        (OUT / "STAGE1_RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"decision": result["decision"], "source_gates": source["gates"]}, indent=2))
        return

    selected = [(x["selected_rank"], x["selected_alpha"], x["selected_weight"])
                for x in source["folds"]]
    adapters = build_outer_adapters(correction, z, selected)
    dino, adapted = stitched_model_vectors(adapters, z)

    # The independent MEG file is not opened until all source gates pass.
    meg = load_meg_stitched()
    ext = external_metrics(dino, adapted, meg)
    shuffle = shuffled_teacher_null(correction, z, selected, meg, ext["primary_delta"]["mean"])
    result["external_meg"] = ext
    result["shuffled_teacher_falsification"] = shuffle

    primary = ext["primary_delta"]
    external_gate = (
        primary["mean"] > .01
        and primary["n_positive"] >= 12
        and primary["exact_signflip_p_two_sided"] < .05
        and primary["bootstrap_mean_95ci"][0] > 0
        and ext["adapted_to_dino_geometry_rho"] > .95
    )
    prestim_gate = ext["prestimulus_delta"]["one_sided_positive_p"] >= .05
    early = ext["early_dino_positive_control"]
    positive_control_gate = early["n_positive"] >= 12 and early["exact_signflip_p_two_sided"] < .05
    shuffled_gate = shuffle["observed_exceeds_all"]
    result["external_gates"] = {
        "primary_external_utility": bool(external_gate),
        "prestimulus_falsification": bool(prestim_gate),
        "early_dino_positive_control": bool(positive_control_gate),
        "shuffled_teacher_falsification": bool(shuffled_gate),
    }
    result["decision"] = (
        "GO_CROSSMODAL_DISTILLATION"
        if all(result["external_gates"].values())
        else "STOP_EXTERNAL_UTILITY"
    )
    (OUT / "STAGE1_RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (OUT / "STAGE1_SOURCE_FOLDS.tsv").open("w", encoding="utf-8") as f:
        keys = [k for k in source["folds"][0] if k != "test_indices_zero_based"]
        f.write("\t".join(keys) + "\n")
        for row in source["folds"]:
            f.write("\t".join(str(row[k]) for k in keys) + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "source_gates": source["gates"],
        "external_primary": ext["primary_delta"],
        "prestimulus": ext["prestimulus_delta"],
        "early_dino": ext["early_dino_positive_control"],
        "preservation": ext["adapted_to_dino_geometry_rho"],
        "shuffle": {k: shuffle[k] for k in ("null_max", "null_mean", "observed_mean", "observed_exceeds_all")},
        "external_gates": result["external_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
