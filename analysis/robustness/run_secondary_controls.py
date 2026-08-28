from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "derived" / "secondary_controls"
ABLATION = ROOT / "source_data" / "supplementary" / "single_measurement_ablation.npz"
NOD_CACHE = Path(os.environ.get("NOD_RDM_CACHE", ROOT / "data" / "nod_rdm_cache"))


def exact_signflip_positive(values: np.ndarray) -> float:
    """Exact one-sided sign-flip P for a positive mean."""
    v = np.asarray(values, dtype=np.float64)
    observed = float(v.mean())
    count = 0
    total = 1 << len(v)
    for start in range(0, total, 65_536):
        ids = np.arange(start, min(start + 65_536, total), dtype=np.uint64)
        bits = ((ids[:, None] >> np.arange(len(v), dtype=np.uint64)) & 1).astype(float)
        means = ((2.0 * bits - 1.0) @ v) / len(v)
        count += int(np.sum(means >= observed - 1e-15))
    return float(count / total)


def tost(values: np.ndarray, margin: float) -> dict:
    """Equivalence test under a symmetric +/- margin using exact sign flips."""
    v = np.asarray(values, dtype=np.float64)
    p_above_lower = exact_signflip_positive(v + margin)
    p_below_upper = exact_signflip_positive(margin - v)
    return {
        "mean_difference": float(v.mean()),
        "margin": float(margin),
        "p_mean_above_minus_margin": p_above_lower,
        "p_mean_below_plus_margin": p_below_upper,
        "tost_p": float(max(p_above_lower, p_below_upper)),
        "equivalent_at_alpha_0.05": bool(max(p_above_lower, p_below_upper) < 0.05),
        "values": v.tolist(),
    }


def rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10_000) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def nod_group_reliability() -> dict:
    files = sorted(NOD_CACHE.glob("sub-*_paired_rdms_v001.npz"))
    if len(files) != 19:
        raise RuntimeError(f"Expected 19 paired NOD caches, found {len(files)}")
    rows: dict[str, list[np.ndarray]] = {"eeg_native_post": [], "meg_native_post": []}
    subjects: list[str] = []
    classes_ref = None
    for path in files:
        z = np.load(path, allow_pickle=False)
        classes = z["classes"].astype(str)
        if classes_ref is None:
            classes_ref = classes
        elif not np.array_equal(classes_ref, classes):
            raise RuntimeError(f"Class order differs in {path}")
        subjects.append(str(z["subject"].item()))
        for key in rows:
            rows[key].append(np.asarray(z[key], dtype=np.float64))

    out: dict[str, object] = {"subjects": subjects, "n": len(subjects)}
    # Prespecified deterministic odd/even participant split.  Because every vector
    # uses the same 1,000 ImageNet classes, this measures reproducibility of the
    # concept-level geometry across independent participant groups.
    split_a = np.arange(0, len(subjects), 2)
    split_b = np.arange(1, len(subjects), 2)
    out["split_a_subjects"] = [subjects[i] for i in split_a]
    out["split_b_subjects"] = [subjects[i] for i in split_b]
    for key, vectors in rows.items():
        x = np.stack(vectors)
        split_rho = rho(x[split_a].mean(axis=0), x[split_b].mean(axis=0))
        loo = []
        for i in range(len(x)):
            others = np.delete(x, i, axis=0).mean(axis=0)
            loo.append(rho(x[i], others))
        loo_arr = np.asarray(loo)
        # Lower/upper RSA noise-ceiling analogues following the standard
        # participant-to-group construction.  Upper includes the participant in
        # the group mean; lower excludes it.
        upper = [rho(x[i], x.mean(axis=0)) for i in range(len(x))]
        out[key] = {
            "odd_even_group_rho": split_rho,
            "leave_one_participant_to_group_mean": float(loo_arr.mean()),
            "leave_one_participant_to_group_95ci": bootstrap_ci(loo_arr, 20260827),
            "leave_one_participant_to_group_positive_n": int(np.sum(loo_arr > 0)),
            "leave_one_participant_to_group_values": loo_arr.tolist(),
            "noise_ceiling_lower_mean": float(np.mean(loo)),
            "noise_ceiling_upper_mean": float(np.mean(upper)),
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ABLATION, allow_pickle=False)
    names = z["teacher_names"].astype(str).tolist()
    idx = {name: i for i, name in enumerate(names)}
    eeg = z["eeg_gain"].mean(axis=2)
    meg = z["meg_gain"].mean(axis=2)
    margin = 0.005
    result = {
        "analysis": "secondary professor-feedback controls",
        "equivalence_margin_basis": (
            "The symmetric +/-0.005 margin matches the manuscript's fixed minimum-effect "
            "criterion and was not selected from these contrasts."
        ),
        "equivalence": {
            "consensus_minus_eeg_only_in_heldout_eeg": tost(
                eeg[idx["consensus"]] - eeg[idx["eeg_only"]], margin
            ),
            "consensus_minus_meg_only_in_heldout_meg": tost(
                meg[idx["consensus"]] - meg[idx["meg_only"]], margin
            ),
            "consensus_minus_meg_only_in_heldout_eeg": tost(
                eeg[idx["consensus"]] - eeg[idx["meg_only"]], margin
            ),
            "consensus_minus_eeg_only_in_heldout_meg": tost(
                meg[idx["consensus"]] - meg[idx["eeg_only"]], margin
            ),
        },
        "nod_group_reliability": nod_group_reliability(),
    }
    path = OUT / "SECONDARY_CONTROLS_RESULTS_v001.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
