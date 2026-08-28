from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "analysis" / "_shared"
sys.path.insert(0, str(SHARED))

import numpy as np
import torch
from scipy.stats import spearmanr

from common import adapted_concept_embeddings, fit_adapter
from run_late_crossmodal_adapter_gate_v001 import (
    category_control_basis,
    cosine_rdm,
    evaluate_rdm,
    group_matrix,
    make_object_folds,
    subset_vec,
)
from run_late_crossmodal_source_gate_v001 import (
    DINO_NAME,
    MAPPING_NAME,
    bootstrap_ci,
    exact_signflip,
    load_eeg,
    load_mapping,
    load_meg,
    upper,
    zr,
    zscore,
)
from run_three_teacher_matched_ablation_v001 import participant_folds


OUT = ROOT / "derived" / "adapter_stability"
EEG_DIR = Path(os.environ.get("KANESHIRO_EEG_DIR", ROOT / "data" / "kaneshiro_eeg"))
MEG_FILE = Path(os.environ.get("CICHY_MEG_FILE", ROOT / "data" / "cichy_meg_rdms.mat"))
SEEDS = (20260722, 20260723, 20260724)
LAMBDA_ANCHOR = 100.0
EPOCHS = 400
EEG_RELIABILITY = 0.5351525309130989
MEG_RELIABILITY = 0.31538592725005304


def summary(values: np.ndarray, seed: int) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "positive_n": int(np.sum(v > 0)),
        "n": int(len(v)),
        "exact_two_sided_signflip_p": float(exact_signflip(v)),
        "bootstrap_mean_95ci": bootstrap_ci(v, seed),
        "values": v.tolist(),
    }


def target(eeg_group: np.ndarray, meg_group: np.ndarray, idx: np.ndarray, eeg_weight: float) -> np.ndarray:
    eeg = zr(upper(eeg_group[np.ix_(idx, idx)]))
    meg = zr(upper(meg_group[np.ix_(idx, idx)]))
    return np.asarray(zscore(eeg_weight * eeg + (1.0 - eeg_weight) * meg), dtype=np.float32)


def category_balanced_random_folds(category: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    test_sets = [list() for _ in range(3)]
    for cat in np.unique(category):
        ids = np.flatnonzero(category == cat)
        if len(ids) != 12:
            raise RuntimeError("Expected 12 images per category")
        ids = rng.permutation(ids)
        for fold in range(3):
            test_sets[fold].extend(ids[4 * fold : 4 * (fold + 1)].tolist())
    all_ids = np.arange(len(category))
    out = []
    for fold, test in enumerate(test_sets):
        test = np.asarray(sorted(test), dtype=int)
        train = np.setdiff1d(all_ids, test)
        out.append({"fold": fold, "train": train, "test": test})
    return out


def evaluate_configuration(
    features: np.ndarray,
    category: np.ndarray,
    eeg: dict,
    meg: dict,
    object_folds: list[dict],
    eeg_weight: float,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg_gain = np.full((10, len(object_folds)), np.nan)
    meg_gain = np.full((16, len(object_folds)), np.nan)
    geometry = np.full((2, len(object_folds)), np.nan)
    for p_idx, p_fold in enumerate(participant_folds()):
        eeg_group = group_matrix(eeg["mean"], p_fold["eeg_teacher"])
        meg_group = group_matrix(meg["late"]["mean"], p_fold["meg_teacher"])
        for f_idx, o_fold in enumerate(object_folds):
            train, test = o_fold["train"], o_fold["test"]
            base_rdm = cosine_rdm(features[test])
            q = category_control_basis(base_rdm, category[test])
            eeg_neural = [subset_vec(eeg["mean"][i], test) for i in p_fold["eeg_eval"]]
            meg_neural = [subset_vec(meg["late"]["mean"][i], test) for i in p_fold["meg_eval"]]
            y = target(eeg_group, meg_group, train, eeg_weight)
            embeds = []
            for seed in seeds:
                model = fit_adapter(
                    features[train, None, :],
                    y,
                    lambda_anchor=LAMBDA_ANCHOR,
                    seed=seed,
                    epochs=EPOCHS,
                )
                embeds.append(adapted_concept_embeddings(model, features[test, None, :]))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            ensemble = np.mean(embeds, axis=0)
            ensemble /= np.maximum(np.linalg.norm(ensemble, axis=1, keepdims=True), 1e-12)
            adapted_rdm = cosine_rdm(ensemble)
            eeg_values, _ = evaluate_rdm(adapted_rdm, base_rdm, eeg_neural, q)
            meg_values, _ = evaluate_rdm(adapted_rdm, base_rdm, meg_neural, q)
            eeg_gain[p_fold["eeg_eval"], f_idx] = eeg_values
            meg_gain[p_fold["meg_eval"], f_idx] = meg_values
            geometry[p_idx, f_idx] = float(spearmanr(upper(adapted_rdm), upper(base_rdm)).statistic)
    if not np.isfinite(eeg_gain).all() or not np.isfinite(meg_gain).all():
        raise RuntimeError("Missing evaluation cells")
    return eeg_gain, meg_gain, geometry


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cichy_idx, category, categories = load_mapping(ROOT / MAPPING_NAME)
    features = np.load(ROOT / DINO_NAME).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    eeg = load_eeg(EEG_DIR)
    meg = load_meg(MEG_FILE, cichy_idx)

    rel_weight = EEG_RELIABILITY / (EEG_RELIABILITY + MEG_RELIABILITY)
    fixed = make_object_folds(category)
    print(f"Reliability-weighted target: EEG={rel_weight:.6f}, MEG={1-rel_weight:.6f}", flush=True)
    rw_eeg, rw_meg, rw_geometry = evaluate_configuration(
        features, category, eeg, meg, fixed, rel_weight, SEEDS
    )

    repeat_eeg = []
    repeat_meg = []
    repeat_fold_means = []
    # Ten deterministic category-balanced repartitions isolate sensitivity to the
    # held-out image assignment.  One fixed optimization seed avoids conflating
    # split variability with optimizer variability; the primary analysis already
    # ensembles three seeds.
    for repeat in range(10):
        split_seed = 2026082700 + repeat
        print(f"Random split repeat {repeat + 1}/10", flush=True)
        folds = category_balanced_random_folds(category, split_seed)
        eg, mg, _ = evaluate_configuration(
            features, category, eeg, meg, folds, 0.5, (SEEDS[0],)
        )
        repeat_eeg.append(eg.mean(axis=1))
        repeat_meg.append(mg.mean(axis=1))
        repeat_fold_means.append(
            {"eeg": eg.mean(axis=0).tolist(), "meg": mg.mean(axis=0).tolist()}
        )

    repeat_eeg = np.stack(repeat_eeg)
    repeat_meg = np.stack(repeat_meg)
    repeat_eeg_means = repeat_eeg.mean(axis=1)
    repeat_meg_means = repeat_meg.mean(axis=1)
    result = {
        "analysis": "adapter reliability-weighting and held-out-image split stability",
        "analysis_class": "post-hoc sensitivity analyses requested during manuscript audit",
        "model": {
            "backbone": "DINOv3",
            "bottleneck": 64,
            "lambda_anchor": LAMBDA_ANCHOR,
            "epochs": EPOCHS,
            "primary_seeds": list(SEEDS),
        },
        "reliability_weighted_target": {
            "eeg_weight": float(rel_weight),
            "meg_weight": float(1.0 - rel_weight),
            "weight_basis": {
                "eeg_source_late_split_half_reliability": EEG_RELIABILITY,
                "meg_source_late_session_reliability": MEG_RELIABILITY,
            },
            "heldout_eeg": summary(rw_eeg.mean(axis=1), 20260831),
            "heldout_meg": summary(rw_meg.mean(axis=1), 20260832),
            "equal_measurement_macro_gain": float(
                0.5 * (rw_eeg.mean() + rw_meg.mean())
            ),
            "portability_floor": float(min(rw_eeg.mean(), rw_meg.mean())),
            "geometry_preservation_min": float(rw_geometry.min()),
            "object_fold_means": {
                "eeg": rw_eeg.mean(axis=0).tolist(),
                "meg": rw_meg.mean(axis=0).tolist(),
            },
        },
        "random_image_split_stability": {
            "n_repartitions": 10,
            "split_rule": "four images per category assigned to each of three held-out folds",
            "optimization_seed": SEEDS[0],
            "mean_gain_across_repartitions": {
                "eeg_mean": float(repeat_eeg_means.mean()),
                "eeg_range": [float(repeat_eeg_means.min()), float(repeat_eeg_means.max())],
                "eeg_positive_repartitions": int(np.sum(repeat_eeg_means > 0)),
                "meg_mean": float(repeat_meg_means.mean()),
                "meg_range": [float(repeat_meg_means.min()), float(repeat_meg_means.max())],
                "meg_positive_repartitions": int(np.sum(repeat_meg_means > 0)),
            },
            "repartition_means": {
                "eeg": repeat_eeg_means.tolist(),
                "meg": repeat_meg_means.tolist(),
            },
            "participant_mean_over_repartitions": {
                "eeg": summary(repeat_eeg.mean(axis=0), 20260833),
                "meg": summary(repeat_meg.mean(axis=0), 20260834),
            },
            "fold_means": repeat_fold_means,
        },
        "categories": categories,
    }
    path = OUT / "ADAPTER_STABILITY_RESULTS_v001.json"
    np.savez_compressed(
        OUT / "ADAPTER_STABILITY_ARRAYS_v001.npz",
        reliability_weighted_eeg=rw_eeg,
        reliability_weighted_meg=rw_meg,
        reliability_weighted_geometry=rw_geometry,
        random_split_eeg=repeat_eeg,
        random_split_meg=repeat_meg,
    )
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
