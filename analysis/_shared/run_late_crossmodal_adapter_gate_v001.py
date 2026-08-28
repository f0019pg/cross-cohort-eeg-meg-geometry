from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from common import adapted_concept_embeddings, fit_adapter
from run_late_crossmodal_source_gate_v001 import (
    DINO_NAME,
    MAPPING_NAME,
    corr,
    find_sources,
    generate_within_category_permutation,
    load_eeg,
    load_mapping,
    load_meg,
    residual_rank,
    sha256,
    summarize,
    upper,
    zscore,
    zr,
)


OUT = ROOT_DEFAULT / "derived" / "adapter_gate"
AMENDMENT = OUT / "AMENDMENT_01_ADAPTER_GATE_KR.md"
STAGE0 = OUT / "STAGE0_RESULTS_v001.json"
SEEDS = [20260722, 20260723, 20260724]
SHUFFLE_SEED = 20260806
N_SHUFFLE = 39
LAMBDA_ANCHOR = 100.0
EPOCHS = 400


def cosine_rdm(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    d = 1.0 - x @ x.T
    np.fill_diagonal(d, 0.0)
    return d


def vec_to_matrix(v: np.ndarray, n: int = 72) -> np.ndarray:
    iu = np.triu_indices(n, 1)
    m = np.zeros((n, n), dtype=np.float64)
    m[iu] = v
    m[(iu[1], iu[0])] = v
    return m


def subset_vec(full_vector: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return upper(vec_to_matrix(full_vector)[np.ix_(idx, idx)])


def category_control_basis(base_rdm: np.ndarray, category: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(category), 1)
    a = np.minimum(category[iu[0]], category[iu[1]])
    b = np.maximum(category[iu[0]], category[iu[1]])
    codes = a * 6 + b
    levels = sorted(np.unique(codes).tolist())
    dummies = np.column_stack([(codes == level).astype(float) for level in levels[1:]])
    x = np.column_stack([np.ones(len(iu[0])), zr(upper(base_rdm)), dummies])
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    tol = np.finfo(float).eps * max(x.shape) * s[0]
    return u[:, s > tol]


def make_object_folds(category: np.ndarray) -> list[dict]:
    folds = []
    for fold in range(3):
        test = []
        for cat in range(6):
            idx = np.flatnonzero(category == cat)
            test.extend(idx[np.arange(12) % 3 == fold].tolist())
        test = np.asarray(sorted(test), dtype=int)
        train = np.asarray([i for i in range(72) if i not in set(test.tolist())], dtype=int)
        if len(train) != 48 or len(test) != 24:
            raise RuntimeError("Invalid object fold")
        if not np.all(np.bincount(category[train], minlength=6) == 8):
            raise RuntimeError("Unbalanced training fold")
        if not np.all(np.bincount(category[test], minlength=6) == 4):
            raise RuntimeError("Unbalanced evaluation fold")
        folds.append({"fold": fold, "train": train, "test": test})
    return folds


def consensus_target(eeg_group_matrix: np.ndarray, meg_group_matrix: np.ndarray, idx: np.ndarray) -> np.ndarray:
    eeg = zr(upper(eeg_group_matrix[np.ix_(idx, idx)]))
    meg = zr(upper(meg_group_matrix[np.ix_(idx, idx)]))
    return zscore(0.5 * eeg + 0.5 * meg).astype(np.float32)


def evaluate_rdm(
    adapted_rdm: np.ndarray,
    base_rdm: np.ndarray,
    neural_vectors: list[np.ndarray],
    q: np.ndarray,
) -> tuple[list[float], list[float]]:
    adapted = upper(adapted_rdm)
    base = upper(base_rdm)
    delta = zscore(rankdata(adapted, method="average") - rankdata(base, method="average"))
    gains, unique = [], []
    for neural in neural_vectors:
        gains.append(float(spearmanr(adapted, neural).statistic - spearmanr(base, neural).statistic))
        unique.append(corr(delta, residual_rank(neural, q)))
    return gains, unique


def group_matrix(vectors: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return vec_to_matrix(zscore(np.mean(vectors[indices], axis=0)))


def check_inputs(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    if not STAGE0.exists():
        raise FileNotFoundError("Stage 0 results are absent")
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    if stage0.get("decision") != "GO_STAGE1_ADAPTER" or not all(stage0.get("gates", {}).values()):
        raise RuntimeError("Stage 0 did not authorize adapter analysis")
    mapping = root / MAPPING_NAME
    dino_path = root / DINO_NAME
    _, category, category_names = load_mapping(mapping)
    features = np.load(dino_path, mmap_mode="r")
    object_folds = make_object_folds(category)
    audit = {
        "status": "STAGE1_INPUT_CHECK_PASSED",
        "stage0_decision": stage0["decision"],
        "eeg_dir": str(eeg_dir),
        "meg_file": str(meg_file),
        "dino_shape": list(features.shape),
        "categories": category_names,
        "participant_folds": {
            "A": "odd EEG/MEG teacher, even EEG/MEG evaluation",
            "B": "even EEG/MEG teacher, odd EEG/MEG evaluation",
        },
        "object_folds": [{"fold": x["fold"], "train_n": len(x["train"]), "test_n": len(x["test"])} for x in object_folds],
        "model": {"bottleneck": 64, "lambda_anchor": LAMBDA_ANCHOR, "epochs": EPOCHS, "seeds": SEEDS},
        "teacher_shuffles": {"n": N_SHUFFLE, "seed": SHUFFLE_SEED},
        "hashes": {
            "stage0": sha256(STAGE0),
            "amendment": sha256(AMENDMENT),
            "script": sha256(Path(__file__)),
            "mapping": sha256(mapping),
            "dino": sha256(dino_path),
        },
    }
    (OUT / "STAGE1_INPUT_AUDIT_v001.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def run_stage1(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    audit = check_inputs(root, eeg_dir, meg_file)
    mapping = root / MAPPING_NAME
    dino_path = root / DINO_NAME
    cichy_idx, category, category_names = load_mapping(mapping)
    features = np.load(dino_path).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    eeg = load_eeg(eeg_dir)
    meg = load_meg(meg_file, cichy_idx)
    object_folds = make_object_folds(category)

    participant_folds = [
        {
            "name": "A_odd_teacher_even_evaluation",
            "eeg_teacher": np.arange(0, 10, 2),
            "eeg_eval": np.arange(1, 10, 2),
            "meg_teacher": np.arange(0, 16, 2),
            "meg_eval": np.arange(1, 16, 2),
        },
        {
            "name": "B_even_teacher_odd_evaluation",
            "eeg_teacher": np.arange(1, 10, 2),
            "eeg_eval": np.arange(0, 10, 2),
            "meg_teacher": np.arange(1, 16, 2),
            "meg_eval": np.arange(0, 16, 2),
        },
    ]

    eeg_gain = np.full((10, 3), np.nan)
    meg_gain = np.full((16, 3), np.nan)
    eeg_unique = np.full((10, 3), np.nan)
    meg_unique = np.full((16, 3), np.nan)
    eeg_gain_single = np.full((10, 3), np.nan)
    meg_gain_single = np.full((16, 3), np.nan)
    geometry = []
    configurations = []

    for p_fold in participant_folds:
        eeg_group = group_matrix(eeg["mean"], p_fold["eeg_teacher"])
        meg_group = group_matrix(meg["late"]["mean"], p_fold["meg_teacher"])
        for o_fold in object_folds:
            train, test = o_fold["train"], o_fold["test"]
            target = consensus_target(eeg_group, meg_group, train)
            base_rdm = cosine_rdm(features[test])
            q = category_control_basis(base_rdm, category[test])
            eeg_neural = [subset_vec(eeg["mean"][i], test) for i in p_fold["eeg_eval"]]
            meg_neural = [subset_vec(meg["late"]["mean"][i], test) for i in p_fold["meg_eval"]]

            seed_embeddings = []
            for seed in SEEDS:
                print(f"observed adapter: {p_fold['name']}, object fold {o_fold['fold']}, seed {seed}", flush=True)
                model = fit_adapter(
                    features[train, None, :], target, lambda_anchor=LAMBDA_ANCHOR,
                    seed=seed, epochs=EPOCHS,
                )
                embedding = adapted_concept_embeddings(model, features[test, None, :])
                seed_embeddings.append(embedding)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            ensemble = np.mean(seed_embeddings, axis=0)
            ensemble /= np.maximum(np.linalg.norm(ensemble, axis=1, keepdims=True), 1e-12)
            adapted_rdm = cosine_rdm(ensemble)
            single_rdm = cosine_rdm(seed_embeddings[0])
            eeg_g, eeg_u = evaluate_rdm(adapted_rdm, base_rdm, eeg_neural, q)
            meg_g, meg_u = evaluate_rdm(adapted_rdm, base_rdm, meg_neural, q)
            eeg_s, _ = evaluate_rdm(single_rdm, base_rdm, eeg_neural, q)
            meg_s, _ = evaluate_rdm(single_rdm, base_rdm, meg_neural, q)
            eeg_gain[p_fold["eeg_eval"], o_fold["fold"]] = eeg_g
            meg_gain[p_fold["meg_eval"], o_fold["fold"]] = meg_g
            eeg_unique[p_fold["eeg_eval"], o_fold["fold"]] = eeg_u
            meg_unique[p_fold["meg_eval"], o_fold["fold"]] = meg_u
            eeg_gain_single[p_fold["eeg_eval"], o_fold["fold"]] = eeg_s
            meg_gain_single[p_fold["meg_eval"], o_fold["fold"]] = meg_s
            preservation = float(spearmanr(upper(adapted_rdm), upper(base_rdm)).statistic)
            geometry.append({"participant_fold": p_fold["name"], "object_fold": o_fold["fold"], "rho": preservation})
            configurations.append({
                "participant_fold": p_fold,
                "object_fold": o_fold,
                "eeg_group": eeg_group,
                "meg_group": meg_group,
                "base_rdm": base_rdm,
                "eeg_neural": eeg_neural,
                "meg_neural": meg_neural,
            })

    if any(np.isnan(x).any() for x in (eeg_gain, meg_gain, eeg_unique, meg_unique, eeg_gain_single, meg_gain_single)):
        raise RuntimeError("Stage 1 cross-validation left unevaluated cells")

    eeg_participant_gain = eeg_gain.mean(axis=1)
    meg_participant_gain = meg_gain.mean(axis=1)
    eeg_participant_unique = eeg_unique.mean(axis=1)
    meg_participant_unique = meg_unique.mean(axis=1)
    object_fold_gain = [
        {"fold": fold, "eeg_mean_gain": float(eeg_gain[:, fold].mean()), "meg_mean_gain": float(meg_gain[:, fold].mean())}
        for fold in range(3)
    ]
    observed_single = 0.5 * (eeg_gain_single.mean() + meg_gain_single.mean())

    rng = np.random.default_rng(SHUFFLE_SEED)
    null = np.empty(N_SHUFFLE, dtype=np.float64)
    for shuffle in range(N_SHUFFLE):
        perm = generate_within_category_permutation(rng, category)
        eeg_values, meg_values = [], []
        for config in configurations:
            train = config["object_fold"]["train"]
            test = config["object_fold"]["test"]
            eeg_perm = config["eeg_group"][np.ix_(perm, perm)]
            meg_perm = config["meg_group"][np.ix_(perm, perm)]
            target = consensus_target(eeg_perm, meg_perm, train)
            model = fit_adapter(
                features[train, None, :], target, lambda_anchor=LAMBDA_ANCHOR,
                seed=SEEDS[0], epochs=EPOCHS,
            )
            embedding = adapted_concept_embeddings(model, features[test, None, :])
            adapted = cosine_rdm(embedding)
            base = config["base_rdm"]
            for neural in config["eeg_neural"]:
                eeg_values.append(float(spearmanr(upper(adapted), neural).statistic - spearmanr(upper(base), neural).statistic))
            for neural in config["meg_neural"]:
                meg_values.append(float(spearmanr(upper(adapted), neural).statistic - spearmanr(upper(base), neural).statistic))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        null[shuffle] = 0.5 * (np.mean(eeg_values) + np.mean(meg_values))
        print(f"teacher shuffle {shuffle + 1}/{N_SHUFFLE}: {null[shuffle]:+.6f}", flush=True)

    shuffle_p = float((1 + np.sum(null >= observed_single)) / (N_SHUFFLE + 1))
    a1 = summarize(eeg_participant_gain, 101)
    a2 = summarize(meg_participant_gain, 102)
    a3_eeg = summarize(eeg_participant_unique, 103)
    a3_meg = summarize(meg_participant_unique, 104)
    min_geometry = float(min(x["rho"] for x in geometry))
    gates = {
        "A1_eeg_alignment_gain": bool(a1["mean"] > 0.005 and a1["positive_n"] >= 8 and a1["exact_two_sided_signflip_p"] < 0.05),
        "A2_meg_alignment_gain": bool(a2["mean"] > 0.005 and a2["positive_n"] >= 12 and a2["exact_two_sided_signflip_p"] < 0.05),
        "A3_unique_displacement": bool(a3_eeg["mean"] > 0.02 and a3_eeg["positive_n"] >= 8 and a3_eeg["exact_two_sided_signflip_p"] < 0.05 and a3_meg["mean"] > 0.02 and a3_meg["positive_n"] >= 12 and a3_meg["exact_two_sided_signflip_p"] < 0.05),
        "A4_all_object_folds_positive": bool(all(x["eeg_mean_gain"] > 0 and x["meg_mean_gain"] > 0 for x in object_fold_gain)),
        "A5_geometry_preservation": bool(min_geometry >= 0.95),
        "A6_teacher_specificity": bool(shuffle_p < 0.05),
    }
    decision = "GO_STAGE2_EXTERNAL" if all(gates.values()) else "STOP_OR_LIMITED_ADAPTER"
    result = {
        "analysis": "late EEG-MEG consensus adapter held-out source gate",
        "decision": decision,
        "input_audit": audit,
        "eeg_alignment_gain": a1,
        "meg_alignment_gain": a2,
        "unique_displacement": {"eeg": a3_eeg, "meg": a3_meg},
        "object_fold_gain": object_fold_gain,
        "geometry_preservation": {"minimum": min_geometry, "cells": geometry},
        "teacher_specificity": {
            "observed_single_seed_equal_modality_gain": float(observed_single),
            "n": N_SHUFFLE,
            "seed": SHUFFLE_SEED,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "null_95th": float(np.quantile(null, 0.95)),
            "one_sided_p": shuffle_p,
        },
        "gates": gates,
        "hashes": {
            "amendment": sha256(AMENDMENT),
            "script": sha256(Path(__file__)),
            "stage0": sha256(STAGE0),
            "dino": sha256(dino_path),
        },
    }
    np.savez_compressed(
        OUT / "STAGE1_PARTICIPANT_VALUES_v001.npz",
        eeg_gain=eeg_gain,
        meg_gain=meg_gain,
        eeg_unique=eeg_unique,
        meg_unique=meg_unique,
        eeg_gain_single=eeg_gain_single,
        meg_gain_single=meg_gain_single,
        teacher_shuffle_null=null,
    )
    (OUT / "STAGE1_RESULTS_v001.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "eeg_alignment_gain": a1,
        "meg_alignment_gain": a2,
        "unique_displacement": {"eeg": a3_eeg, "meg": a3_meg},
        "object_fold_gain": object_fold_gain,
        "minimum_geometry_preservation": min_geometry,
        "teacher_specificity": result["teacher_specificity"],
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
    action.add_argument("--run-stage1", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    eeg_dir, meg_file = find_sources(root, args.eeg_dir, args.meg_file)
    if args.check_inputs:
        print(json.dumps(check_inputs(root, eeg_dir, meg_file), indent=2, ensure_ascii=False))
    else:
        run_stage1(root, eeg_dir, meg_file)


if __name__ == "__main__":
    main()
