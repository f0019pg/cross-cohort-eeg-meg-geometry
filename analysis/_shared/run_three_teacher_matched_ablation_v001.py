from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

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
    find_sources,
    load_eeg,
    load_mapping,
    load_meg,
    sha256,
    upper,
    zr,
    zscore,
)


OUT = ROOT_DEFAULT / "derived" / "single_measurement_ablation"
PROTOCOL = ROOT_DEFAULT / "config" / "protocols" / "single_measurement_ablation.md"
SOURCE_RESULT = ROOT_DEFAULT / "results" / "reported" / "source_adapter_gate.json"
SOURCE_ARRAYS = ROOT_DEFAULT / "source_data" / "main" / "source_participant_values.npz"
SEEDS = [20260722, 20260723, 20260724]
LAMBDA_ANCHOR = 100.0
EPOCHS = 400
TEACHERS = ("eeg_only", "meg_only", "consensus")
TEACHER_INDEX = {name: i for i, name in enumerate(TEACHERS)}


def teacher_target(
    name: str,
    eeg_group_matrix: np.ndarray,
    meg_group_matrix: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    eeg = zr(upper(eeg_group_matrix[np.ix_(idx, idx)]))
    meg = zr(upper(meg_group_matrix[np.ix_(idx, idx)]))
    if name == "eeg_only":
        target = eeg
    elif name == "meg_only":
        target = meg
    elif name == "consensus":
        target = zscore(0.5 * eeg + 0.5 * meg)
    else:
        raise ValueError(f"Unknown teacher: {name}")
    return np.asarray(target, dtype=np.float32)


def summary(values: np.ndarray, seed_offset: int) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "positive_n": int(np.sum(v > 0)),
        "n": int(v.size),
        "exact_one_sided_positive_signflip_p": exact_signflip(v, one_sided_positive=True),
        "exact_two_sided_signflip_p": exact_signflip(v),
        "bootstrap_mean_95ci": bootstrap_ci(v, seed_offset),
        "values": [float(x) for x in v],
    }


def participant_folds() -> list[dict]:
    return [
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


def check_inputs(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    if not PROTOCOL.exists() or not SOURCE_RESULT.exists() or not SOURCE_ARRAYS.exists():
        raise FileNotFoundError("Protocol or source Stage 1 evidence is absent")
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    if source_result.get("decision") != "GO_STAGE2_EXTERNAL":
        raise RuntimeError("Original consensus Stage 1 did not pass")
    mapping = root / MAPPING_NAME
    dino_path = root / DINO_NAME
    _, category, category_names = load_mapping(mapping)
    features = np.load(dino_path, mmap_mode="r")
    folds = make_object_folds(category)
    audit = {
        "status": "INPUT_CHECK_PASSED",
        "analysis_class": "prospectively locked post-hoc matched ablation",
        "source_stage1_decision": source_result["decision"],
        "eeg_dir": str(eeg_dir),
        "meg_file": str(meg_file),
        "dino_shape": list(features.shape),
        "categories": category_names,
        "teachers": list(TEACHERS),
        "participant_folds": [x["name"] for x in participant_folds()],
        "object_folds": [
            {"fold": x["fold"], "train_n": int(len(x["train"])), "test_n": int(len(x["test"]))}
            for x in folds
        ],
        "model": {
            "bottleneck": 64,
            "lambda_anchor": LAMBDA_ANCHOR,
            "epochs": EPOCHS,
            "seeds": SEEDS,
        },
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "script": sha256(Path(__file__)),
            "source_stage1_result": sha256(SOURCE_RESULT),
            "source_stage1_arrays": sha256(SOURCE_ARRAYS),
            "source_stage1_script": sha256(SOURCE / "run_late_crossmodal_adapter_gate_v001.py"),
            "mapping": sha256(mapping),
            "dino": sha256(dino_path),
        },
    }
    (OUT / "INPUT_AUDIT_v001.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def run(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    audit = check_inputs(root, eeg_dir, meg_file)
    mapping = root / MAPPING_NAME
    dino_path = root / DINO_NAME
    cichy_idx, category, _ = load_mapping(mapping)
    features = np.load(dino_path).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    eeg = load_eeg(eeg_dir)
    meg = load_meg(meg_file, cichy_idx)
    object_folds = make_object_folds(category)

    eeg_gain = np.full((len(TEACHERS), 10, 3), np.nan, dtype=np.float64)
    meg_gain = np.full((len(TEACHERS), 16, 3), np.nan, dtype=np.float64)
    geometry = np.full((len(TEACHERS), 2, 3), np.nan, dtype=np.float64)

    for p_idx, p_fold in enumerate(participant_folds()):
        eeg_group = group_matrix(eeg["mean"], p_fold["eeg_teacher"])
        meg_group = group_matrix(meg["late"]["mean"], p_fold["meg_teacher"])
        for o_fold in object_folds:
            train, test = o_fold["train"], o_fold["test"]
            base_rdm = cosine_rdm(features[test])
            q = category_control_basis(base_rdm, category[test])
            eeg_neural = [subset_vec(eeg["mean"][i], test) for i in p_fold["eeg_eval"]]
            meg_neural = [subset_vec(meg["late"]["mean"][i], test) for i in p_fold["meg_eval"]]

            for teacher in TEACHERS:
                t_idx = TEACHER_INDEX[teacher]
                target = teacher_target(teacher, eeg_group, meg_group, train)
                seed_embeddings = []
                for seed in SEEDS:
                    print(
                        f"teacher={teacher}, participant={p_fold['name']}, "
                        f"object_fold={o_fold['fold']}, seed={seed}",
                        flush=True,
                    )
                    model = fit_adapter(
                        features[train, None, :],
                        target,
                        lambda_anchor=LAMBDA_ANCHOR,
                        seed=seed,
                        epochs=EPOCHS,
                    )
                    seed_embeddings.append(
                        adapted_concept_embeddings(model, features[test, None, :])
                    )
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                ensemble = np.mean(seed_embeddings, axis=0)
                ensemble /= np.maximum(np.linalg.norm(ensemble, axis=1, keepdims=True), 1e-12)
                adapted_rdm = cosine_rdm(ensemble)
                eeg_values, _ = evaluate_rdm(adapted_rdm, base_rdm, eeg_neural, q)
                meg_values, _ = evaluate_rdm(adapted_rdm, base_rdm, meg_neural, q)
                eeg_gain[t_idx, p_fold["eeg_eval"], o_fold["fold"]] = eeg_values
                meg_gain[t_idx, p_fold["meg_eval"], o_fold["fold"]] = meg_values
                geometry[t_idx, p_idx, o_fold["fold"]] = float(
                    spearmanr(upper(adapted_rdm), upper(base_rdm)).statistic
                )

    if not np.isfinite(eeg_gain).all() or not np.isfinite(meg_gain).all() or not np.isfinite(geometry).all():
        raise RuntimeError("Non-finite or missing matched-ablation cells")

    eeg_participant = eeg_gain.mean(axis=2)
    meg_participant = meg_gain.mean(axis=2)
    teacher_summaries = {}
    object_fold_means = {}
    geometry_summaries = {}
    for teacher in TEACHERS:
        t_idx = TEACHER_INDEX[teacher]
        teacher_summaries[teacher] = {
            "eeg": summary(eeg_participant[t_idx], 400 + 10 * t_idx),
            "meg": summary(meg_participant[t_idx], 401 + 10 * t_idx),
        }
        object_fold_means[teacher] = {
            "eeg": [float(x) for x in eeg_gain[t_idx].mean(axis=0)],
            "meg": [float(x) for x in meg_gain[t_idx].mean(axis=0)],
        }
        eeg_mean = teacher_summaries[teacher]["eeg"]["mean"]
        meg_mean = teacher_summaries[teacher]["meg"]["mean"]
        teacher_summaries[teacher]["macro_equal_modality_gain"] = float(0.5 * (eeg_mean + meg_mean))
        teacher_summaries[teacher]["portability_floor"] = float(min(eeg_mean, meg_mean))
        geometry_summaries[teacher] = {
            "minimum": float(geometry[t_idx].min()),
            "mean": float(geometry[t_idx].mean()),
            "cells": geometry[t_idx].tolist(),
        }

    contrast_specs = {
        "consensus_minus_eeg_only_in_eeg": ("consensus", "eeg_only", "eeg"),
        "consensus_minus_meg_only_in_meg": ("consensus", "meg_only", "meg"),
        "consensus_minus_meg_only_in_eeg": ("consensus", "meg_only", "eeg"),
        "consensus_minus_eeg_only_in_meg": ("consensus", "eeg_only", "meg"),
        "eeg_only_minus_meg_only_in_eeg": ("eeg_only", "meg_only", "eeg"),
        "eeg_only_minus_meg_only_in_meg": ("eeg_only", "meg_only", "meg"),
    }
    contrasts = {}
    for idx, (label, (left, right, modality)) in enumerate(contrast_specs.items()):
        source_array = eeg_participant if modality == "eeg" else meg_participant
        raw_array = eeg_gain if modality == "eeg" else meg_gain
        diff = source_array[TEACHER_INDEX[left]] - source_array[TEACHER_INDEX[right]]
        contrasts[label] = summary(diff, 500 + idx)
        contrasts[label]["object_fold_means"] = [
            float(x)
            for x in (
                raw_array[TEACHER_INDEX[left]].mean(axis=0)
                - raw_array[TEACHER_INDEX[right]].mean(axis=0)
            )
        ]

    old = np.load(SOURCE_ARRAYS)
    c_idx = TEACHER_INDEX["consensus"]
    eeg_reproduction_error = float(np.max(np.abs(eeg_gain[c_idx] - old["eeg_gain"])))
    meg_reproduction_error = float(np.max(np.abs(meg_gain[c_idx] - old["meg_gain"])))
    reproduction = {
        "eeg_max_abs_error": eeg_reproduction_error,
        "meg_max_abs_error": meg_reproduction_error,
        "tolerance": 1e-6,
        "passed": bool(eeg_reproduction_error <= 1e-6 and meg_reproduction_error <= 1e-6),
    }

    p1 = contrasts["consensus_minus_eeg_only_in_eeg"]
    p2 = contrasts["consensus_minus_meg_only_in_meg"]
    p1_folds = np.asarray(p1["object_fold_means"])
    p2_folds = np.asarray(p2["object_fold_means"])
    geometry_ok = all(x["minimum"] >= 0.95 for x in geometry_summaries.values())
    finite_ok = bool(np.isfinite(eeg_gain).all() and np.isfinite(meg_gain).all() and np.isfinite(geometry).all())
    technical_validity = bool(finite_ok and reproduction["passed"] and geometry_ok)
    superiority = bool(
        technical_validity
        and p1["mean"] > 0
        and p1["positive_n"] >= 8
        and p1["exact_one_sided_positive_signflip_p"] < 0.05
        and np.all(p1_folds > 0)
        and p2["mean"] > 0
        and p2["positive_n"] >= 12
        and p2["exact_one_sided_positive_signflip_p"] < 0.05
        and np.all(p2_folds > 0)
    )
    baseline_reproduced = bool(
        teacher_summaries["consensus"]["eeg"]["mean"] > 0.005
        and teacher_summaries["consensus"]["eeg"]["positive_n"] >= 8
        and teacher_summaries["consensus"]["meg"]["mean"] > 0.005
        and teacher_summaries["consensus"]["meg"]["positive_n"] >= 12
    )
    consensus_macro = teacher_summaries["consensus"]["macro_equal_modality_gain"]
    consensus_floor = teacher_summaries["consensus"]["portability_floor"]
    single_at_least_consensus = any(
        teacher_summaries[name]["macro_equal_modality_gain"] >= consensus_macro
        or teacher_summaries[name]["portability_floor"] >= consensus_floor
        for name in ("eeg_only", "meg_only")
    )
    single_sufficient_or_preferred = bool(
        not superiority
        and (p1["mean"] <= 0 or p2["mean"] <= 0 or single_at_least_consensus)
    )

    if not technical_validity:
        decision = "TECHNICAL_FAILURE"
    elif superiority:
        decision = "CONSENSUS_MATCHED_SUPERIORITY"
    elif single_sufficient_or_preferred:
        decision = "SINGLE_TEACHER_SUFFICIENT_OR_PREFERRED"
    else:
        decision = "CONSENSUS_PORTABLE_NOT_MATCHED_SUPERIOR"

    gates = {
        "technical_validity": technical_validity,
        "consensus_baseline_reproduced": baseline_reproduced,
        "P1_eeg_matched_superiority": bool(
            p1["mean"] > 0
            and p1["positive_n"] >= 8
            and p1["exact_one_sided_positive_signflip_p"] < 0.05
            and np.all(p1_folds > 0)
        ),
        "P2_meg_matched_superiority": bool(
            p2["mean"] > 0
            and p2["positive_n"] >= 12
            and p2["exact_one_sided_positive_signflip_p"] < 0.05
            and np.all(p2_folds > 0)
        ),
        "full_consensus_matched_superiority": superiority,
    }

    result = {
        "analysis": "late EEG-MEG three-teacher matched ablation",
        "analysis_class": "prospectively locked post-hoc sensitivity analysis",
        "decision": decision,
        "input_audit": audit,
        "teacher_summaries": teacher_summaries,
        "object_fold_means": object_fold_means,
        "contrasts": contrasts,
        "geometry_preservation": geometry_summaries,
        "original_consensus_reproduction": reproduction,
        "gates": gates,
        "interpretation_flags": {
            "single_teacher_at_least_consensus_on_macro_or_floor": bool(single_at_least_consensus),
            "matched_superiority_supported": superiority,
            "not_independent_confirmation": True,
        },
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "script": sha256(Path(__file__)),
            "source_stage1_result": sha256(SOURCE_RESULT),
            "source_stage1_arrays": sha256(SOURCE_ARRAYS),
            "dino": sha256(dino_path),
        },
    }
    np.savez_compressed(
        OUT / "RESULT_ARRAYS_v001.npz",
        teacher_names=np.asarray(TEACHERS),
        eeg_gain=eeg_gain,
        meg_gain=meg_gain,
        geometry=geometry,
    )
    (OUT / "FINAL_RESULTS_v001.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--eeg-dir", type=str)
    parser.add_argument("--meg-file", type=str)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-inputs", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    eeg_dir, meg_file = find_sources(root, args.eeg_dir, args.meg_file)
    if args.check_inputs:
        print(json.dumps(check_inputs(root, eeg_dir, meg_file), indent=2, ensure_ascii=False))
    else:
        run(root, eeg_dir, meg_file)


if __name__ == "__main__":
    main()
