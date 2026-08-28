from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
SHARED = ROOT_DEFAULT / "analysis" / "_shared"
sys.path.insert(0, str(SHARED))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

from common import (
    ALLJOINED_DIR,
    B_SLOTS,
    THINGS_DIR,
    adapted_concept_embeddings,
    cosine_rdm_matrix,
    fit_adapter,
    load_dino,
    load_patterns,
    load_spose,
    load_table,
    participant_rdms,
    split_indices,
)
from run_late_crossmodal_source_gate_v001 import (
    DINO_NAME,
    MAPPING_NAME,
    exact_signflip,
    find_sources,
    load_eeg,
    load_mapping,
    load_meg,
    sha256,
    summarize,
    upper,
    zscore,
    zr,
)


OUT = ROOT_DEFAULT / "derived" / "external_transfer"
AMENDMENT = ROOT_DEFAULT / "config" / "protocols" / "external_transfer.md"
STAGE1 = ROOT_DEFAULT / "results" / "reported" / "source_adapter_gate.json"
NOD_EXT_REL = Path("data/nod_external")
NOD_CACHE_REL = Path("data/nod_eeg_rdm_cache.npz")
SEEDS = [20260722, 20260723, 20260724]
LAMBDA_ANCHOR = 100.0
EPOCHS = 400
N_BEHAV_PERM = 9_999
BEHAV_SEED = 20260806


def cosine_upper_torch(features: np.ndarray) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    x = F.normalize(x, dim=-1)
    d = 1.0 - x @ x.T
    iu = torch.triu_indices(len(x), len(x), offset=1, device=device)
    return d[iu[0], iu[1]].cpu().numpy().astype(np.float64)


def grouped_mean(x: np.ndarray, inverse: np.ndarray, nclass: int = 1000) -> np.ndarray:
    out = np.zeros((nclass, x.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse, minlength=nclass).astype(np.float64)
    np.add.at(out, inverse, x)
    if np.any(counts == 0):
        raise RuntimeError("A NOD class has no image trials")
    out /= counts[:, None]
    return out.astype(np.float32)


@torch.no_grad()
def adapt_flat_ensemble(models: list, features: np.ndarray, batch: int = 4096) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_models = [model.to(device).eval() for model in models]
    rows = []
    for start in range(0, len(features), batch):
        x = torch.as_tensor(np.asarray(features[start : start + batch]), dtype=torch.float32, device=device)
        z = torch.stack([model(x) for model in gpu_models], dim=0).mean(dim=0)
        rows.append(F.normalize(z, dim=-1).cpu().numpy())
    for model in gpu_models:
        model.cpu()
    return np.concatenate(rows).astype(np.float32)


def source_consensus(eeg_vectors: np.ndarray, meg_vectors: np.ndarray) -> np.ndarray:
    eeg = zr(np.mean(eeg_vectors, axis=0))
    meg = zr(np.mean(meg_vectors, axis=0))
    return zscore(0.5 * eeg + 0.5 * meg).astype(np.float32)


def subset_upper(matrix: np.ndarray, idx: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix)[np.ix_(idx, idx)]
    return np.asarray(m[np.triu_indices(len(idx), 1)], dtype=np.float64)


def participant_gains(model_vec: np.ndarray, base_vec: np.ndarray, rdms: np.ndarray, idx: np.ndarray) -> np.ndarray:
    values = []
    for rdm in rdms:
        target = subset_upper(rdm, idx)
        values.append(float(spearmanr(model_vec, target).statistic - spearmanr(base_vec, target).statistic))
    return np.asarray(values)


def behavior_permutation(
    model_vec: np.ndarray,
    base_vec: np.ndarray,
    target_matrix: np.ndarray,
    observed_gain: float,
) -> dict:
    rng = np.random.default_rng(BEHAV_SEED)
    iu = np.triu_indices(len(target_matrix), 1)
    model_z = zr(model_vec)
    base_z = zr(base_vec)
    null = np.empty(N_BEHAV_PERM, dtype=np.float64)
    for p in range(N_BEHAV_PERM):
        perm = rng.permutation(len(target_matrix))
        target = zr(target_matrix[np.ix_(perm, perm)][iu])
        null[p] = float(np.mean(model_z * target) - np.mean(base_z * target))
    pvalue = float((1 + np.sum(null >= observed_gain)) / (N_BEHAV_PERM + 1))
    return {
        "n": N_BEHAV_PERM,
        "seed": BEHAV_SEED,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "null_95th": float(np.quantile(null, 0.95)),
        "one_sided_p": pvalue,
    }


def check_inputs(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    if not STAGE1.exists():
        raise FileNotFoundError("Stage 1 results are absent")
    stage1 = json.loads(STAGE1.read_text(encoding="utf-8"))
    if stage1.get("decision") != "GO_STAGE2_EXTERNAL" or not all(stage1.get("gates", {}).values()):
        raise RuntimeError("Stage 1 did not authorize external evaluation")
    nod_ext = root / NOD_EXT_REL
    nod_cache_path = root / NOD_CACHE_REL
    required = [
        root / DINO_NAME,
        root / MAPPING_NAME,
        nod_ext / "NOD_PARTICIPANT_TRIAL_INDEX.csv",
        nod_ext / "nod_actual_dinov3_384d.npy",
        nod_ext / "NOD_FEATURE_AUDIT.json",
        nod_cache_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 2 inputs:\n" + "\n".join(missing))
    df = load_table()
    _, conf, _ = split_indices(df)
    dino = load_dino()
    things_files = sorted(THINGS_DIR.glob("sub-*_patterns.npz"))
    alljoined_files = sorted(ALLJOINED_DIR.glob("sub-*_patterns.npz"))
    nod_features = np.load(nod_ext / "nod_actual_dinov3_384d.npy", mmap_mode="r")
    nod_index = pd.read_csv(nod_ext / "NOD_PARTICIPANT_TRIAL_INDEX.csv", usecols=["subject", "feature_row"])
    nod_cache = np.load(nod_cache_path, mmap_mode="r", allow_pickle=False)
    audit = {
        "status": "STAGE2_INPUT_CHECK_PASSED",
        "stage1_decision": stage1["decision"],
        "confirmation_concepts": int(len(conf)),
        "dino_shape": list(dino.shape),
        "things_participants": len(things_files),
        "alljoined_participants": len(alljoined_files),
        "nod_feature_shape": list(nod_features.shape),
        "nod_trial_rows": int(len(nod_index)),
        "nod_subjects": int(nod_index["subject"].nunique()),
        "nod_cache_post_shape": list(nod_cache["post"].shape),
        "nod_cache_pre_shape": list(nod_cache["pre"].shape),
        "source_models": {"seeds": SEEDS, "epochs": EPOCHS, "lambda_anchor": LAMBDA_ANCHOR},
        "hashes": {
            "stage1": sha256(STAGE1),
            "amendment": sha256(AMENDMENT),
            "script": sha256(Path(__file__)),
            "nod_feature_audit": sha256(nod_ext / "NOD_FEATURE_AUDIT.json"),
            "nod_cache": sha256(nod_cache_path),
        },
    }
    expected = {
        "confirmation_concepts": 183,
        "things_participants": 8,
        "alljoined_participants": 20,
        "nod_trial_rows": 56000,
        "nod_subjects": 19,
    }
    for key, value in expected.items():
        if audit[key] != value:
            raise RuntimeError(f"Unexpected {key}: {audit[key]} != {value}")
    if tuple(dino.shape) != (884, 10, 384):
        raise RuntimeError(f"Unexpected THINGS DINO shape: {dino.shape}")
    if tuple(nod_features.shape) != (55761, 384):
        raise RuntimeError(f"Unexpected NOD feature shape: {nod_features.shape}")
    if tuple(nod_cache["post"].shape) != (19, 499500) or tuple(nod_cache["pre"].shape) != (19, 499500):
        raise RuntimeError("Unexpected NOD EEG cache shape")
    (OUT / "STAGE2_INPUT_AUDIT_v001.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def run_stage2(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    audit = check_inputs(root, eeg_dir, meg_file)
    cichy_idx, _, _ = load_mapping(root / MAPPING_NAME)
    source_features = np.load(root / DINO_NAME).astype(np.float32)
    source_features /= np.maximum(np.linalg.norm(source_features, axis=1, keepdims=True), 1e-12)
    eeg_source = load_eeg(eeg_dir)
    meg_source = load_meg(meg_file, cichy_idx)
    target = source_consensus(eeg_source["mean"], meg_source["late"]["mean"])

    models = []
    checkpoint_dir = OUT / "stage2_final_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        print(f"training final late-consensus adapter, seed={seed}", flush=True)
        model = fit_adapter(
            source_features[:, None, :], target,
            lambda_anchor=LAMBDA_ANCHOR, seed=seed, epochs=EPOCHS,
        )
        torch.save({"seed": seed, "state_dict": model.state_dict()}, checkpoint_dir / f"late_consensus_seed_{seed}.pt")
        models.append(model)

    df = load_table()
    _, conf, _ = split_indices(df)
    dino = load_dino()
    things, _ = load_patterns(THINGS_DIR)
    alljoined, _ = load_patterns(ALLJOINED_DIR)
    things_rdms = participant_rdms(things, 1)
    alljoined_rdms = participant_rdms(alljoined, 1)
    test_features = dino[conf][:, B_SLOTS]
    base_embeddings = test_features.mean(axis=1)
    base_embeddings /= np.maximum(np.linalg.norm(base_embeddings, axis=1, keepdims=True), 1e-12)
    seed_embeddings = [adapted_concept_embeddings(model, test_features) for model in models]
    adapted_embeddings = np.mean(seed_embeddings, axis=0)
    adapted_embeddings /= np.maximum(np.linalg.norm(adapted_embeddings, axis=1, keepdims=True), 1e-12)
    base_rdm = cosine_rdm_matrix(base_embeddings)
    adapted_rdm = cosine_rdm_matrix(adapted_embeddings)
    base_vec = upper(base_rdm)
    adapted_vec = upper(adapted_rdm)

    things_gain = participant_gains(adapted_vec, base_vec, things_rdms, conf)
    alljoined_gain = participant_gains(adapted_vec, base_vec, alljoined_rdms, conf)
    things_summary = summarize(things_gain, 201)
    alljoined_summary = summarize(alljoined_gain, 202)
    things_geometry = float(spearmanr(base_vec, adapted_vec).statistic)

    spose = load_spose(df)[conf]
    spose_rdm = cosine_rdm_matrix(spose)
    spose_vec = upper(spose_rdm)
    behavior_base = float(spearmanr(base_vec, spose_vec).statistic)
    behavior_adapted = float(spearmanr(adapted_vec, spose_vec).statistic)
    behavior_gain = behavior_adapted - behavior_base
    behavior_perm = behavior_permutation(adapted_vec, base_vec, spose_rdm, behavior_gain)

    nod_ext = root / NOD_EXT_REL
    nod_cache = np.load(root / NOD_CACHE_REL, allow_pickle=False)
    nod_subjects = nod_cache["subjects"].astype(str).tolist()
    nod_post = np.asarray(nod_cache["post"])
    nod_pre = np.asarray(nod_cache["pre"])
    nod_index = pd.read_csv(
        nod_ext / "NOD_PARTICIPANT_TRIAL_INDEX.csv",
        dtype={"image_id": str, "class_id": str},
    )
    nod_features = np.load(nod_ext / "nod_actual_dinov3_384d.npy", mmap_mode="r")
    print("adapting NOD image features with the frozen source ensemble", flush=True)
    nod_adapted_features = adapt_flat_ensemble(models, nod_features)
    nod_post_gain, nod_pre_gain, nod_geometry = [], [], []
    nod_rows = []
    for s, subject in enumerate(nod_subjects):
        rows = nod_index[nod_index["subject"] == subject].sort_values("trial").reset_index(drop=True)
        class_ids = np.sort(rows["class_id"].astype(str).unique())
        if len(class_ids) != 1000:
            raise RuntimeError(f"Expected 1000 NOD classes for {subject}")
        cmap = {c: i for i, c in enumerate(class_ids)}
        inverse = rows["class_id"].astype(str).map(cmap).to_numpy()
        feature_rows = rows["feature_row"].astype(int).to_numpy()
        base_class = grouped_mean(np.asarray(nod_features[feature_rows]), inverse)
        adapted_class = grouped_mean(nod_adapted_features[feature_rows], inverse)
        base = cosine_upper_torch(base_class)
        adapted = cosine_upper_torch(adapted_class)
        base_post = float(spearmanr(base, nod_post[s]).statistic)
        adapted_post = float(spearmanr(adapted, nod_post[s]).statistic)
        base_pre = float(spearmanr(base, nod_pre[s]).statistic)
        adapted_pre = float(spearmanr(adapted, nod_pre[s]).statistic)
        post_gain = adapted_post - base_post
        pre_gain = adapted_pre - base_pre
        preservation = float(spearmanr(base, adapted).statistic)
        nod_post_gain.append(post_gain)
        nod_pre_gain.append(pre_gain)
        nod_geometry.append(preservation)
        nod_rows.append({
            "subject": subject,
            "post_gain": post_gain,
            "prestimulus_gain": pre_gain,
            "geometry_preservation": preservation,
            "base_post_rho": base_post,
            "adapted_post_rho": adapted_post,
        })
        print(f"NOD {subject}: post gain={post_gain:+.6f}, pre gain={pre_gain:+.6f}, geometry={preservation:.5f}", flush=True)

    nod_post_gain = np.asarray(nod_post_gain)
    nod_pre_gain = np.asarray(nod_pre_gain)
    nod_difference = nod_post_gain - nod_pre_gain
    nod_post_summary = summarize(nod_post_gain, 203)
    nod_pre_summary = summarize(nod_pre_gain, 204)
    nod_pre_summary["exact_one_sided_positive_signflip_p"] = exact_signflip(nod_pre_gain, one_sided_positive=True)
    nod_diff_summary = summarize(nod_difference, 205)
    nod_geometry_mean = float(np.mean(nod_geometry))

    gates = {
        "E1_things_eeg": bool(things_summary["mean"] >= 0.005 and things_summary["positive_n"] == 8 and things_summary["exact_two_sided_signflip_p"] < 0.05),
        "E2_alljoined_eeg": bool(alljoined_summary["mean"] >= 0.005 and alljoined_summary["positive_n"] >= 15 and alljoined_summary["exact_two_sided_signflip_p"] < 0.05),
        "E3_nod_poststimulus": bool(nod_post_summary["mean"] >= 0.005 and nod_post_summary["positive_n"] >= 15 and nod_post_summary["exact_two_sided_signflip_p"] < 0.05),
        "E4_nod_temporal_specificity": bool(abs(nod_pre_summary["mean"]) < 0.005 and nod_pre_summary["exact_one_sided_positive_signflip_p"] >= 0.05 and nod_diff_summary["mean"] > 0.005 and nod_diff_summary["positive_n"] >= 15 and nod_diff_summary["exact_two_sided_signflip_p"] < 0.05),
        "E5_geometry_preservation": bool(things_geometry >= 0.95 and nod_geometry_mean >= 0.95),
    }
    decision = "EXTERNAL_TRANSFER_CANDIDATE" if all(gates.values()) else "SOURCE_SUPPORTED_EXTERNAL_LIMITED"
    result = {
        "analysis": "late EEG-MEG consensus adapter external transfer",
        "decision": decision,
        "input_audit": audit,
        "things_eeg": things_summary,
        "alljoined_eeg": alljoined_summary,
        "human_similarity": {
            "frozen_rho": behavior_base,
            "adapted_rho": behavior_adapted,
            "gain": behavior_gain,
            "label_permutation": behavior_perm,
            "status": "convergent descriptive endpoint, not an external neural gate",
        },
        "nod": {
            "poststimulus": nod_post_summary,
            "prestimulus": nod_pre_summary,
            "post_minus_prestimulus": nod_diff_summary,
            "geometry_preservation_mean": nod_geometry_mean,
            "participants": nod_rows,
        },
        "things_geometry_preservation": things_geometry,
        "gates": gates,
        "hashes": {
            "amendment": sha256(AMENDMENT),
            "script": sha256(Path(__file__)),
            "stage1": sha256(STAGE1),
            "nod_cache": sha256(root / NOD_CACHE_REL),
        },
    }
    (OUT / "STAGE2_RESULTS_v001.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(nod_rows).to_csv(OUT / "STAGE2_NOD_PARTICIPANTS_v001.csv", index=False)
    print(json.dumps({
        "decision": decision,
        "things_eeg": things_summary,
        "alljoined_eeg": alljoined_summary,
        "human_similarity": result["human_similarity"],
        "nod": result["nod"],
        "things_geometry_preservation": things_geometry,
        "gates": gates,
    }, indent=2), flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--eeg-dir", type=str)
    parser.add_argument("--meg-file", type=str)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-inputs", action="store_true")
    action.add_argument("--run-stage2", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    eeg_dir, meg_file = find_sources(root, args.eeg_dir, args.meg_file)
    if args.check_inputs:
        print(json.dumps(check_inputs(root, eeg_dir, meg_file), indent=2, ensure_ascii=False))
    else:
        run_stage2(root, eeg_dir, meg_file)


if __name__ == "__main__":
    main()
