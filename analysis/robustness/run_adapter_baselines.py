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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

from common import adapted_concept_embeddings, fit_adapter, set_seed, torch_upper_cosine, pearson_loss
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
    residual_rank,
    upper,
    zr,
    zscore,
)


OUT = ROOT / "derived" / "adapter_baselines"
EEG_DIR = Path(os.environ.get("KANESHIRO_EEG_DIR", ROOT / "data" / "kaneshiro_eeg"))
MEG_FILE = Path(os.environ.get("CICHY_MEG_FILE", ROOT / "data" / "cichy_meg_rdms.mat"))
SEEDS = (20260722, 20260723, 20260724)
TARGETS = ("raw_consensus", "controlled_residual", "category_only", "dino_self")


def participant_folds() -> list[dict]:
    return [
        {"eeg_teacher": np.arange(0, 10, 2), "eeg_eval": np.arange(1, 10, 2),
         "meg_teacher": np.arange(0, 16, 2), "meg_eval": np.arange(1, 16, 2)},
        {"eeg_teacher": np.arange(1, 10, 2), "eeg_eval": np.arange(0, 10, 2),
         "meg_teacher": np.arange(1, 16, 2), "meg_eval": np.arange(0, 16, 2)},
    ]


def category_pair_codes(category: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(category), 1)
    a = np.minimum(category[iu[0]], category[iu[1]])
    b = np.maximum(category[iu[0]], category[iu[1]])
    return a * 6 + b


def make_target(
    name: str,
    eeg_group: np.ndarray,
    meg_group: np.ndarray,
    features: np.ndarray,
    category: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    eeg = zr(upper(eeg_group[np.ix_(idx, idx)]))
    meg = zr(upper(meg_group[np.ix_(idx, idx)]))
    consensus = zscore(0.5 * eeg + 0.5 * meg)
    base = cosine_rdm(features[idx])
    if name == "raw_consensus":
        y = consensus
    elif name == "controlled_residual":
        q = category_control_basis(base, category[idx])
        # The same DINO + complete category-pair design used in the source RSA.
        y = residual_rank(consensus, q)
    elif name == "category_only":
        codes = category_pair_codes(category[idx])
        y = np.empty_like(consensus)
        for code in np.unique(codes):
            mask = codes == code
            y[mask] = consensus[mask].mean()
        y = zscore(y)
    elif name == "dino_self":
        y = zr(upper(base))
    else:
        raise ValueError(name)
    return np.asarray(y, dtype=np.float32)


def summarize(v: np.ndarray, seed: int) -> dict:
    x = np.asarray(v, dtype=float)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "positive_n": int(np.sum(x > 0)),
        "n": int(len(x)),
        "exact_two_sided_signflip_p": float(exact_signflip(x)),
        "bootstrap_mean_95ci": bootstrap_ci(x, seed),
        "values": x.tolist(),
    }


class FlexibleResidualAdapter(nn.Module):
    def __init__(self, width: int = 384, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.up(F.gelu(self.down(self.norm(x)))), dim=-1)


def fit_flexible(
    image_features: np.ndarray,
    target: np.ndarray,
    bottleneck: int,
    lambda_anchor: float,
    seed: int,
    epochs: int = 400,
) -> FlexibleResidualAdapter:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    model = FlexibleResidualAdapter(bottleneck=bottleneck).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    base = F.normalize(x, dim=-1)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        zi = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        zc = F.normalize(zi.mean(dim=1), dim=-1)
        pred = torch_upper_cosine(zc)
        anchor = 1.0 - (zi * base).sum(dim=-1).mean()
        loss = pearson_loss(pred, y) + lambda_anchor * anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.cpu()


@torch.no_grad()
def embed_flexible(model: nn.Module, features: np.ndarray) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    x = torch.as_tensor(features[:, None, :], dtype=torch.float32, device=device)
    z = model(x.reshape(-1, x.shape[-1])).reshape(x.shape).mean(dim=1)
    return F.normalize(z, dim=-1).cpu().numpy().astype(np.float32)


def run_target_baselines(features: np.ndarray, category: np.ndarray, eeg: dict, meg: dict):
    folds = make_object_folds(category)
    gains_eeg = np.full((len(TARGETS), 10, 3), np.nan)
    gains_meg = np.full((len(TARGETS), 16, 3), np.nan)
    unique_eeg = np.full((len(TARGETS), 10, 3), np.nan)
    unique_meg = np.full((len(TARGETS), 16, 3), np.nan)
    for p_idx, pf in enumerate(participant_folds()):
        eg = group_matrix(eeg["mean"], pf["eeg_teacher"])
        mg = group_matrix(meg["late"]["mean"], pf["meg_teacher"])
        for fold in folds:
            train, test, f = fold["train"], fold["test"], fold["fold"]
            base = cosine_rdm(features[test])
            q = category_control_basis(base, category[test])
            e_neural = [subset_vec(eeg["mean"][i], test) for i in pf["eeg_eval"]]
            m_neural = [subset_vec(meg["late"]["mean"][i], test) for i in pf["meg_eval"]]
            for ti, name in enumerate(TARGETS):
                print(f"baseline target={name}, participant_fold={p_idx}, image_fold={f}", flush=True)
                y = make_target(name, eg, mg, features, category, train)
                embeddings = []
                for seed in SEEDS:
                    model = fit_adapter(features[train, None, :], y, 100.0, seed, 400)
                    embeddings.append(adapted_concept_embeddings(model, features[test, None, :]))
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                z = np.mean(embeddings, axis=0)
                z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
                adapted = cosine_rdm(z)
                egain, eunique = evaluate_rdm(adapted, base, e_neural, q)
                mgain, munique = evaluate_rdm(adapted, base, m_neural, q)
                gains_eeg[ti, pf["eeg_eval"], f] = egain
                gains_meg[ti, pf["meg_eval"], f] = mgain
                unique_eeg[ti, pf["eeg_eval"], f] = eunique
                unique_meg[ti, pf["meg_eval"], f] = munique
    return gains_eeg, gains_meg, unique_eeg, unique_meg


def run_hyper_grid(features: np.ndarray, category: np.ndarray, eeg: dict, meg: dict):
    folds = make_object_folds(category)
    configs = [(b, l) for b in (32, 64, 128) for l in (10.0, 100.0, 1000.0)]
    out = []
    for bottleneck, lam in configs:
        egain = np.full((10, 3), np.nan)
        mgain = np.full((16, 3), np.nan)
        for p_idx, pf in enumerate(participant_folds()):
            eg = group_matrix(eeg["mean"], pf["eeg_teacher"])
            mg = group_matrix(meg["late"]["mean"], pf["meg_teacher"])
            for fold in folds:
                train, test, f = fold["train"], fold["test"], fold["fold"]
                base = cosine_rdm(features[test])
                q = category_control_basis(base, category[test])
                y = make_target("raw_consensus", eg, mg, features, category, train)
                print(f"grid bottleneck={bottleneck}, lambda={lam}, participant_fold={p_idx}, image_fold={f}", flush=True)
                model = fit_flexible(features[train, None, :], y, bottleneck, lam, SEEDS[0])
                z = embed_flexible(model, features[test])
                adapted = cosine_rdm(z)
                e_neural = [subset_vec(eeg["mean"][i], test) for i in pf["eeg_eval"]]
                m_neural = [subset_vec(meg["late"]["mean"][i], test) for i in pf["meg_eval"]]
                egain[pf["eeg_eval"], f] = evaluate_rdm(adapted, base, e_neural, q)[0]
                mgain[pf["meg_eval"], f] = evaluate_rdm(adapted, base, m_neural, q)[0]
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        out.append({
            "bottleneck": bottleneck,
            "lambda_anchor": lam,
            "eeg_mean_gain": float(egain.mean()),
            "eeg_positive_n": int(np.sum(egain.mean(axis=1) > 0)),
            "meg_mean_gain": float(mgain.mean()),
            "meg_positive_n": int(np.sum(mgain.mean(axis=1) > 0)),
            "macro_gain": float(0.5 * (egain.mean() + mgain.mean())),
            "eeg_values": egain.mean(axis=1).tolist(),
            "meg_values": mgain.mean(axis=1).tolist(),
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cichy_idx, category, categories = load_mapping(ROOT / MAPPING_NAME)
    features = np.load(ROOT / DINO_NAME).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    eeg = load_eeg(EEG_DIR)
    meg = load_meg(MEG_FILE, cichy_idx)
    ge, gm, ue, um = run_target_baselines(features, category, eeg, meg)
    baseline_results = {}
    for i, name in enumerate(TARGETS):
        baseline_results[name] = {
            "heldout_eeg_gain": summarize(ge[i].mean(axis=1), 20260900 + i),
            "heldout_meg_gain": summarize(gm[i].mean(axis=1), 20260910 + i),
            "heldout_eeg_unique_movement": summarize(ue[i].mean(axis=1), 20260920 + i),
            "heldout_meg_unique_movement": summarize(um[i].mean(axis=1), 20260930 + i),
            "object_fold_gain_means": {
                "eeg": ge[i].mean(axis=0).tolist(),
                "meg": gm[i].mean(axis=0).tolist(),
            },
        }
    grid = run_hyper_grid(features, category, eeg, meg)
    result = {
        "analysis": "matched target baselines and fixed hyperparameter-grid sensitivity",
        "analysis_class": "post-hoc sensitivity analyses requested during manuscript audit",
        "target_definitions": {
            "raw_consensus": "equal-weight average of rank-transformed late EEG and MEG RDMs",
            "controlled_residual": "raw consensus residualized against DINOv3 and complete category-pair design within each training fold",
            "category_only": "category-pair means from the raw consensus, with all image-specific deviations removed",
            "dino_self": "DINOv3 training-image geometry; architecture and anchor matched negative control",
        },
        "target_baselines": baseline_results,
        "hyperparameter_grid": {
            "selection_rule": "No configuration was selected; all nine prespecified grid cells are reported.",
            "optimization_seed": SEEDS[0],
            "cells": grid,
        },
        "categories": categories,
    }
    np.savez_compressed(
        OUT / "ADAPTER_BASELINE_HYPERPARAM_ARRAYS_v001.npz",
        target_names=np.asarray(TARGETS), eeg_gain=ge, meg_gain=gm,
        eeg_unique=ue, meg_unique=um,
    )
    path = OUT / "ADAPTER_BASELINE_HYPERPARAM_RESULTS_v001.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
