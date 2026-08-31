from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "analysis" / "_shared"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SHARED))

from common import pearson_loss, set_seed, torch_upper_cosine
from run_late_crossmodal_adapter_gate_v001 import (
    category_control_basis,
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
)
from analysis.adaptation.run_multibackbone_adaptation import cosine_rdm
from analysis.robustness.run_adapter_baselines import make_target, participant_folds


SEEDS = (20260722, 20260723, 20260724)
EPOCHS = 400
LAMBDA_ANCHOR = 100.0
BOTTLENECK = 64
PROTOCOL = ROOT / "config" / "protocols" / "adapter_architecture_baselines.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NonlinearResidualAdapter(nn.Module):
    def __init__(self, width: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.up(F.gelu(self.down(self.norm(x)))), dim=-1)


class LinearResidualAdapter(nn.Module):
    def __init__(self, width: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.up(self.down(self.norm(x))), dim=-1)


class DiagonalFeatureReweighting(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.log_weight = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = self.log_weight - self.log_weight.mean()
        weight = torch.exp(torch.clamp(centered, -6.0, 6.0))
        return F.normalize(x * weight, dim=-1)


def build_model(name: str, width: int) -> nn.Module:
    if name == "nonlinear_residual":
        return NonlinearResidualAdapter(width, BOTTLENECK)
    if name == "linear_residual":
        return LinearResidualAdapter(width, BOTTLENECK)
    if name == "diagonal_reweighting":
        return DiagonalFeatureReweighting(width)
    raise ValueError(name)


def fit_model(name: str, image_features: np.ndarray, target: np.ndarray, seed: int) -> nn.Module:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    model = build_model(name, x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    base = F.normalize(x, dim=-1)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        adapted = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        concept = F.normalize(adapted.mean(dim=1), dim=-1)
        predicted = torch_upper_cosine(concept)
        anchor = 1.0 - (adapted * base).sum(dim=-1).mean()
        loss = pearson_loss(predicted, y) + LAMBDA_ANCHOR * anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.cpu()


@torch.no_grad()
def embeddings(model: nn.Module, features: np.ndarray) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    x = torch.as_tensor(features[:, None, :], dtype=torch.float32, device=device)
    out = model(x.reshape(-1, x.shape[-1])).reshape(x.shape).mean(dim=1)
    return F.normalize(out, dim=-1).cpu().numpy().astype(np.float32)


def summarize(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive_n": int(np.sum(values > 0)),
        "n": int(len(values)),
        "exact_two_sided_signflip_p": float(exact_signflip(values)),
        "bootstrap_mean_95ci": bootstrap_ci(values, seed),
        "values": values.tolist(),
    }


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def run(eeg_dir: Path, meg_file: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    cichy_idx, category, _ = load_mapping(ROOT / MAPPING_NAME)
    features = np.load(ROOT / DINO_NAME).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    eeg = load_eeg(eeg_dir)
    meg = load_meg(meg_file, cichy_idx)
    folds = make_object_folds(category)
    architecture_names = ["nonlinear_residual", "linear_residual", "diagonal_reweighting"]
    eeg_gain = np.full((len(architecture_names), 10, 3), np.nan)
    meg_gain = np.full((len(architecture_names), 16, 3), np.nan)

    for participant_fold_index, participant_fold in enumerate(participant_folds()):
        eeg_teacher = group_matrix(eeg["mean"], participant_fold["eeg_teacher"])
        meg_teacher = group_matrix(meg["late"]["mean"], participant_fold["meg_teacher"])
        for fold in folds:
            train, test, image_fold = fold["train"], fold["test"], fold["fold"]
            target = make_target(
                "raw_consensus", eeg_teacher, meg_teacher, features, category, train
            )
            frozen = cosine_rdm(features[test])
            controls = category_control_basis(frozen, category[test])
            eeg_neural = [subset_vec(eeg["mean"][index], test) for index in participant_fold["eeg_eval"]]
            meg_neural = [
                subset_vec(meg["late"]["mean"][index], test)
                for index in participant_fold["meg_eval"]
            ]
            for architecture_index, architecture in enumerate(architecture_names):
                print(
                    f"architecture={architecture}, participant_fold={participant_fold_index}, "
                    f"image_fold={image_fold}",
                    flush=True,
                )
                seed_embeddings = []
                for seed in SEEDS:
                    fitted = fit_model(architecture, features[train, None, :], target, seed)
                    seed_embeddings.append(embeddings(fitted, features[test]))
                    del fitted
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                adapted = np.mean(seed_embeddings, axis=0)
                adapted /= np.maximum(np.linalg.norm(adapted, axis=1, keepdims=True), 1e-12)
                adapted_rdm = cosine_rdm(adapted)
                eeg_gain[architecture_index, participant_fold["eeg_eval"], image_fold] = evaluate_rdm(
                    adapted_rdm, frozen, eeg_neural, controls
                )[0]
                meg_gain[architecture_index, participant_fold["meg_eval"], image_fold] = evaluate_rdm(
                    adapted_rdm, frozen, meg_neural, controls
                )[0]

    counts = {
        name: parameter_count(build_model(name, features.shape[1])) for name in architecture_names
    }
    return eeg_gain, meg_gain, architecture_names, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-dir", type=Path, required=True)
    parser.add_argument("--meg-file", type=Path, required=True)
    args = parser.parse_args()

    eeg_gain, meg_gain, architecture_names, counts = run(args.eeg_dir, args.meg_file)
    results = {}
    for index, name in enumerate(architecture_names):
        eeg_values = eeg_gain[index].mean(axis=1)
        meg_values = meg_gain[index].mean(axis=1)
        results[name] = {
            "trainable_parameters": counts[name],
            "heldout_eeg_gain": summarize(eeg_values, 20261000 + index),
            "heldout_meg_gain": summarize(meg_values, 20261010 + index),
        }

    reference = architecture_names.index("nonlinear_residual")
    contrasts = {}
    for index, name in enumerate(architecture_names):
        if index == reference:
            continue
        contrasts[f"{name}_minus_nonlinear_residual"] = {
            "heldout_eeg": summarize(
                eeg_gain[index].mean(axis=1) - eeg_gain[reference].mean(axis=1),
                20261020 + index,
            ),
            "heldout_meg": summarize(
                meg_gain[index].mean(axis=1) - meg_gain[reference].mean(axis=1),
                20261030 + index,
            ),
        }

    payload = {
        "analysis": "post-hoc adapter architecture sensitivity",
        "protocol": "config/protocols/adapter_architecture_baselines.md",
        "protocol_sha256": sha256(PROTOCOL),
        "optimization": {
            "epochs": EPOCHS,
            "anchor_weight": LAMBDA_ANCHOR,
            "seeds": list(SEEDS),
        },
        "architectures": results,
        "paired_contrasts": contrasts,
        "random_target_control": {
            "analysis": "category-preserving neural-target permutations",
            "n_permutations": 9999,
            "reported_file": "results/reported/target_specificity_permutations.json",
        },
    }
    output_json = ROOT / "results" / "reported" / "adapter_architecture_baselines.json"
    output_npz = ROOT / "source_data" / "supplementary" / "adapter_architecture_baselines.npz"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_npz,
        architecture_names=np.asarray(architecture_names),
        eeg_gain=eeg_gain,
        meg_gain=meg_gain,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
