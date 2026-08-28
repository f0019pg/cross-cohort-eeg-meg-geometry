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
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

from common import pearson_loss, set_seed, torch_upper_cosine
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


OUT = ROOT_DEFAULT / "derived" / "multibackbone_adaptation"
PROTOCOL = ROOT_DEFAULT / "config" / "protocols" / "multibackbone_protocol.md"
ORIGINAL_STAGE0 = ROOT_DEFAULT / "results" / "reported" / "source_geometry.json"
ORIGINAL_STAGE1 = ROOT_DEFAULT / "results" / "reported" / "source_adapter_gate.json"
FEATURE_DIR = ROOT_DEFAULT / "source_data" / "model_features"
FEATURE_MANIFEST = FEATURE_DIR / "FEATURE_MANIFEST.json"
SEEDS = [20260722, 20260723, 20260724]
SHUFFLE_SEED = 20260807
N_SHUFFLE = 99
LAMBDA_ANCHOR = 100.0
EPOCHS = 400

MODELS = {
    "DINOv3": {
        "path": FEATURE_DIR / "features_dinov3_92x384.npy",
        "width": 384,
        "bottleneck": 64,
        "new_inference": False,
    },
    "CLIP-B32": {
        "path": FEATURE_DIR / "features_clip_b32_image_92.npy",
        "width": 512,
        "bottleneck": 48,
        "new_inference": True,
    },
    "SigLIP-base": {
        "path": FEATURE_DIR / "features_siglip_base_image_92.npy",
        "width": 768,
        "bottleneck": 32,
        "new_inference": True,
    },
}
NEW_MODELS = [name for name, spec in MODELS.items() if spec["new_inference"]]


class DynamicResidualAdapter(nn.Module):
    def __init__(self, width: int, bottleneck: int):
        super().__init__()
        self.width = int(width)
        self.bottleneck = int(bottleneck)
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(x))))
        return F.normalize(x + residual, dim=-1)


def parameter_count(width: int, bottleneck: int) -> int:
    # LayerNorm weight/bias + down weight/bias + up weight/bias.
    return 2 * width + width * bottleneck + bottleneck + bottleneck * width + width


def fit_dynamic_adapter(
    image_features: np.ndarray,
    target: np.ndarray,
    width: int,
    bottleneck: int,
    seed: int,
    epochs: int = EPOCHS,
    lambda_anchor: float = LAMBDA_ANCHOR,
) -> DynamicResidualAdapter:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    if x.shape[-1] != width:
        raise ValueError(f"Expected width {width}, got {x.shape[-1]}")
    model = DynamicResidualAdapter(width=width, bottleneck=bottleneck).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    base = F.normalize(x, dim=-1)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        zi = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        zc = F.normalize(zi.mean(dim=1), dim=-1)
        pred = torch_upper_cosine(zc)
        anchor = 1.0 - (zi * base).sum(dim=-1).mean()
        loss = pearson_loss(pred, y) + float(lambda_anchor) * anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.cpu()


@torch.no_grad()
def adapted_embeddings(model: DynamicResidualAdapter, features: np.ndarray) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    zi = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    zc = F.normalize(zi.mean(dim=1), dim=-1)
    return zc.cpu().numpy().astype(np.float32)


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
        train_set = set(test.tolist())
        train = np.asarray([i for i in range(72) if i not in train_set], dtype=int)
        if len(train) != 48 or len(test) != 24:
            raise RuntimeError("Invalid object fold")
        if not np.all(np.bincount(category[train], minlength=6) == 8):
            raise RuntimeError("Unbalanced training fold")
        if not np.all(np.bincount(category[test], minlength=6) == 4):
            raise RuntimeError("Unbalanced evaluation fold")
        folds.append({"fold": fold, "train": train, "test": test})
    return folds


def consensus_target(eeg_group: np.ndarray, meg_group: np.ndarray, idx: np.ndarray) -> np.ndarray:
    eeg = zr(upper(eeg_group[np.ix_(idx, idx)]))
    meg = zr(upper(meg_group[np.ix_(idx, idx)]))
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


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_features_72() -> tuple[dict[str, np.ndarray], np.ndarray]:
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    idx = np.asarray(manifest["eeg72_indices_1based"], dtype=int) - 1
    if len(idx) != 72 or len(np.unique(idx)) != 72:
        raise RuntimeError("Invalid fixed 72-image index")
    features = {}
    for name, spec in MODELS.items():
        x = np.load(spec["path"]).astype(np.float32)
        if x.shape != (92, spec["width"]):
            raise RuntimeError(f"Unexpected {name} feature shape {x.shape}")
        x = x[idx]
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        features[name] = x
    original_dino = np.load(ROOT_DEFAULT / DINO_NAME).astype(np.float32)
    if not np.array_equal(features["DINOv3"], original_dino / np.maximum(np.linalg.norm(original_dino, axis=1, keepdims=True), 1e-12)):
        # The raw files are exactly equal; this fallback comparison tolerates only normalization arithmetic.
        if not np.allclose(features["DINOv3"], original_dino, atol=1e-7, rtol=1e-7):
            raise RuntimeError("DINO 92->72 subset does not reproduce the original feature file")
    return features, idx


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


def self_test() -> dict:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 1, 32)).astype(np.float32)
    y = rng.normal(size=66).astype(np.float32)
    set_seed(7)
    model = DynamicResidualAdapter(32, 8)
    with torch.no_grad():
        before = model(torch.as_tensor(x[:, 0])).numpy()
    base = x[:, 0] / np.linalg.norm(x[:, 0], axis=1, keepdims=True)
    if not np.allclose(before, base, atol=1e-6):
        raise RuntimeError("Zero-init adapter does not reproduce base embeddings")
    fitted = fit_dynamic_adapter(x, y, 32, 8, seed=7, epochs=2)
    out = adapted_embeddings(fitted, x)
    if out.shape != (12, 32) or not np.isfinite(out).all():
        raise RuntimeError("Dynamic adapter self-test failed")
    return {"status": "SELF_TEST_PASSED", "shape": list(out.shape)}


def check_inputs(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    if not PROTOCOL.exists():
        raise FileNotFoundError("Protocol lock is absent")
    if not ORIGINAL_STAGE0.exists() or not ORIGINAL_STAGE1.exists():
        raise FileNotFoundError("Original stage results are absent")
    stage0 = json.loads(ORIGINAL_STAGE0.read_text(encoding="utf-8"))
    stage1 = json.loads(ORIGINAL_STAGE1.read_text(encoding="utf-8"))
    if stage0.get("decision") != "GO_STAGE1_ADAPTER" or stage1.get("decision") != "GO_STAGE2_EXTERNAL":
        raise RuntimeError("Original source gates did not authorize the robustness audit")
    mapping = root / MAPPING_NAME
    _, category, category_names = load_mapping(mapping)
    features, idx = load_features_72()
    folds = make_object_folds(category)
    model_audit = {}
    for name, spec in MODELS.items():
        params = parameter_count(spec["width"], spec["bottleneck"])
        model_audit[name] = {
            "source_shape_92": [92, spec["width"]],
            "analysis_shape_72": list(features[name].shape),
            "width": spec["width"],
            "bottleneck": spec["bottleneck"],
            "trainable_parameters": params,
            "feature_path": str(spec["path"]),
            "feature_sha256": hash_file(spec["path"]),
            "new_inference": spec["new_inference"],
        }
    counts = [x["trainable_parameters"] for x in model_audit.values()]
    audit = {
        "status": "MULTIBACKBONE_INPUT_CHECK_PASSED",
        "protocol_sha256": hash_file(PROTOCOL),
        "original_stage0_sha256": hash_file(ORIGINAL_STAGE0),
        "original_stage1_sha256": hash_file(ORIGINAL_STAGE1),
        "mapping_sha256": sha256(mapping),
        "feature_manifest_sha256": hash_file(FEATURE_MANIFEST),
        "fixed_72_indices_1based": (idx + 1).tolist(),
        "categories": category_names,
        "models": model_audit,
        "parameter_count_max_over_min": float(max(counts) / min(counts)),
        "object_folds": [{"fold": x["fold"], "train_n": len(x["train"]), "test_n": len(x["test"])} for x in folds],
        "eeg_dir": str(eeg_dir),
        "meg_file": str(meg_file),
        "seeds": SEEDS,
        "teacher_shuffles": {"n": N_SHUFFLE, "seed": SHUFFLE_SEED},
        "self_test": self_test(),
    }
    (OUT / "INPUT_AUDIT_v001.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def run_observed_for_model(
    name: str,
    spec: dict,
    features: np.ndarray,
    eeg: dict,
    meg: dict,
    category: np.ndarray,
    folds: list[dict],
) -> tuple[dict, list[dict]]:
    eeg_gain = np.full((10, 3), np.nan)
    meg_gain = np.full((16, 3), np.nan)
    eeg_unique = np.full((10, 3), np.nan)
    meg_unique = np.full((16, 3), np.nan)
    eeg_single = np.full((10, 3), np.nan)
    meg_single = np.full((16, 3), np.nan)
    geometry = []
    configs = []
    for p_fold in participant_folds():
        eeg_group = group_matrix(eeg["mean"], p_fold["eeg_teacher"])
        meg_group = group_matrix(meg["late"]["mean"], p_fold["meg_teacher"])
        for o_fold in folds:
            train, test = o_fold["train"], o_fold["test"]
            target = consensus_target(eeg_group, meg_group, train)
            base_rdm = cosine_rdm(features[test])
            q = category_control_basis(base_rdm, category[test])
            eeg_neural = [subset_vec(eeg["mean"][i], test) for i in p_fold["eeg_eval"]]
            meg_neural = [subset_vec(meg["late"]["mean"][i], test) for i in p_fold["meg_eval"]]
            embeddings = []
            for seed in SEEDS:
                print(f"observed {name}: {p_fold['name']}, fold {o_fold['fold']}, seed {seed}", flush=True)
                model = fit_dynamic_adapter(
                    features[train, None, :], target, spec["width"], spec["bottleneck"], seed
                )
                embeddings.append(adapted_embeddings(model, features[test, None, :]))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            ensemble = np.mean(embeddings, axis=0)
            ensemble /= np.maximum(np.linalg.norm(ensemble, axis=1, keepdims=True), 1e-12)
            adapted_rdm = cosine_rdm(ensemble)
            single_rdm = cosine_rdm(embeddings[0])
            eeg_g, eeg_u = evaluate_rdm(adapted_rdm, base_rdm, eeg_neural, q)
            meg_g, meg_u = evaluate_rdm(adapted_rdm, base_rdm, meg_neural, q)
            eeg_s, _ = evaluate_rdm(single_rdm, base_rdm, eeg_neural, q)
            meg_s, _ = evaluate_rdm(single_rdm, base_rdm, meg_neural, q)
            fi = o_fold["fold"]
            eeg_gain[p_fold["eeg_eval"], fi] = eeg_g
            meg_gain[p_fold["meg_eval"], fi] = meg_g
            eeg_unique[p_fold["eeg_eval"], fi] = eeg_u
            meg_unique[p_fold["meg_eval"], fi] = meg_u
            eeg_single[p_fold["eeg_eval"], fi] = eeg_s
            meg_single[p_fold["meg_eval"], fi] = meg_s
            geometry.append({
                "participant_fold": p_fold["name"],
                "object_fold": fi,
                "rho": float(spearmanr(upper(adapted_rdm), upper(base_rdm)).statistic),
            })
            configs.append({
                "participant_fold": p_fold,
                "object_fold": o_fold,
                "eeg_group": eeg_group,
                "meg_group": meg_group,
                "base_rdm": base_rdm,
                "eeg_neural": eeg_neural,
                "meg_neural": meg_neural,
            })
    arrays = (eeg_gain, meg_gain, eeg_unique, meg_unique, eeg_single, meg_single)
    if any(np.isnan(x).any() for x in arrays):
        raise RuntimeError(f"{name} left unevaluated cells")
    result = {
        "eeg_gain": eeg_gain,
        "meg_gain": meg_gain,
        "eeg_unique": eeg_unique,
        "meg_unique": meg_unique,
        "eeg_single": eeg_single,
        "meg_single": meg_single,
        "geometry": geometry,
        "object_fold_gain": [
            {
                "fold": fold,
                "eeg_mean_gain": float(eeg_gain[:, fold].mean()),
                "meg_mean_gain": float(meg_gain[:, fold].mean()),
            }
            for fold in range(3)
        ],
    }
    return result, configs


def dino_reproduction(dino: dict) -> dict:
    original = json.loads(ORIGINAL_STAGE1.read_text(encoding="utf-8"))
    comparisons = {
        "eeg_mean_abs_error": abs(float(dino["eeg_gain"].mean()) - original["eeg_alignment_gain"]["mean"]),
        "meg_mean_abs_error": abs(float(dino["meg_gain"].mean()) - original["meg_alignment_gain"]["mean"]),
        "eeg_unique_mean_abs_error": abs(float(dino["eeg_unique"].mean()) - original["unique_displacement"]["eeg"]["mean"]),
        "meg_unique_mean_abs_error": abs(float(dino["meg_unique"].mean()) - original["unique_displacement"]["meg"]["mean"]),
        "geometry_min_abs_error": abs(float(min(x["rho"] for x in dino["geometry"])) - original["geometry_preservation"]["minimum"]),
    }
    return {"passed": bool(max(comparisons.values()) < 1e-7), **comparisons}


def run_teacher_null(
    observed: dict[str, dict],
    configs: dict[str, list[dict]],
    features: dict[str, np.ndarray],
    category: np.ndarray,
) -> tuple[float, np.ndarray]:
    observed_by_model = {
        name: 0.5 * (observed[name]["eeg_single"].mean() + observed[name]["meg_single"].mean())
        for name in NEW_MODELS
    }
    observed_family = float(np.mean(list(observed_by_model.values())))
    rng = np.random.default_rng(SHUFFLE_SEED)
    null = np.empty(N_SHUFFLE, dtype=np.float64)
    for shuffle in range(N_SHUFFLE):
        perm = generate_within_category_permutation(rng, category)
        model_scores = []
        for name in NEW_MODELS:
            spec = MODELS[name]
            eeg_values, meg_values = [], []
            for config in configs[name]:
                train = config["object_fold"]["train"]
                test = config["object_fold"]["test"]
                target = consensus_target(
                    config["eeg_group"][np.ix_(perm, perm)],
                    config["meg_group"][np.ix_(perm, perm)],
                    train,
                )
                model = fit_dynamic_adapter(
                    features[name][train, None, :], target,
                    spec["width"], spec["bottleneck"], SEEDS[0]
                )
                embedding = adapted_embeddings(model, features[name][test, None, :])
                adapted = cosine_rdm(embedding)
                base = config["base_rdm"]
                for neural in config["eeg_neural"]:
                    eeg_values.append(float(spearmanr(upper(adapted), neural).statistic - spearmanr(upper(base), neural).statistic))
                for neural in config["meg_neural"]:
                    meg_values.append(float(spearmanr(upper(adapted), neural).statistic - spearmanr(upper(base), neural).statistic))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            model_scores.append(0.5 * (np.mean(eeg_values) + np.mean(meg_values)))
        null[shuffle] = float(np.mean(model_scores))
        print(f"joint teacher shuffle {shuffle + 1}/{N_SHUFFLE}: {null[shuffle]:+.6f}", flush=True)
    return observed_family, null


def run(root: Path, eeg_dir: Path, meg_file: Path) -> dict:
    if (OUT / "RESULTS_v001.json").exists():
        raise FileExistsError("Refusing to overwrite terminal results")
    audit = check_inputs(root, eeg_dir, meg_file)
    mapping = root / MAPPING_NAME
    cichy_idx, category, _ = load_mapping(mapping)
    features, _ = load_features_72()
    eeg = load_eeg(eeg_dir)
    meg = load_meg(meg_file, cichy_idx)
    folds = make_object_folds(category)
    observed, configs = {}, {}
    for name, spec in MODELS.items():
        observed[name], configs[name] = run_observed_for_model(
            name, spec, features[name], eeg, meg, category, folds
        )
    reproduction = dino_reproduction(observed["DINOv3"])
    if not reproduction["passed"]:
        raise RuntimeError(f"DINO reproduction failed: {reproduction}")

    observed_family_null_stat, null = run_teacher_null(observed, configs, features, category)
    shuffle_p = float((1 + np.sum(null >= observed_family_null_stat)) / (N_SHUFFLE + 1))

    individual = {}
    for mi, name in enumerate(MODELS):
        x = observed[name]
        individual[name] = {
            "eeg_alignment_gain": summarize(x["eeg_gain"].mean(axis=1), 1000 + mi * 10 + 1),
            "meg_alignment_gain": summarize(x["meg_gain"].mean(axis=1), 1000 + mi * 10 + 2),
            "unique_displacement": {
                "eeg": summarize(x["eeg_unique"].mean(axis=1), 1000 + mi * 10 + 3),
                "meg": summarize(x["meg_unique"].mean(axis=1), 1000 + mi * 10 + 4),
            },
            "object_fold_gain": x["object_fold_gain"],
            "geometry_preservation_minimum": float(min(g["rho"] for g in x["geometry"])),
            "single_seed_equal_modality_gain": float(0.5 * (x["eeg_single"].mean() + x["meg_single"].mean())),
        }

    family_eeg = np.mean([observed[name]["eeg_gain"].mean(axis=1) for name in NEW_MODELS], axis=0)
    family_meg = np.mean([observed[name]["meg_gain"].mean(axis=1) for name in NEW_MODELS], axis=0)
    family_eeg_unique = np.mean([observed[name]["eeg_unique"].mean(axis=1) for name in NEW_MODELS], axis=0)
    family_meg_unique = np.mean([observed[name]["meg_unique"].mean(axis=1) for name in NEW_MODELS], axis=0)
    fs_eeg = summarize(family_eeg, 2001)
    fs_meg = summarize(family_meg, 2002)
    fs_ue = summarize(family_eeg_unique, 2003)
    fs_um = summarize(family_meg_unique, 2004)

    a1 = fs_eeg["mean"] > 0.005 and fs_eeg["positive_n"] >= 8 and fs_eeg["exact_two_sided_signflip_p"] < 0.05
    a2 = fs_meg["mean"] > 0.005 and fs_meg["positive_n"] >= 12 and fs_meg["exact_two_sided_signflip_p"] < 0.05
    a3 = all(individual[name]["eeg_alignment_gain"]["mean"] > 0 and individual[name]["meg_alignment_gain"]["mean"] > 0 for name in NEW_MODELS)
    a4 = all(
        all(f["eeg_mean_gain"] > 0 and f["meg_mean_gain"] > 0 for f in individual[name]["object_fold_gain"])
        for name in NEW_MODELS
    )
    a5 = (
        fs_ue["mean"] > 0.02 and fs_ue["exact_two_sided_signflip_p"] < 0.05
        and fs_um["mean"] > 0.02 and fs_um["exact_two_sided_signflip_p"] < 0.05
    )
    a6 = all(individual[name]["geometry_preservation_minimum"] >= 0.95 for name in NEW_MODELS)
    a7 = shuffle_p < 0.05
    gates = {
        "A1_eeg_family_gain": bool(a1),
        "A2_meg_family_gain": bool(a2),
        "A3_each_backbone_positive": bool(a3),
        "A4_all_backbone_object_folds_positive": bool(a4),
        "A5_family_unique_displacement": bool(a5),
        "A6_geometry_preservation": bool(a6),
        "A7_joint_teacher_specificity": bool(a7),
    }
    if all(gates.values()):
        decision = "BACKBONE_GENERAL_LATE_CONSENSUS"
    elif all(gates[k] for k in ["A1_eeg_family_gain", "A2_meg_family_gain", "A5_family_unique_displacement", "A6_geometry_preservation", "A7_joint_teacher_specificity"]):
        decision = "PARTIAL_BACKBONE_GENERALITY"
    else:
        decision = "DINO_LIMITED_OR_INCONCLUSIVE"

    result = {
        "analysis": "parameter-matched non-DINO late EEG-MEG consensus adapter audit",
        "decision": decision,
        "input_audit": audit,
        "dino_reproduction": reproduction,
        "individual_backbones": individual,
        "non_dino_family": {
            "models": NEW_MODELS,
            "eeg_alignment_gain": fs_eeg,
            "meg_alignment_gain": fs_meg,
            "unique_displacement": {"eeg": fs_ue, "meg": fs_um},
            "teacher_specificity": {
                "observed_single_seed_equal_modality_gain": observed_family_null_stat,
                "n": N_SHUFFLE,
                "seed": SHUFFLE_SEED,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std()),
                "null_95th": float(np.quantile(null, 0.95)),
                "one_sided_p": shuffle_p,
            },
        },
        "gates": gates,
        "hashes": {
            "protocol": hash_file(PROTOCOL),
            "script": hash_file(Path(__file__)),
            "original_stage0": hash_file(ORIGINAL_STAGE0),
            "original_stage1": hash_file(ORIGINAL_STAGE1),
        },
    }
    arrays = {"teacher_shuffle_null": null}
    for name in MODELS:
        key = name.lower().replace("-", "_")
        for metric in ("eeg_gain", "meg_gain", "eeg_unique", "meg_unique", "eeg_single", "meg_single"):
            arrays[f"{key}_{metric}"] = observed[name][metric]
    np.savez_compressed(OUT / "PARTICIPANT_VALUES_v001.npz", **arrays)
    (OUT / "RESULTS_v001.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "dino_reproduction": reproduction,
        "individual_backbones": individual,
        "non_dino_family": result["non_dino_family"],
        "gates": gates,
    }, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--eeg-dir", type=str)
    parser.add_argument("--meg-file", type=str)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--check-inputs", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    root = args.root.resolve()
    eeg_dir, meg_file = find_sources(root, args.eeg_dir, args.meg_file)
    if args.check_inputs:
        print(json.dumps(check_inputs(root, eeg_dir, meg_file), indent=2, ensure_ascii=False))
    else:
        run(root, eeg_dir, meg_file)


if __name__ == "__main__":
    main()
