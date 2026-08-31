from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.adaptation.run_multibackbone_adaptation import DynamicResidualAdapter


SEEDS = (20260722, 20260723, 20260724)
EPOCHS = 400
LAMBDA_ANCHOR = 100.0
PAIR_COUNT = 10_000
PAIR_SEED = 20260831
BOOTSTRAP_SEED = 20260831
PROTOCOL = ROOT / "config" / "protocols" / "nod_directional_retraining.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / values.std(ddof=0)


def vector_to_matrix(vector: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float64)
    upper = np.triu_indices(n, 1)
    matrix[upper] = vector
    matrix[(upper[1], upper[0])] = vector
    return matrix


def matrix_upper(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.triu_indices(len(matrix), 1)], dtype=np.float64)


def participant_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    even = np.arange(0, n, 2)
    odd = np.arange(1, n, 2)
    return [(even, odd), (odd, even)]


def concept_folds(classes: np.ndarray, superclasses: np.ndarray) -> list[np.ndarray]:
    fold_id = np.full(len(classes), -1, dtype=int)
    for superclass in sorted(np.unique(superclasses)):
        indices = np.flatnonzero(superclasses == superclass)
        indices = indices[np.argsort(classes[indices])]
        fold_id[indices] = np.arange(len(indices)) % 4
    if np.any(fold_id < 0):
        raise RuntimeError("Incomplete concept-fold assignment")
    return [np.flatnonzero(fold_id == fold) for fold in range(4)]


def class_features(feature_file: Path, index_file: Path, classes: np.ndarray) -> np.ndarray:
    features = np.load(feature_file, mmap_mode="r")
    index = pd.read_csv(index_file, dtype={"image_id": str, "class_id": str})
    unique = index.drop_duplicates("image_id")
    class_lookup = {class_id: position for position, class_id in enumerate(classes)}
    sums = np.zeros((len(classes), features.shape[1]), dtype=np.float64)
    counts = np.zeros(len(classes), dtype=np.int64)
    for row in unique.itertuples(index=False):
        if row.class_id not in class_lookup:
            continue
        position = class_lookup[row.class_id]
        sums[position] += np.asarray(features[int(row.feature_row)], dtype=np.float64)
        counts[position] += 1
    if np.any(counts == 0):
        raise RuntimeError("At least one class lacks frozen features")
    centroids = sums / counts[:, None]
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    return centroids.astype(np.float32)


def load_neural(cache_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(cache_dir.glob("sub-*_paired_rdms_v001.npz"))
    if len(paths) != 19:
        raise RuntimeError(f"Expected 19 paired caches, found {len(paths)}")
    subjects, eeg, meg = [], [], []
    classes = superclasses = None
    for path in paths:
        data = np.load(path, allow_pickle=True)
        current_classes = data["classes"].astype(str)
        current_superclasses = data["superclasses"].astype(str)
        if classes is None:
            classes = current_classes
            superclasses = current_superclasses
        elif not np.array_equal(classes, current_classes) or not np.array_equal(superclasses, current_superclasses):
            raise RuntimeError(f"Class ordering differs in {path}")
        subjects.append(str(data["subject"].item()))
        eeg.append(vector_to_matrix(data["eeg_native_post"].astype(float), len(current_classes)))
        meg.append(vector_to_matrix(data["meg_native_post"].astype(float), len(current_classes)))
    return subjects, np.asarray(classes), np.asarray(superclasses), np.asarray(eeg), np.asarray(meg)


def teacher_matrix(matrices: np.ndarray, teacher_indices: np.ndarray) -> np.ndarray:
    ranked = []
    for index in teacher_indices:
        vector = zscore(rankdata(matrix_upper(matrices[index]), method="average"))
        ranked.append(vector_to_matrix(vector, matrices.shape[1]))
    return np.mean(ranked, axis=0)


def selected_training_pairs(train: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.triu_indices(len(train), 1)
    rng = np.random.default_rng(PAIR_SEED + fold)
    chosen = rng.choice(len(left), size=min(PAIR_COUNT, len(left)), replace=False)
    return train[left[chosen]], train[right[chosen]]


def fit_adapter(
    features: np.ndarray,
    target_matrix: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    seed: int,
) -> DynamicResidualAdapter:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_matrix[pair_left, pair_right], dtype=torch.float32, device=device)
    target = (target - target.mean()) / target.std(unbiased=False).clamp_min(1e-12)
    left = torch.as_tensor(pair_left, dtype=torch.long, device=device)
    right = torch.as_tensor(pair_right, dtype=torch.long, device=device)
    model = DynamicResidualAdapter(width=features.shape[1], bottleneck=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    base = F.normalize(x, dim=-1)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        adapted = model(x)
        predicted = 1.0 - (adapted[left] * adapted[right]).sum(dim=1)
        predicted = (predicted - predicted.mean()) / predicted.std(unbiased=False).clamp_min(1e-12)
        correlation = (predicted * target).mean()
        anchor = 1.0 - (adapted * base).sum(dim=1).mean()
        loss = 1.0 - correlation + LAMBDA_ANCHOR * anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.cpu()


@torch.no_grad()
def adapted_embeddings(model: DynamicResidualAdapter, features: np.ndarray) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return model(tensor).cpu().numpy().astype(np.float32)


def cosine_rdm(features: np.ndarray) -> np.ndarray:
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    matrix = 1.0 - features @ features.T
    np.fill_diagonal(matrix, 0.0)
    return matrix


def gain(model_matrix: np.ndarray, frozen_matrix: np.ndarray, neural_matrix: np.ndarray, test: np.ndarray) -> float:
    neural = matrix_upper(neural_matrix[np.ix_(test, test)])
    adapted = matrix_upper(model_matrix[np.ix_(test, test)])
    frozen = matrix_upper(frozen_matrix[np.ix_(test, test)])
    return float(spearmanr(adapted, neural).statistic - spearmanr(frozen, neural).statistic)


def exact_signflip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    count = total = 0
    for start in range(0, 1 << len(values), 65_536):
        ids = np.arange(start, min(start + 65_536, 1 << len(values)), dtype=np.uint64)
        bits = ((ids[:, None] >> np.arange(len(values), dtype=np.uint64)) & 1).astype(float)
        means = ((2.0 * bits - 1.0) @ values) / len(values)
        count += int(np.sum(np.abs(means) >= observed - 1e-15))
        total += len(means)
    return float(count / total)


def summarize(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive_n": int(np.sum(values > 0)),
        "n": int(len(values)),
        "exact_two_sided_signflip_p": exact_signflip(values),
        "bootstrap_mean_95ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "values": values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()

    subjects, classes, superclasses, eeg, meg = load_neural(args.cache_dir)
    features = class_features(args.features, args.index, classes)
    folds = concept_folds(classes, superclasses)
    directions = {"eeg_to_meg": (eeg, meg), "meg_to_eeg": (meg, eeg)}
    values = {direction: np.full((len(subjects), 4), np.nan) for direction in directions}
    frozen = cosine_rdm(features)

    for participant_fold_index, (teacher_indices, evaluation_indices) in enumerate(participant_folds(len(subjects))):
        for direction, (teacher_data, evaluation_data) in directions.items():
            target_matrix = teacher_matrix(teacher_data, teacher_indices)
            for fold, test in enumerate(folds):
                train = np.setdiff1d(np.arange(len(classes)), test)
                pair_left, pair_right = selected_training_pairs(train, fold)
                seed_embeddings = []
                print(
                    f"direction={direction}, participant_fold={participant_fold_index}, concept_fold={fold}",
                    flush=True,
                )
                train_lookup = {class_index: local_index for local_index, class_index in enumerate(train)}
                local_left = np.asarray([train_lookup[index] for index in pair_left], dtype=int)
                local_right = np.asarray([train_lookup[index] for index in pair_right], dtype=int)
                local_target = target_matrix[np.ix_(train, train)]
                for seed in SEEDS:
                    model = fit_adapter(features[train], local_target, local_left, local_right, seed)
                    seed_embeddings.append(adapted_embeddings(model, features))
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                adapted = np.mean(seed_embeddings, axis=0)
                adapted /= np.maximum(np.linalg.norm(adapted, axis=1, keepdims=True), 1e-12)
                adapted_matrix = cosine_rdm(adapted)
                for participant in evaluation_indices:
                    values[direction][participant, fold] = gain(
                        adapted_matrix, frozen, evaluation_data[participant], test
                    )

    if any(np.isnan(array).any() for array in values.values()):
        raise RuntimeError("Incomplete participant-by-fold result")
    summaries = {
        direction: summarize(array.mean(axis=1), BOOTSTRAP_SEED + offset)
        for offset, (direction, array) in enumerate(values.items())
    }
    payload = {
        "analysis": "post-hoc NOD directional retraining with participant- and concept-disjoint evaluation",
        "protocol": "config/protocols/nod_directional_retraining.md",
        "protocol_sha256": sha256(PROTOCOL),
        "participants": subjects,
        "n_participants": len(subjects),
        "concept_fold_sizes": [int(len(fold)) for fold in folds],
        "optimization_pair_count": PAIR_COUNT,
        "seeds": list(SEEDS),
        "directions": summaries,
        "claim_boundary": "Exploratory robustness analysis within a previously examined dataset; not an untouched replication.",
    }
    output_json = ROOT / "results" / "reported" / "nod_directional_retraining.json"
    output_npz = ROOT / "source_data" / "supplementary" / "nod_directional_retraining.npz"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_npz,
        subjects=np.asarray(subjects),
        classes=classes,
        superclasses=superclasses,
        concept_folds=np.asarray([np.isin(np.arange(len(classes)), fold) for fold in folds]),
        eeg_to_meg_gain=values["eeg_to_meg"],
        meg_to_eeg_gain=values["meg_to_eeg"],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
