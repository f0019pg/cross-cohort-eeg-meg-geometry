from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = REPO / "analysis" / "adaptation" / "run_multibackbone_adaptation.py"
SOURCE_VALUES = REPO / "source_data" / "main" / "multibackbone_participant_values.npz"
DEFAULT_OUT = REPO / "derived" / "target_specificity_permutation"
DEFAULT_N = 9_999
DEFAULT_SEEDS = (20260722, 20260723, 20260724)
PERMUTATION_SEED = 20260807
CHECKPOINT_EVERY = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def import_source():
    spec = importlib.util.spec_from_file_location("locked_multibackbone_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_inputs(module):
    root = module.ROOT_DEFAULT.resolve()
    eeg_dir, meg_file = module.find_sources(root, None, None)
    mapping = root / module.MAPPING_NAME
    cichy_idx, category, category_names = module.load_mapping(mapping)
    features, feature_indices = module.load_features_72()
    eeg = module.load_eeg(eeg_dir)
    meg = module.load_meg(meg_file, cichy_idx)
    folds = module.make_object_folds(category)

    configs: dict[str, list[dict]] = {}
    for name in module.NEW_MODELS:
        configs[name] = []
        for participant_fold in module.participant_folds():
            eeg_group = module.group_matrix(eeg["mean"], participant_fold["eeg_teacher"])
            meg_group = module.group_matrix(meg["late"]["mean"], participant_fold["meg_teacher"])
            for object_fold in folds:
                test = object_fold["test"]
                base_rdm = module.cosine_rdm(features[name][test])
                eeg_neural = [
                    module.subset_vec(eeg["mean"][i], test)
                    for i in participant_fold["eeg_eval"]
                ]
                meg_neural = [
                    module.subset_vec(meg["late"]["mean"][i], test)
                    for i in participant_fold["meg_eval"]
                ]
                configs[name].append(
                    {
                        "participant_fold_name": participant_fold["name"],
                        "object_fold": object_fold,
                        "eeg_group": eeg_group,
                        "meg_group": meg_group,
                        "base_rdm": base_rdm,
                        "eeg_neural": eeg_neural,
                        "meg_neural": meg_neural,
                    }
                )

    return {
        "root": root,
        "eeg_dir": Path(eeg_dir),
        "meg_file": Path(meg_file),
        "mapping": mapping,
        "category": category,
        "category_names": category_names,
        "feature_indices": feature_indices,
        "features": features,
        "configs": configs,
    }


def generate_permutations(module, category: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(PERMUTATION_SEED)
    return np.stack(
        [module.generate_within_category_permutation(rng, category) for _ in range(n)],
        axis=0,
    ).astype(np.int16)


def alignment_gains(module, adapted_rdm: np.ndarray, base_rdm: np.ndarray, neural_vectors):
    adapted = module.upper(adapted_rdm)
    base = module.upper(base_rdm)
    return [
        float(
            module.spearmanr(adapted, neural).statistic
            - module.spearmanr(base, neural).statistic
        )
        for neural in neural_vectors
    ]


def score_target(module, inputs: dict, seeds: tuple[int, ...], permutation: np.ndarray | None):
    model_ensemble_scores: list[float] = []
    model_seed_scores: list[np.ndarray] = []

    for name in module.NEW_MODELS:
        spec = module.MODELS[name]
        ensemble_eeg: list[float] = []
        ensemble_meg: list[float] = []
        seed_eeg: list[list[float]] = [[] for _ in seeds]
        seed_meg: list[list[float]] = [[] for _ in seeds]

        for config in inputs["configs"][name]:
            train = config["object_fold"]["train"]
            test = config["object_fold"]["test"]
            eeg_group = config["eeg_group"]
            meg_group = config["meg_group"]
            if permutation is not None:
                eeg_group = eeg_group[np.ix_(permutation, permutation)]
                meg_group = meg_group[np.ix_(permutation, permutation)]
            target = module.consensus_target(eeg_group, meg_group, train)

            embeddings: list[np.ndarray] = []
            for seed_index, seed in enumerate(seeds):
                model = module.fit_dynamic_adapter(
                    inputs["features"][name][train, None, :],
                    target,
                    spec["width"],
                    spec["bottleneck"],
                    seed,
                )
                embedding = module.adapted_embeddings(
                    model, inputs["features"][name][test, None, :]
                )
                embeddings.append(embedding)
                seed_rdm = module.cosine_rdm(embedding)
                eeg_gain = alignment_gains(
                    module, seed_rdm, config["base_rdm"], config["eeg_neural"]
                )
                meg_gain = alignment_gains(
                    module, seed_rdm, config["base_rdm"], config["meg_neural"]
                )
                seed_eeg[seed_index].extend(eeg_gain)
                seed_meg[seed_index].extend(meg_gain)
                del model

            ensemble = np.mean(embeddings, axis=0)
            ensemble /= np.maximum(np.linalg.norm(ensemble, axis=1, keepdims=True), 1e-12)
            ensemble_rdm = module.cosine_rdm(ensemble)
            eeg_gain = alignment_gains(
                module, ensemble_rdm, config["base_rdm"], config["eeg_neural"]
            )
            meg_gain = alignment_gains(
                module, ensemble_rdm, config["base_rdm"], config["meg_neural"]
            )
            ensemble_eeg.extend(eeg_gain)
            ensemble_meg.extend(meg_gain)

        model_ensemble_scores.append(
            0.5 * (float(np.mean(ensemble_eeg)) + float(np.mean(ensemble_meg)))
        )
        model_seed_scores.append(
            np.asarray(
                [
                    0.5 * (float(np.mean(seed_eeg[i])) + float(np.mean(seed_meg[i])))
                    for i in range(len(seeds))
                ],
                dtype=np.float64,
            )
        )

    ensemble_score = float(np.mean(model_ensemble_scores))
    per_seed_score = np.mean(np.stack(model_seed_scores, axis=0), axis=0)
    return ensemble_score, per_seed_score


def locked_observed_from_original(module, seeds: tuple[int, ...]) -> float:
    values = np.load(SOURCE_VALUES, allow_pickle=False)
    metric = "eeg_single" if len(seeds) == 1 else "eeg_gain"
    modality_scores = []
    for name in module.NEW_MODELS:
        key = name.lower().replace("-", "_")
        modality_scores.append(
            0.5 * (
                float(values[f"{key}_{metric}"].mean())
                + float(values[f"{key}_{metric.replace('eeg_', 'meg_')}"] .mean())
            )
        )
    return float(np.mean(modality_scores))


def summarize(
    n: int,
    completed: int,
    observed: float,
    null: np.ndarray,
    seed_null: np.ndarray,
    started_at: float,
) -> dict:
    done = null[:completed]
    exceedances = int(np.sum(done >= observed)) if completed else 0
    elapsed = time.time() - started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining_seconds = (n - completed) / rate if rate > 0 else None
    payload = {
        "status": "complete" if completed == n else "running",
        "updated_at_utc": utc_now(),
        "n_requested": n,
        "n_completed": completed,
        "observed_three_seed_equal_modality_gain": observed,
        "exceedances_so_far": exceedances,
        "provisional_plus_one_p": (1 + exceedances) / (completed + 1) if completed else None,
        "elapsed_seconds_this_process": elapsed,
        "permutations_per_second_this_process": rate,
        "estimated_remaining_seconds": remaining_seconds,
    }
    if completed:
        payload.update(
            {
                "null_mean": float(done.mean()),
                "null_sd": float(done.std()),
                "null_95th": float(np.quantile(done, 0.95)),
                "seed_null_means": [float(x) for x in np.nanmean(seed_null[:completed], axis=0)],
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--stop-after", type=int, default=None)
    args = parser.parse_args()
    if args.n < 1:
        raise ValueError("--n must be positive")
    seeds = tuple(int(x) for x in args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = import_source()
    if not module.torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full permutation analysis")
    inputs = build_inputs(module)

    permutation_path = output / "permutations.npy"
    if permutation_path.exists():
        permutations = np.load(permutation_path, allow_pickle=False)
        if permutations.shape != (args.n, 72):
            raise RuntimeError(f"Existing permutation shape mismatch: {permutations.shape}")
    else:
        permutations = generate_permutations(module, inputs["category"], args.n)
        np.save(permutation_path, permutations, allow_pickle=False)

    checkpoint_path = output / "checkpoint.npz"
    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        null = checkpoint["ensemble_null"].astype(np.float64)
        seed_null = checkpoint["seed_null"].astype(np.float64)
        observed = float(checkpoint["observed_ensemble"])
        observed_by_seed = checkpoint["observed_by_seed"].astype(np.float64)
        if null.shape != (args.n,) or seed_null.shape != (args.n, len(seeds)):
            raise RuntimeError("Checkpoint dimensions do not match requested run")
    else:
        null = np.full(args.n, np.nan, dtype=np.float64)
        seed_null = np.full((args.n, len(seeds)), np.nan, dtype=np.float64)
        print("computing observed statistic with matched seed aggregation", flush=True)
        observed, observed_by_seed = score_target(module, inputs, seeds, permutation=None)
        locked_observed = locked_observed_from_original(module, seeds)
        if not np.isclose(observed, locked_observed, atol=1e-7):
            raise RuntimeError(
                f"Observed reproduction failed: recomputed={observed}, locked={locked_observed}"
            )
        atomic_npz(
            checkpoint_path,
            ensemble_null=null,
            seed_null=seed_null,
            observed_ensemble=np.asarray(observed),
            observed_by_seed=observed_by_seed,
        )

    completed_indices = np.flatnonzero(np.isfinite(null))
    completed = int(completed_indices[-1] + 1) if completed_indices.size else 0
    if completed and not np.isfinite(null[:completed]).all():
        raise RuntimeError("Checkpoint contains a gap")

    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        atomic_json(
            manifest_path,
            {
                "analysis": "extended neural-target specificity permutation",
                "created_at_utc": utc_now(),
                "n_permutations": args.n,
                "permutation_seed": PERMUTATION_SEED,
                "training_seeds": list(seeds),
                "seed_aggregation": "mean adapted embeddings, then cosine RDM",
                "test_statistic": "equal model mean of equal EEG-MEG mean alignment gains",
                "plus_one_correction": True,
                "within_category_joint_eeg_meg_permutation": True,
                "source_script": str(SOURCE_SCRIPT),
                "source_script_sha256": sha256(SOURCE_SCRIPT),
                "source_values": str(SOURCE_VALUES),
                "source_values_sha256": sha256(SOURCE_VALUES),
                "mapping": str(inputs["mapping"]),
                "mapping_sha256": sha256(inputs["mapping"]),
                "feature_indices_1based": (inputs["feature_indices"] + 1).tolist(),
                "models": list(module.NEW_MODELS),
                "participant_folds": 2,
                "image_folds": 3,
                "epochs": module.EPOCHS,
                "lambda_anchor": module.LAMBDA_ANCHOR,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": module.torch.__version__,
                "cuda_device": module.torch.cuda.get_device_name(0),
            },
        )

    started_at = time.time()
    status_path = output / "status.json"
    print(
        f"resume={completed}/{args.n}; observed={observed:+.9f}; seeds={list(seeds)}",
        flush=True,
    )
    for index in range(completed, args.n):
        ensemble_score, per_seed_score = score_target(
            module, inputs, seeds, permutations[index].astype(int)
        )
        null[index] = ensemble_score
        seed_null[index] = per_seed_score
        done = index + 1
        if done <= 3 or done % args.checkpoint_every == 0 or done == args.n:
            atomic_npz(
                checkpoint_path,
                ensemble_null=null,
                seed_null=seed_null,
                observed_ensemble=np.asarray(observed),
                observed_by_seed=observed_by_seed,
            )
            status = summarize(args.n, done, observed, null, seed_null, started_at)
            atomic_json(status_path, status)
            print(
                f"permutation {done}/{args.n}: null={ensemble_score:+.9f}; "
                f"exceed={status['exceedances_so_far']}; "
                f"p~={status['provisional_plus_one_p']:.6f}",
                flush=True,
            )
            if module.torch.cuda.is_available():
                module.torch.cuda.empty_cache()
        if args.stop_after is not None and done >= args.stop_after:
            print(f"intentional stop after {done}", flush=True)
            return

    final = summarize(args.n, args.n, observed, null, seed_null, started_at)
    final["status"] = "complete"
    final["completed_at_utc"] = utc_now()
    atomic_json(output / "results.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
