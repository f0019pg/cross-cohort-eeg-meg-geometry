from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SHARD_SPECS = (
    ("shard4_00_0000_2500", 0, 2500),
    ("shard4_01_2500_5000", 2500, 5000),
    ("shard4_02_5000_7500", 5000, 7500),
    ("shard4_03_7500_9999", 7500, 9999),
)


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    ensemble = np.full(9999, np.nan, dtype=np.float64)
    seed_null = np.full((9999, 3), np.nan, dtype=np.float64)
    observed_values = []
    observed_by_seed_values = []

    for name, start, end in SHARD_SPECS:
        checkpoint_path = root / name / "checkpoint.npz"
        if not checkpoint_path.exists():
            raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        local = checkpoint["ensemble_null"].astype(np.float64)
        local_seed = checkpoint["seed_null"].astype(np.float64)
        indices = checkpoint["global_indices"].astype(int)
        expected = np.arange(start, end)
        if not np.array_equal(indices, expected):
            raise RuntimeError(f"Global-index mismatch in {name}")
        if local.shape != (end - start,) or local_seed.shape != (end - start, 3):
            raise RuntimeError(f"Array-shape mismatch in {name}")
        ensemble[start:end] = local
        seed_null[start:end] = local_seed
        observed_values.append(float(checkpoint["observed_ensemble"]))
        observed_by_seed_values.append(checkpoint["observed_by_seed"].astype(np.float64))

    if not np.isfinite(ensemble).all() or not np.isfinite(seed_null).all():
        missing = int(np.sum(~np.isfinite(ensemble)))
        raise RuntimeError(f"Shards are incomplete: {missing} ensemble nulls missing")
    if not np.allclose(observed_values, observed_values[0], atol=0.0, rtol=0.0):
        raise RuntimeError("Observed ensemble statistic differs across shards")
    if not all(
        np.array_equal(value, observed_by_seed_values[0])
        for value in observed_by_seed_values[1:]
    ):
        raise RuntimeError("Observed per-seed statistics differ across shards")

    observed = observed_values[0]
    exceedances = int(np.sum(ensemble >= observed))
    result = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_permutations": 9999,
        "n_training_seeds": 3,
        "observed_three_seed_equal_modality_gain": observed,
        "exceedances": exceedances,
        "plus_one_p": (1 + exceedances) / 10000,
        "null_mean": float(ensemble.mean()),
        "null_sd": float(ensemble.std()),
        "null_95th": float(np.quantile(ensemble, 0.95)),
        "null_99th": float(np.quantile(ensemble, 0.99)),
        "seed_null_means": [float(x) for x in seed_null.mean(axis=0)],
        "source_shards": [name for name, _, _ in SHARD_SPECS],
    }
    atomic_npz(
        root / "combined_checkpoint.npz",
        ensemble_null=ensemble,
        seed_null=seed_null,
        observed_ensemble=np.asarray(observed),
        observed_by_seed=observed_by_seed_values[0],
        global_indices=np.arange(9999, dtype=np.int32),
    )
    atomic_json(root / "combined_results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
