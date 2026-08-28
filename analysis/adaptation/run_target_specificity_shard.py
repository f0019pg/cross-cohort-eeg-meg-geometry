from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASE_SCRIPT = Path(__file__).with_name("run_target_specificity_permutation.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_base():
    spec = importlib.util.spec_from_file_location("teacher_specificity_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(
    start: int,
    end: int,
    completed: int,
    observed: float,
    null: np.ndarray,
    seed_null: np.ndarray,
    started_at: float,
) -> dict:
    done = null[:completed]
    elapsed = time.time() - started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (len(null) - completed) / rate if rate > 0 else None
    payload = {
        "status": "complete" if completed == len(null) else "running",
        "updated_at_utc": utc_now(),
        "global_start_index_zero_based_inclusive": start,
        "global_end_index_zero_based_exclusive": end,
        "n_requested_shard": len(null),
        "n_completed_shard": completed,
        "next_global_index_zero_based": start + completed,
        "observed_three_seed_equal_modality_gain": observed,
        "elapsed_seconds_this_process": elapsed,
        "permutations_per_second_this_process": rate,
        "estimated_remaining_seconds": remaining,
    }
    if completed:
        payload.update(
            {
                "exceedances_so_far": int(np.sum(done >= observed)),
                "null_mean": float(done.mean()),
                "null_sd": float(done.std()),
                "seed_null_means": [
                    float(x) for x in np.nanmean(seed_null[:completed], axis=0)
                ],
            }
        )
    return payload


def redirect_logs(output: Path) -> None:
    stdout = (output / "run.stdout.log").open("a", encoding="utf-8", buffering=1)
    stderr = (output / "run.stderr.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = stdout
    sys.stderr = stderr


def register_worker_pid(output: Path) -> Path:
    pid_path = output / "worker.pid"
    pid_text = str(os.getpid())
    pid_path.write_text(pid_text, encoding="ascii")

    def cleanup() -> None:
        try:
            if pid_path.exists() and pid_path.read_text(encoding="ascii").strip() == pid_text:
                pid_path.unlink()
        except OSError:
            pass

    atexit.register(cleanup)
    return pid_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--permutations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260722, 20260723, 20260724])
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    redirect_logs(output)
    register_worker_pid(output)
    pause_path = output.parent / "pause.requested"
    if pause_path.exists():
        print(f"[{utc_now()}] pause already requested; exiting before input load", flush=True)
        return

    if args.start < 0 or args.end <= args.start:
        raise ValueError("Require 0 <= start < end")
    seeds = tuple(int(x) for x in args.seeds)
    permutations = np.load(args.permutations.resolve(), allow_pickle=False)
    if permutations.ndim != 2 or permutations.shape[1] != 72:
        raise RuntimeError(f"Unexpected permutation shape: {permutations.shape}")
    if args.end > len(permutations):
        raise ValueError(f"end={args.end} exceeds n={len(permutations)}")

    base = load_base()
    source = base.import_source()
    if not source.torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"[{utc_now()}] loading inputs for global range [{args.start}, {args.end})", flush=True)
    inputs = base.build_inputs(source)
    count = args.end - args.start
    checkpoint_path = output / "checkpoint.npz"

    if checkpoint_path.exists():
        # On Windows an open NpzFile keeps checkpoint.npz locked, which makes
        # the next atomic os.replace fail after a resumed run. Copy all arrays
        # inside a context manager so the read handle is closed before fitting.
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            null = checkpoint["ensemble_null"].astype(np.float64)
            seed_null = checkpoint["seed_null"].astype(np.float64)
            observed = float(checkpoint["observed_ensemble"])
            observed_by_seed = checkpoint["observed_by_seed"].astype(np.float64)
        if null.shape != (count,) or seed_null.shape != (count, len(seeds)):
            raise RuntimeError("Checkpoint dimensions do not match shard")
    else:
        null = np.full(count, np.nan, dtype=np.float64)
        seed_null = np.full((count, len(seeds)), np.nan, dtype=np.float64)
        print(f"[{utc_now()}] reproducing observed statistic", flush=True)
        observed, observed_by_seed = base.score_target(source, inputs, seeds, None)
        locked = base.locked_observed_from_original(source, seeds)
        if not np.isclose(observed, locked, atol=1e-7):
            raise RuntimeError(f"Observed reproduction failed: {observed} versus {locked}")
        atomic_npz(
            checkpoint_path,
            ensemble_null=null,
            seed_null=seed_null,
            observed_ensemble=np.asarray(observed),
            observed_by_seed=observed_by_seed,
            global_indices=np.arange(args.start, args.end, dtype=np.int32),
        )

    finite = np.flatnonzero(np.isfinite(null))
    completed = int(finite[-1] + 1) if finite.size else 0
    if completed and not np.isfinite(null[:completed]).all():
        raise RuntimeError("Shard checkpoint contains a gap")

    atomic_json(
        output / "manifest.json",
        {
            "analysis": "sharded extended neural-target specificity permutation",
            "created_or_resumed_at_utc": utc_now(),
            "global_start_index_zero_based_inclusive": args.start,
            "global_end_index_zero_based_exclusive": args.end,
            "master_permutations": str(args.permutations.resolve()),
            "training_seeds": list(seeds),
            "base_script": str(BASE_SCRIPT),
            "statistical_definition_changed": False,
        },
    )

    started_at = time.time()
    print(
        f"[{utc_now()}] resume={completed}/{count}; observed={observed:+.9f}; "
        f"global=[{args.start}, {args.end})",
        flush=True,
    )
    for local_index in range(completed, count):
        global_index = args.start + local_index
        ensemble, per_seed = base.score_target(
            source, inputs, seeds, permutations[global_index].astype(int)
        )
        null[local_index] = ensemble
        seed_null[local_index] = per_seed
        done = local_index + 1
        pause_requested = pause_path.exists()
        if (
            done <= 3
            or done % args.checkpoint_every == 0
            or done == count
            or pause_requested
        ):
            atomic_npz(
                checkpoint_path,
                ensemble_null=null,
                seed_null=seed_null,
                observed_ensemble=np.asarray(observed),
                observed_by_seed=observed_by_seed,
                global_indices=np.arange(args.start, args.end, dtype=np.int32),
            )
            status = summarize(
                args.start, args.end, done, observed, null, seed_null, started_at
            )
            if pause_requested:
                status["status"] = "paused"
                status["paused_at_utc"] = utc_now()
            atomic_json(output / "status.json", status)
            print(
                f"[{utc_now()}] local={done}/{count}; global={global_index}; "
                f"null={ensemble:+.9f}; rate={status['permutations_per_second_this_process']:.5f}/s",
                flush=True,
            )
            source.torch.cuda.empty_cache()
            if pause_requested:
                print(
                    f"[{utc_now()}] graceful pause after local={done}; "
                    f"global_next={args.start + done}",
                    flush=True,
                )
                return

    result = summarize(args.start, args.end, count, observed, null, seed_null, started_at)
    result["status"] = "complete"
    result["completed_at_utc"] = utc_now()
    atomic_json(output / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
