from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.adaptation.run_multibackbone_adaptation import MODELS, fit_dynamic_adapter


def gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def cpu_name() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return platform.processor()


def benchmark_model(name: str, repeats: int, epochs: int) -> dict:
    spec = MODELS[name]
    features = np.load(spec["path"]).astype(np.float32)[:48]
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    n_pairs = features.shape[0] * (features.shape[0] - 1) // 2
    rng = np.random.default_rng(20260831)
    target = rng.standard_normal(n_pairs).astype(np.float32)
    target = (target - target.mean()) / target.std(ddof=0)

    fit_dynamic_adapter(
        features[:, None, :], target, spec["width"], spec["bottleneck"],
        seed=20260831, epochs=10,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for repeat in range(repeats):
        start = time.perf_counter()
        model = fit_dynamic_adapter(
            features[:, None, :], target, spec["width"], spec["bottleneck"],
            seed=20260831 + repeat, epochs=epochs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "model": name,
        "width": int(spec["width"]),
        "bottleneck": int(spec["bottleneck"]),
        "train_images": int(features.shape[0]),
        "epochs": int(epochs),
        "repeats": int(repeats),
        "seconds": times,
        "mean_seconds": statistics.fmean(times),
        "sd_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reported" / "adapter_runtime_benchmark.json",
    )
    args = parser.parse_args()

    results = [benchmark_model(name, args.repeats, args.epochs) for name in MODELS]
    payload = {
        "purpose": "Wall-clock benchmark using the reported architecture, 48-image training fold and optimization schedule",
        "note": "A fixed synthetic target was used because target values do not change the computational graph.",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu_name(),
            "cpu": cpu_name(),
        },
        "models": results,
        "mean_seconds_across_models": statistics.fmean(r["mean_seconds"] for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
