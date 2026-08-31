from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "source_data" / "main" / "figure5.npz"
OUT_CSV = REPO / "source_data" / "supplementary" / "heldout_absolute_alignment.csv"
OUT_JSON = REPO / "results" / "reported" / "heldout_absolute_alignment.json"

MODELS = (
    ("DINOv3", "dinov3"),
    ("CLIP", "clip_b32"),
    ("SigLIP", "siglip_base"),
)
MEASUREMENTS = (("EEG", "eeg"), ("MEG", "meg"))
N_BOOT = 10_000
SEED = 20260831


def summarize(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    samples = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    ci = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
        "participant_values": values.tolist(),
    }


def main() -> None:
    arrays = np.load(SOURCE, allow_pickle=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for model_label, model_key in MODELS:
        for measurement_label, measurement_key in MEASUREMENTS:
            prefix = f"{model_key}_{measurement_key}"
            frozen = arrays[f"{prefix}_base"].mean(axis=1)
            adapted = arrays[f"{prefix}_adapted"].mean(axis=1)
            gain = arrays[f"{prefix}_gain"].mean(axis=1)
            if not np.allclose(adapted - frozen, gain, atol=1e-12, rtol=0):
                raise RuntimeError(f"Gain mismatch for {model_label} {measurement_label}")
            rows.append(
                {
                    "backbone": model_label,
                    "measurement": measurement_label,
                    "n_participants": int(len(frozen)),
                    "frozen": summarize(frozen, rng),
                    "adapted": summarize(adapted, rng),
                    "gain": summarize(gain, rng),
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "backbone",
                "measurement",
                "n_participants",
                "frozen_mean_rho",
                "frozen_ci95_lower",
                "frozen_ci95_upper",
                "adapted_mean_rho",
                "adapted_ci95_lower",
                "adapted_ci95_upper",
                "gain_mean_delta_rho",
                "gain_ci95_lower",
                "gain_ci95_upper",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["backbone"],
                    row["measurement"],
                    row["n_participants"],
                    row["frozen"]["mean"],
                    *row["frozen"]["ci95"],
                    row["adapted"]["mean"],
                    *row["adapted"]["ci95"],
                    row["gain"]["mean"],
                    *row["gain"]["ci95"],
                ]
            )

    payload = {
        "analysis": "held-out absolute neural alignment",
        "source": str(SOURCE.relative_to(REPO)).replace("\\", "/"),
        "unit": "participant mean across three held-out image folds",
        "confidence_intervals": {
            "method": "percentile bootstrap over participants",
            "draws": N_BOOT,
            "seed": SEED,
        },
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUT_CSV)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
