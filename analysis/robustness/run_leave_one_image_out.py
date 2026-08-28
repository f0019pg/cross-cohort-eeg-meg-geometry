import json
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ROOT / "source_data" / "supplementary" / "core_rsa_robustness.npz"
OUT = ROOT / "derived" / "leave_one_image_out"


def partial_spearman_from_rank_residuals(a, b):
    # The saved arrays are already residuals of rank-transformed dissimilarities.
    # Pearson correlation of those residuals is therefore the partial Spearman statistic.
    return float(np.corrcoef(a, b)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ARRAYS)
    eeg = z["primary_eeg_residuals"]
    meg = z["primary_meg_residuals"]
    i, j = np.triu_indices(72, 1)
    rows = []
    for omitted in range(72):
        keep = (i != omitted) & (j != omitted)
        eeg_group = eeg[:, keep].mean(0)
        meg_group = meg[:, keep].mean(0)
        direct = partial_spearman_from_rank_residuals(eeg_group, meg_group)
        meg_to_eeg = np.mean([partial_spearman_from_rank_residuals(eeg[p, keep], meg_group) for p in range(eeg.shape[0])])
        eeg_to_meg = np.mean([partial_spearman_from_rank_residuals(meg[p, keep], eeg_group) for p in range(meg.shape[0])])
        rows.append({
            "omitted_image_index_zero_based": omitted,
            "direct_group_rho": direct,
            "meg_group_to_eeg_mean": float(meg_to_eeg),
            "eeg_group_to_meg_mean": float(eeg_to_meg),
        })
    def summary(key):
        x = np.array([r[key] for r in rows])
        return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "all_positive": bool(np.all(x > 0))}
    result = {
        "analysis": "leave-one-image-out influence on primary controlled late EEG-MEG correspondence",
        "status": "POST_OUTCOME_SENSITIVITY_NOT_PREREGISTERED",
        "n_images": 72,
        "pairwise_residuals": "DINOv3 and 21 category-pair controls fitted in the full 72-image sample",
        "summary": {
            "direct_group_rho": summary("direct_group_rho"),
            "meg_group_to_eeg_mean": summary("meg_group_to_eeg_mean"),
            "eeg_group_to_meg_mean": summary("eeg_group_to_meg_mean"),
        },
        "rows": rows,
    }
    path = OUT / "LEAVE_ONE_IMAGE_RESULTS_v001.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(path)


if __name__ == "__main__":
    main()
