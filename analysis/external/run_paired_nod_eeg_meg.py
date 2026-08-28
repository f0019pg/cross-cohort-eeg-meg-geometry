from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis" / "_shared"))

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import cheby1, sosfiltfilt
from scipy.stats import spearmanr

from common import ResidualAdapter


EEG_ROOT = Path(os.environ.get("NOD_EEG_ROOT", ROOT / "data" / "nod_eeg"))
MEG_ROOT = Path(os.environ.get("NOD_MEG_ROOT", ROOT / "data" / "nod_meg"))
FEATURES = Path(os.environ.get("NOD_DINOV3_FEATURES", ROOT / "data" / "nod_dinov3.npy"))
INDEX = Path(os.environ.get("NOD_TRIAL_INDEX", ROOT / "data" / "nod_trial_index.csv"))
CHECKPOINT_DIR = Path(os.environ.get("ADAPTER_CHECKPOINT_DIR", ROOT / "source_data" / "checkpoints"))
PROTOCOL = ROOT / "config" / "protocols" / "paired_nod.md"
AMENDMENT = ROOT / "config" / "protocols" / "paired_nod_input.md"
AMENDMENT_02 = ROOT / "config" / "protocols" / "paired_nod_duplicate_images.md"
AMENDMENT_03 = ROOT / "config" / "protocols" / "paired_nod_serialization.md"
OUTPUT = ROOT / "derived" / "paired_nod"
AUDIT = OUTPUT / "input_audit.json"
RESULTS = OUTPUT / "results.json"
ROWS = OUTPUT / "participant_results.csv"
CACHE = OUTPUT / "cache"
STATUS = OUTPUT / "status.md"

EEG_WINDOW = (0.192, 0.320)
MEG_WINDOW = (0.180, 0.300)
PRE_WINDOW = (-0.100, 0.000)
SEEDS = (20260722, 20260723, 20260724)
N_BOOT = 10_000
BOOT_SEED = 20260807
LOWPASS_HZ = 25.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def epoch_path(root: Path, subject: str, modality: str) -> Path:
    return root / "derivatives" / "preprocessed" / "epochs" / f"{subject}_{modality}_epo.fif"


def discover_subjects() -> list[str]:
    eeg = {p.name.split("_eeg_epo.fif")[0] for p in (EEG_ROOT / "derivatives" / "preprocessed" / "epochs").glob("sub-*_eeg_epo.fif")}
    meg = {p.name.split("_meg_epo.fif")[0] for p in (MEG_ROOT / "derivatives" / "preprocessed" / "epochs").glob("sub-*_meg_epo.fif")}
    return sorted(eeg & meg)


def metadata(ep) -> pd.DataFrame:
    if ep.metadata is None:
        raise RuntimeError("Epoch metadata are absent")
    md = ep.metadata.reset_index(drop=True).copy()
    for col in ("image_id", "class_id", "super_class"):
        if col not in md:
            raise RuntimeError(f"Required metadata column is absent: {col}")
        md[col] = md[col].astype(str)
    return md


def shared_design(subject: str, feature_index: pd.DataFrame) -> dict:
    eeg = mne.read_epochs(epoch_path(EEG_ROOT, subject, "eeg"), preload=False, verbose="error")
    meg = mne.read_epochs(epoch_path(MEG_ROOT, subject, "meg"), preload=False, verbose="error")
    emd = metadata(eeg)
    mmd = metadata(meg)
    six = feature_index[feature_index["subject"] == subject].drop_duplicates("image_id").copy()
    feature_ids = set(six["image_id"])
    common = sorted(set(emd["image_id"]) & set(mmd["image_id"]) & feature_ids)
    ee = emd[emd["image_id"].isin(common)].drop_duplicates("image_id").set_index("image_id")
    mm = mmd[mmd["image_id"].isin(common)].drop_duplicates("image_id").set_index("image_id")
    if not np.array_equal(ee.loc[common, "class_id"].to_numpy(), mm.loc[common, "class_id"].to_numpy()):
        raise RuntimeError(f"EEG/MEG class mismatch for {subject}")
    counts = ee.loc[common].groupby("class_id").size()
    eligible = len(common) >= 1000 and len(counts) == 1000 and int(counts.min()) >= 1
    record = {
        "subject": subject,
        "eligible": bool(eligible),
        "eeg_trials": int(len(emd)),
        "meg_trials": int(len(mmd)),
        "shared_feature_images": int(len(common)),
        "shared_classes": int(len(counts)),
        "images_per_class_min": int(counts.min()) if len(counts) else 0,
        "images_per_class_max": int(counts.max()) if len(counts) else 0,
        "eeg_sfreq": float(eeg.info["sfreq"]),
        "meg_sfreq": float(meg.info["sfreq"]),
        "eeg_tmin": float(eeg.tmin),
        "eeg_tmax": float(eeg.tmax),
        "meg_tmin": float(meg.tmin),
        "meg_tmax": float(meg.tmax),
        "eeg_channels": int(len(eeg.ch_names)),
        "meg_channels": int(len(meg.ch_names)),
        "eeg_highpass": float(eeg.info["highpass"]),
        "eeg_lowpass": float(eeg.info["lowpass"]),
        "meg_highpass": float(meg.info["highpass"]),
        "meg_lowpass": float(meg.info["lowpass"]),
        "shared_image_hash": stable_hash(common),
    }
    del eeg, meg
    return record


def input_audit(write: bool = True) -> dict:
    required = [PROTOCOL, AMENDMENT, AMENDMENT_02, AMENDMENT_03, FEATURES, INDEX]
    required.extend(CHECKPOINT_DIR / f"late_consensus_seed_{seed}.pt" for seed in SEEDS)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    feature_index = pd.read_csv(INDEX, dtype={"subject": str, "image_id": str, "class_id": str})
    features = np.load(FEATURES, mmap_mode="r")
    rows = [shared_design(subject, feature_index) for subject in discover_subjects()]
    eligible = [row["subject"] for row in rows if row["eligible"]]
    if len(eligible) < 12:
        raise RuntimeError(f"Fewer than 12 metadata-eligible paired participants: {len(eligible)}")
    result = {
        "status": "INPUT_CHECK_PASSED",
        "created_kst": datetime.now().astimezone().isoformat(),
        "analysis": "paired NOD EEG-MEG late-consensus adapter transfer",
        "discovered_subjects": discover_subjects(),
        "eligible_subjects": eligible,
        "rows": rows,
        "feature_shape": list(features.shape),
        "checkpoints": [
            {"file": f"late_consensus_seed_{seed}.pt", "sha256": sha256(CHECKPOINT_DIR / f"late_consensus_seed_{seed}.pt")}
            for seed in SEEDS
        ],
        "windows_seconds": {"eeg_late": EEG_WINDOW, "meg_late": MEG_WINDOW, "pre": PRE_WINDOW},
        "bandwidths": {"native": "released epoch header 0.1-40 Hz", "sensitivity_lowpass_hz": LOWPASS_HZ},
        "hashes": {"protocol": sha256(PROTOCOL), "amendment": sha256(AMENDMENT), "amendment_02": sha256(AMENDMENT_02), "amendment_03": sha256(AMENDMENT_03), "index": sha256(INDEX), "features": sha256(FEATURES), "script": sha256(Path(__file__))},
    }
    if write:
        AUDIT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def load_models() -> list[ResidualAdapter]:
    models = []
    for seed in SEEDS:
        p = CHECKPOINT_DIR / f"late_consensus_seed_{seed}.pt"
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        if int(ckpt.get("seed", seed)) != seed:
            raise RuntimeError(f"Checkpoint seed mismatch: {p}")
        model = ResidualAdapter()
        model.load_state_dict(ckpt["state_dict"])
        models.append(model.eval())
    return models


@torch.no_grad()
def adapt_ensemble(models: list[ResidualAdapter], features: np.ndarray, batch: int = 4096) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [model.to(device).eval() for model in models]
    rows = []
    for start in range(0, len(features), batch):
        x = torch.as_tensor(np.asarray(features[start:start + batch]), dtype=torch.float32, device=device)
        z = torch.stack([model(x) for model in models], dim=0).mean(dim=0)
        rows.append(F.normalize(z, dim=-1).cpu().numpy().astype(np.float32))
    for model in models:
        model.cpu()
    return np.concatenate(rows)


def grouped_mean(x: np.ndarray, inverse: np.ndarray, nclass: int = 1000) -> np.ndarray:
    out = np.zeros((nclass, x.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse, minlength=nclass).astype(np.float64)
    np.add.at(out, inverse, x)
    if np.any(counts < 1):
        raise RuntimeError("A shared class is empty")
    return np.asarray(out / counts[:, None], dtype=np.float32)


def grouped_mean_nd(x: np.ndarray, inverse: np.ndarray, ngroup: int) -> np.ndarray:
    out = np.zeros((ngroup,) + x.shape[1:], dtype=np.float64)
    counts = np.bincount(inverse, minlength=ngroup).astype(np.float64)
    np.add.at(out, inverse, x)
    if np.any(counts < 1):
        raise RuntimeError("An exact-image group is empty")
    shape = (ngroup,) + (1,) * (x.ndim - 1)
    return np.asarray(out / counts.reshape(shape), dtype=np.float32)


def correlation_rdm_vector(patterns: np.ndarray) -> np.ndarray:
    x = np.asarray(patterns, dtype=np.float32)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    d = 1.0 - x @ x.T
    return np.asarray(d[np.triu_indices(len(x), 1)], dtype=np.float32)


def cosine_rdm_vector(patterns: np.ndarray) -> np.ndarray:
    x = np.asarray(patterns, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    d = 1.0 - x @ x.T
    return np.asarray(d[np.triu_indices(len(x), 1)], dtype=np.float32)


def lowpass_epochs(data: np.ndarray, sfreq: float, batch: int = 128) -> np.ndarray:
    sos = cheby1(8, 0.5, LOWPASS_HZ, btype="lowpass", fs=sfreq, output="sos")
    out = np.empty_like(data, dtype=np.float32)
    for start in range(0, len(data), batch):
        out[start:start + batch] = sosfiltfilt(sos, data[start:start + batch], axis=-1).astype(np.float32)
    return out


def neural_vectors(data: np.ndarray, times: np.ndarray, inverse: np.ndarray, late_window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    pre = (times >= PRE_WINDOW[0] - 1e-9) & (times <= PRE_WINDOW[1] + 1e-9)
    late = (times >= late_window[0] - 1e-9) & (times <= late_window[1] + 1e-9)
    baseline = data[:, :, pre].mean(axis=2, keepdims=True)
    late_trial = (data[:, :, late] - baseline).reshape(len(data), -1)
    pre_trial = data[:, :, pre].reshape(len(data), -1)
    late_class = grouped_mean(late_trial, inverse)
    pre_class = grouped_mean(pre_trial, inverse)
    return correlation_rdm_vector(late_class), correlation_rdm_vector(pre_class)


def participant_design(subject: str, feature_index: pd.DataFrame):
    eeg = mne.read_epochs(epoch_path(EEG_ROOT, subject, "eeg"), preload=False, verbose="error")
    meg = mne.read_epochs(epoch_path(MEG_ROOT, subject, "meg"), preload=False, verbose="error")
    emd, mmd = metadata(eeg), metadata(meg)
    six = feature_index[feature_index["subject"] == subject].drop_duplicates("image_id").set_index("image_id")
    common = sorted(set(emd["image_id"]) & set(mmd["image_id"]) & set(six.index))
    common_map = {image_id: i for i, image_id in enumerate(common)}
    eidx = np.flatnonzero(emd["image_id"].isin(common).to_numpy())
    midx = np.flatnonzero(mmd["image_id"].isin(common).to_numpy())
    einverse = emd.iloc[eidx]["image_id"].map(common_map).to_numpy(dtype=int)
    minverse = mmd.iloc[midx]["image_id"].map(common_map).to_numpy(dtype=int)
    ee = emd[emd["image_id"].isin(common)].drop_duplicates("image_id").set_index("image_id").loc[common]
    mm = mmd[mmd["image_id"].isin(common)].drop_duplicates("image_id").set_index("image_id").loc[common]
    class_ids = ee["class_id"].to_numpy(dtype=str)
    if not np.array_equal(class_ids, mm["class_id"].to_numpy(dtype=str)):
        raise RuntimeError(f"Class alignment failed for {subject}")
    classes = sorted(np.unique(class_ids))
    if len(classes) != 1000:
        raise RuntimeError(f"Expected 1000 shared classes for {subject}, found {len(classes)}")
    cmap = {c: i for i, c in enumerate(classes)}
    inverse = np.asarray([cmap[c] for c in class_ids], dtype=int)
    supers = ee.groupby("class_id")["super_class"].first().reindex(classes).astype(str).to_numpy()
    frows = six.loc[common, "feature_row"].astype(int).to_numpy()
    return eeg, meg, eidx, midx, einverse, minverse, inverse, classes, supers, frows, common


def build_cache(subject: str, feature_index: pd.DataFrame, features: np.ndarray, models: list[ResidualAdapter], protocol_hash: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{subject}_paired_rdms_v001.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["protocol_hash"].item()) != protocol_hash:
            raise RuntimeError(f"Protocol hash mismatch in cache {path}")
        return path
    eeg, meg, eidx, midx, einverse, minverse, inverse, classes, supers, frows, common = participant_design(subject, feature_index)
    print(f"{subject}: shared images={len(common)}, classes={len(classes)}", flush=True)

    eeg_data = grouped_mean_nd(eeg.get_data(picks="eeg")[eidx].astype(np.float32), einverse, len(common))
    meg_data = grouped_mean_nd(meg.get_data(picks="mag")[midx].astype(np.float32), minverse, len(common))
    eeg_native_post, eeg_native_pre = neural_vectors(eeg_data, eeg.times, inverse, EEG_WINDOW)
    meg_native_post, meg_native_pre = neural_vectors(meg_data, meg.times, inverse, MEG_WINDOW)

    eeg_lp = lowpass_epochs(eeg_data, float(eeg.info["sfreq"]))
    eeg_lp_post, _ = neural_vectors(eeg_lp, eeg.times, inverse, EEG_WINDOW)
    del eeg_lp
    meg_lp = lowpass_epochs(meg_data, float(meg.info["sfreq"]))
    meg_lp_post, _ = neural_vectors(meg_lp, meg.times, inverse, MEG_WINDOW)
    del meg_lp, eeg_data, meg_data, eeg, meg

    image_x = np.asarray(features[frows], dtype=np.float32)
    base_class = grouped_mean(image_x, inverse)
    adapted_images = adapt_ensemble(models, image_x)
    adapted_class = grouped_mean(adapted_images, inverse)
    base = cosine_rdm_vector(base_class)
    adapted = cosine_rdm_vector(adapted_class)

    np.savez_compressed(
        path,
        protocol_hash=np.asarray(protocol_hash), subject=np.asarray(subject),
        classes=np.asarray(classes, dtype=str), superclasses=np.asarray(supers, dtype=str), shared_images=np.asarray(len(common)), shared_image_hash=np.asarray(stable_hash(common)),
        base=base, adapted=adapted,
        eeg_native_post=eeg_native_post, eeg_native_pre=eeg_native_pre,
        meg_native_post=meg_native_post, meg_native_pre=meg_native_pre,
        eeg_lowpass_post=eeg_lp_post, meg_lowpass_post=meg_lp_post,
    )
    return path


def rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def exact_signflip(values: np.ndarray, one_sided: bool = False) -> float:
    v = np.asarray(values, dtype=np.float64)
    observed = float(v.mean()) if one_sided else float(abs(v.mean()))
    count, total = 0, 0
    chunk = 65536
    for start in range(0, 1 << len(v), chunk):
        ids = np.arange(start, min(start + chunk, 1 << len(v)), dtype=np.uint64)
        bits = ((ids[:, None] >> np.arange(len(v), dtype=np.uint64)) & 1).astype(np.float64)
        means = ((bits * 2.0 - 1.0) @ v) / len(v)
        count += int(np.sum(means >= observed - 1e-15) if one_sided else np.sum(np.abs(means) >= observed - 1e-15))
        total += len(ids)
    return float(count / total)


def summary(values: np.ndarray, seed_offset: int = 0) -> dict:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOT_SEED + seed_offset)
    take = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    return {
        "mean": float(v.mean()), "median": float(np.median(v)), "positive_n": int(np.sum(v > 0)), "n": int(len(v)),
        "exact_two_sided_signflip_p": exact_signflip(v),
        "bootstrap_mean_95ci": [float(x) for x in np.quantile(v[take].mean(axis=1), [0.025, 0.975])],
        "values": v.tolist(),
    }


def participant_row(subject: str, cache: Path) -> dict:
    z = np.load(cache, allow_pickle=True)
    base, adapted = z["base"].astype(float), z["adapted"].astype(float)
    supers = z["superclasses"].astype(str)
    iu = np.triu_indices(len(supers), 1)
    same = supers[iu[0]] == supers[iu[1]]
    row = {"subject": subject, "shared_images": int(z["shared_images"].item()) if "shared_images" in z.files else int(len(z["classes"]))}
    for band in ("native", "lowpass"):
        for modality in ("eeg", "meg"):
            post = z[f"{modality}_{band}_post"].astype(float)
            row[f"{band}_{modality}_base_post"] = rho(base, post)
            row[f"{band}_{modality}_adapted_post"] = rho(adapted, post)
            row[f"{band}_{modality}_delta_post"] = row[f"{band}_{modality}_adapted_post"] - row[f"{band}_{modality}_base_post"]
            row[f"{band}_{modality}_within_delta"] = rho(adapted[same], post[same]) - rho(base[same], post[same])
        row[f"{band}_neural_eeg_meg_rho"] = rho(z[f"eeg_{band}_post"], z[f"meg_{band}_post"])
    for modality in ("eeg", "meg"):
        pre = z[f"{modality}_native_pre"].astype(float)
        row[f"native_{modality}_delta_pre"] = rho(adapted, pre) - rho(base, pre)
        row[f"native_{modality}_post_minus_pre"] = row[f"native_{modality}_delta_post"] - row[f"native_{modality}_delta_pre"]
    row["geometry_preservation"] = rho(base, adapted)
    row["native_joint_delta"] = 0.5 * (row["native_eeg_delta_post"] + row["native_meg_delta_post"])
    row["lowpass_joint_delta"] = 0.5 * (row["lowpass_eeg_delta_post"] + row["lowpass_meg_delta_post"])
    row["native_both_positive"] = bool(row["native_eeg_delta_post"] > 0 and row["native_meg_delta_post"] > 0)
    row["lowpass_both_positive"] = bool(row["lowpass_eeg_delta_post"] > 0 and row["lowpass_meg_delta_post"] > 0)
    return row


def run() -> dict:
    if RESULTS.exists():
        raise RuntimeError("Once-only result already exists")
    audit = input_audit(write=not AUDIT.exists()) if not AUDIT.exists() else json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["hashes"]["protocol"] != sha256(PROTOCOL) or audit["hashes"]["amendment"] != sha256(AMENDMENT) or audit["hashes"]["amendment_02"] != sha256(AMENDMENT_02) or audit["hashes"]["amendment_03"] != sha256(AMENDMENT_03):
        raise RuntimeError("Protocol or amendment changed after input audit")
    feature_index = pd.read_csv(INDEX, dtype={"subject": str, "image_id": str, "class_id": str})
    features = np.load(FEATURES, mmap_mode="r")
    models = load_models()
    for subject in audit["eligible_subjects"]:
        build_cache(subject, feature_index, features, models, audit["hashes"]["protocol"])
    rows = [participant_row(subject, CACHE / f"{subject}_paired_rdms_v001.npz") for subject in audit["eligible_subjects"]]
    tab = pd.DataFrame(rows)
    tab.to_csv(ROWS, index=False)
    n = len(tab)
    positive_75 = math.ceil(0.75 * n)
    paired_60 = math.ceil(0.60 * n)

    endpoints = {}
    seed_offset = 0
    for band in ("native", "lowpass"):
        for modality in ("eeg", "meg"):
            for name in ("delta_post", "within_delta"):
                endpoints[f"{band}_{modality}_{name}"] = summary(tab[f"{band}_{modality}_{name}"].to_numpy(), seed_offset)
                seed_offset += 1
        endpoints[f"{band}_joint_delta"] = summary(tab[f"{band}_joint_delta"].to_numpy(), seed_offset); seed_offset += 1
        endpoints[f"{band}_neural_eeg_meg_rho"] = summary(tab[f"{band}_neural_eeg_meg_rho"].to_numpy(), seed_offset); seed_offset += 1
    for modality in ("eeg", "meg"):
        endpoints[f"native_{modality}_delta_pre"] = summary(tab[f"native_{modality}_delta_pre"].to_numpy(), seed_offset); seed_offset += 1
        endpoints[f"native_{modality}_post_minus_pre"] = summary(tab[f"native_{modality}_post_minus_pre"].to_numpy(), seed_offset); seed_offset += 1
        endpoints[f"native_{modality}_pre_positive_one_sided_p"] = exact_signflip(tab[f"native_{modality}_delta_pre"].to_numpy(), one_sided=True)

    both_native = int(tab["native_both_positive"].sum())
    both_lowpass = int(tab["lowpass_both_positive"].sum())
    geometry_mean = float(tab["geometry_preservation"].mean())
    gates = {
        "G1_native_eeg": endpoints["native_eeg_delta_post"]["mean"] >= 0.005 and endpoints["native_eeg_delta_post"]["positive_n"] >= positive_75 and endpoints["native_eeg_delta_post"]["exact_two_sided_signflip_p"] < 0.05,
        "G2_native_meg": endpoints["native_meg_delta_post"]["mean"] >= 0.005 and endpoints["native_meg_delta_post"]["positive_n"] >= positive_75 and endpoints["native_meg_delta_post"]["exact_two_sided_signflip_p"] < 0.05,
        "G3_native_joint": endpoints["native_joint_delta"]["mean"] >= 0.005 and endpoints["native_joint_delta"]["exact_two_sided_signflip_p"] < 0.05 and both_native >= paired_60,
        "G4_native_temporal_specificity": all(
            abs(endpoints[f"native_{m}_delta_pre"]["mean"]) < 0.002
            and endpoints[f"native_{m}_pre_positive_one_sided_p"] >= 0.05
            and endpoints[f"native_{m}_post_minus_pre"]["mean"] > 0
            and endpoints[f"native_{m}_post_minus_pre"]["positive_n"] >= positive_75
            and endpoints[f"native_{m}_post_minus_pre"]["exact_two_sided_signflip_p"] < 0.05
            for m in ("eeg", "meg")
        ),
        "G5_native_within_superclass": all(
            endpoints[f"native_{m}_within_delta"]["mean"] >= 0.003
            and endpoints[f"native_{m}_within_delta"]["positive_n"] >= positive_75
            and endpoints[f"native_{m}_within_delta"]["exact_two_sided_signflip_p"] < 0.05
            for m in ("eeg", "meg")
        ),
        "G6_geometry_preservation": geometry_mean >= 0.95,
        "B1_lowpass_eeg": endpoints["lowpass_eeg_delta_post"]["mean"] > 0 and endpoints["lowpass_eeg_delta_post"]["exact_two_sided_signflip_p"] < 0.05,
        "B2_lowpass_meg": endpoints["lowpass_meg_delta_post"]["mean"] > 0 and endpoints["lowpass_meg_delta_post"]["exact_two_sided_signflip_p"] < 0.05,
        "B3_lowpass_joint_positive": both_lowpass >= paired_60,
        "B4_lowpass_within_superclass": all(
            endpoints[f"lowpass_{m}_within_delta"]["mean"] > 0
            and endpoints[f"lowpass_{m}_within_delta"]["exact_two_sided_signflip_p"] < 0.05
            for m in ("eeg", "meg")
        ),
    }
    native_keys = [k for k in gates if k.startswith("G")]
    band_keys = [k for k in gates if k.startswith("B")]
    if all(gates[k] for k in native_keys) and all(gates[k] for k in band_keys):
        decision = "PAIRED_CROSSMODAL_TRANSFER_BANDWIDTH_ROBUST"
    elif all(gates[k] for k in native_keys):
        decision = "PAIRED_CROSSMODAL_TRANSFER_NATIVE_ONLY"
    else:
        decision = "STOP_OR_LIMITED"
    result = {
        "analysis": "paired NOD EEG-MEG transfer of frozen Kaneshiro-Cichy late-consensus adapter",
        "created_kst": datetime.now().astimezone().isoformat(),
        "participants": audit["eligible_subjects"], "n": n,
        "positive_count_threshold": positive_75, "both_modalities_threshold": paired_60,
        "endpoints": endpoints,
        "native_both_positive_n": both_native, "lowpass_both_positive_n": both_lowpass,
        "geometry_preservation_mean_rho": geometry_mean,
        "gates": gates, "decision": decision,
        "claim_boundary": "Prospectively specified paired cross-modal addendum; NOD was used in prior analyses and this is not a wholly untouched independent confirmation.",
        "hashes": {"protocol": sha256(PROTOCOL), "audit": sha256(AUDIT), "script": sha256(Path(__file__)), "features": sha256(FEATURES)},
    }
    RESULTS.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--check-inputs", action="store_true")
    p.add_argument("--run-all", action="store_true")
    args = p.parse_args()
    if args.check_inputs:
        print(json.dumps(input_audit(True), indent=2, ensure_ascii=False))
    elif args.run_all:
        run()
    else:
        p.error("Choose --check-inputs or --run-all")


if __name__ == "__main__":
    main()
