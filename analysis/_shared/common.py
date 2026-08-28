from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[2]

TABLE = Path(os.environ.get("THINGS_CONCEPT_TABLE", REPO / "data" / "things_concepts.csv"))
DINO = Path(os.environ.get("THINGS_DINO_FEATURES", REPO / "data" / "things_dinov3.npy"))
SPOSE = Path(os.environ.get("SPOSE_EMBEDDING", REPO / "data" / "spose_embedding.txt"))
THINGS_DIR = Path(os.environ.get("THINGS_EEG_PATTERNS", REPO / "data" / "things_eeg_patterns"))
ALLJOINED_DIR = Path(os.environ.get("ALLJOINED_EEG_PATTERNS", REPO / "data" / "alljoined_patterns"))

A_SLOTS = np.array([0, 2, 4, 6, 8])
B_SLOTS = np.array([1, 3, 5, 7, 9])
SEEDS = [20260722, 20260723, 20260724]
LAMBDAS = [0.0, 0.01, 0.1, 1.0]


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_table() -> pd.DataFrame:
    df = pd.read_csv(TABLE).sort_values("concept_id").reset_index(drop=True)
    required = {
        "concept_id", "official_things_row", "unique_id", "word",
        "category", "allocation", "development_fold",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing table columns: {sorted(missing)}")
    if len(df) != 884 or df["concept_id"].duplicated().any():
        raise RuntimeError("Frozen concept table is not the expected unique 884-row table")
    return df


def split_indices(df: pd.DataFrame):
    dev = np.flatnonzero(df["allocation"].astype(str).str.lower().eq("development"))
    conf = np.flatnonzero(~df["allocation"].astype(str).str.lower().eq("development"))
    folds = df.loc[dev, "development_fold"].astype(int).to_numpy()
    if len(dev) != 701 or len(conf) != 183 or set(np.unique(folds)) != {0, 1, 2, 3}:
        raise RuntimeError(
            f"Unexpected frozen split: development={len(dev)}, confirmation={len(conf)}, folds={sorted(set(folds))}"
        )
    return dev, conf, folds


def load_dino() -> np.ndarray:
    x = np.load(DINO).reshape(884, 10, 384).astype(np.float32)
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite DINO input")
    return x


def load_spose(df: pd.DataFrame) -> np.ndarray:
    all_spose = np.loadtxt(SPOSE).astype(np.float32)
    rows = df["official_things_row"].astype(int).to_numpy()
    x = all_spose[rows]
    if x.shape != (884, 66):
        raise RuntimeError(f"Unexpected SPoSE shape {x.shape}")
    return x


def load_patterns(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("sub-*_patterns.npz"))
    if not files:
        raise RuntimeError(f"No pattern files in {folder}")
    out = []
    expected_ids = None
    for f in files:
        z = np.load(f)
        p = z["post"].astype(np.float32)
        ids = z["concept_ids"].astype(int)
        if p.shape[0:2] != (2, 884):
            raise RuntimeError(f"Unexpected post shape in {f}: {p.shape}")
        if expected_ids is None:
            expected_ids = ids
        elif not np.array_equal(ids, expected_ids):
            raise RuntimeError(f"Concept order mismatch in {f}")
        out.append(p)
    return np.stack(out), expected_ids


def correlation_rdm_matrix(patterns: np.ndarray) -> np.ndarray:
    """patterns: concepts x features; returns square correlation-distance RDM."""
    x = np.asarray(patterns, dtype=np.float64)
    x -= x.mean(axis=1, keepdims=True)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    d = 1.0 - x @ x.T
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32)


def cosine_rdm_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    d = 1.0 - x @ x.T
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32)


def participant_rdms(pattern_array: np.ndarray, half: int) -> np.ndarray:
    return np.stack([correlation_rdm_matrix(p[half]) for p in pattern_array])


def upper(matrix: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    m = matrix if idx is None else matrix[np.ix_(idx, idx)]
    iu = np.triu_indices(len(m), 1)
    return m[iu]


def group_rank_teacher(subject_rdms: np.ndarray, idx: np.ndarray) -> np.ndarray:
    rows = []
    for rdm in subject_rdms:
        v = upper(rdm, idx)
        rows.append(rankdata(v, method="average"))
    y = np.mean(rows, axis=0)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return y.astype(np.float32)


def spearman_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class ResidualAdapter(nn.Module):
    def __init__(self, width: int = 384, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(x))))
        return F.normalize(x + residual, dim=-1)


def torch_upper_cosine(concept_embeddings: torch.Tensor) -> torch.Tensor:
    z = F.normalize(concept_embeddings, dim=-1)
    d = 1.0 - z @ z.T
    iu = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    return d[iu[0], iu[1]]


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred - pred.mean()
    t = target - target.mean()
    corr = (p * t).sum() / (torch.linalg.vector_norm(p) * torch.linalg.vector_norm(t) + 1e-12)
    return 1.0 - corr


def fit_adapter(
    image_features: np.ndarray,
    target: np.ndarray,
    lambda_anchor: float,
    seed: int,
    epochs: int = 400,
    device: str | None = None,
) -> ResidualAdapter:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    model = ResidualAdapter().to(device)
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
def adapted_concept_embeddings(model: ResidualAdapter, image_features: np.ndarray) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    x = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    zi = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    zc = F.normalize(zi.mean(dim=1), dim=-1)
    return zc.cpu().numpy().astype(np.float32)


def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
