from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    json_files = sorted(ROOT.glob("results/reported/**/*.json"))
    for path in json_files:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)

    array_files = sorted(ROOT.glob("source_data/**/*.np[yz]"))
    for path in array_files:
        value = np.load(path, allow_pickle=False)
        if isinstance(value, np.lib.npyio.NpzFile):
            for key in value.files:
                value[key]
            value.close()

    manifest_path = ROOT / "SOURCE_DATA_MANIFEST.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = ROOT / row["path"]
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Size mismatch: {row['path']}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {row['path']}")

    print(
        f"Validated {len(json_files)} JSON files, {len(array_files)} NumPy files "
        f"and {len(rows)} manifest entries."
    )


if __name__ == "__main__":
    main()
