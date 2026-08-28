from __future__ import annotations

import sys
from pathlib import Path


SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED))

from run_three_teacher_matched_ablation_v001 import main


if __name__ == "__main__":
    main()
