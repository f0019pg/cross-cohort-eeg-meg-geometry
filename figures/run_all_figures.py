from __future__ import annotations

import subprocess
import sys
from pathlib import Path


FIGURE_SCRIPTS = [
    "make_figure1.py",
    "make_figure2.py",
    "make_figure3.py",
    "make_figure4.py",
    "make_figure5.py",
    "make_figure6.py",
    "make_figure7_and_supplement.py",
]


def main() -> None:
    figure_dir = Path(__file__).resolve().parent
    for script in FIGURE_SCRIPTS:
        subprocess.run([sys.executable, str(figure_dir / script)], check=True)


if __name__ == "__main__":
    main()
