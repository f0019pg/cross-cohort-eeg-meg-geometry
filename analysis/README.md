# Analysis entry points

The canonical analyses are grouped by scientific role. Shared functions live in `_shared/`; they are imported by the public entry-point scripts and are not separate analyses.

- `temporal/`: EEG reliability, early-to-later residuals, time-resolved correspondence and bidirectional source-cohort RSA.
- `adaptation/`: parameter-matched residual adapters, single-measurement target ablation and category-preserving target permutation.
- `external/`: THINGS/Alljoined/SPoSE evaluation and paired NOD EEG–MEG transfer.
- `robustness/`: low-level and model controls, crossed resampling, adapter stability and leave-one-out analyses.

Each script writes new output below `derived/`. It does not modify `results/reported/`, which records the values used in the manuscript.
