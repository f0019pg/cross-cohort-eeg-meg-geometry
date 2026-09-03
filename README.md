# Cross-cohort EEG–MEG object geometry

Code, numerical source data and reported derived results for the manuscript **“Image-level representational geometry corresponds across independent EEG and MEG studies.”**

The study tests whether object-relation structure recurs across independent EEG and MEG cohorts that viewed the same 72 photographs. It then evaluates how far this structure can supervise small residual adapters attached to frozen vision models. The repository preserves positive, null and boundary results, including the single-measurement target ablation, category-preserving target permutations and paired NOD EEG–MEG evaluation.

## Repository contents

```text
analysis/
  temporal/       EEG temporal geometry and EEG–MEG source analyses
  adaptation/     frozen-backbone adapter analyses and target specificity
  external/       THINGS, Alljoined and paired NOD evaluations
  robustness/     control models, resampling and stability analyses
  reporting/      manuscript tables and runtime reporting
  _shared/        shared RDM, adapter and statistical functions
config/           figure style, path template and dated analysis protocols
figures/          main and supplementary figure-generation scripts
source_data/      numerical figure data, model features and final checkpoints
results/reported/ machine-readable results reported in the manuscript
experimental/     non-canonical analyses, if added in future
```

Raw participant data are not redistributed. The included `source_data/` files are the numerical inputs required to regenerate the reported display items and to audit the reported estimates.
`SOURCE_DATA_MANIFEST.csv` records the byte size and SHA-256 digest of every released source-data and reported-result file.

Validate the machine-readable results, NumPy archives and manifest hashes with:

```bash
python tools/validate_release.py
```

## Public datasets

| Dataset | Role | Access |
|---|---|---|
| Kaneshiro EEG | Source EEG geometry; 10 participants, 72 photographs | [NEMAR nm000263](https://nemar.org/dataset/nm000263); [primary article](https://doi.org/10.1371/journal.pone.0135697) |
| Cichy MEG | Independent source MEG geometry; 16 participants, the same 72 photographs | See the data release cited in the manuscript and set `CICHY_MEG_FILE` to the downloaded RDM file |
| THINGS-EEG2 | External EEG and unseen-concept evaluation | [THINGS initiative](https://things-initiative.org/); [OpenNeuro ds003825](https://openneuro.org/datasets/ds003825) |
| Alljoined | Independent EEG acquisition | Dataset release cited in the manuscript |
| NOD-EEG | ImageNet-based EEG transfer | [OpenNeuro ds005811](https://openneuro.org/datasets/ds005811) |
| NOD-MEG | Paired ImageNet-based MEG transfer | [OpenNeuro ds005810](https://openneuro.org/datasets/ds005810) |
| THINGS / SPoSE | Images and human object-similarity structure | [THINGS initiative](https://things-initiative.org/) |

Use the original releases under their own licences and participant-data terms.
The six representative Kaneshiro stimulus thumbnails included for Fig. 1 are redistributed under the source article's [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/) with attribution to Kaneshiro et al. (2015).

## Environment

Python 3.12.13 was used for the final reproducibility checks. Create an environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The exact package versions used for the final checks are recorded in
`requirements-reproducibility.txt`. The PyTorch build listed there uses CUDA
12.8 and may require the corresponding PyTorch package index when recreating
the environment.

Copy `config/paths.example.env` to a local, untracked file or define the variables in your shell. Absolute local paths are deliberately excluded from tracked files.

## Regenerate the figures

The committed numerical source data are sufficient to regenerate the main figures without downloading raw participant data:

```bash
python figures/run_all_figures.py
```

Outputs are written to `generated_figures/`, which is ignored by Git. Individual scripts can also be run directly, for example:

```bash
python figures/make_figure4.py
```

## Rerun the analyses

The main entry points are:

```bash
python analysis/temporal/run_locked_stage0.py
python analysis/temporal/run_locked_stage1.py
python analysis/temporal/run_source_geometry.py --eeg-dir <KANESHIRO_DIR> --meg-file <CICHY_RDM_FILE>
python analysis/adaptation/run_multibackbone_adaptation.py
python analysis/adaptation/run_single_measurement_ablation.py
python analysis/adaptation/run_target_specificity_permutation.py
python analysis/external/run_external_things_nod.py
python analysis/external/run_paired_nod_eeg_meg.py
python analysis/robustness/run_adapter_architecture_baselines.py --eeg-dir <KANESHIRO_DIR> --meg-file <CICHY_RDM_FILE>
python analysis/external/run_nod_directional_retraining.py --cache-dir <PAIRED_NOD_CACHE> --features <NOD_DINOV3_FEATURES> --index <NOD_FEATURE_INDEX>
python analysis/reporting/build_heldout_alignment_table.py
python analysis/reporting/benchmark_adapter_runtime.py
```

These analyses require the public raw or preprocessed datasets at the paths declared in the environment. Dated protocol files in `config/protocols/` record the fixed windows, participant and image splits, random seeds, controls and decision rules. Newly generated results are written below `derived/` and do not overwrite the committed `results/reported/` audit records.

## Reproducibility notes

- Participant-level inference, bootstrap confidence intervals and permutation tests use the units and seeds recorded in the corresponding scripts and protocols.
- The Kaneshiro–Cichy mapping is fixed in `source_data/supplementary/stimulus_mapping.csv` and includes image hashes without personal file paths.
- Adapter checkpoints in `source_data/checkpoints/` correspond to the three reported DINOv3 optimization seeds.
- `results/reported/` contains the exact continuous effect estimates, confidence intervals, participant counts and null results used in the manuscript.
- `source_data/model_features/FEATURE_MANIFEST.json` records checkpoint identifiers, revisions, preprocessing and feature-file hashes for all three frozen backbones.
- The adapter-architecture and NOD-directional analyses are explicitly post-hoc; their dated protocols and protocol hashes are retained in `config/protocols/` and `results/reported/`.
- The manuscript should be treated as the authoritative description of the scientific design; the repository is its executable and numerical companion.

## Citation

Please cite the accompanying article. Citation metadata are provided in `CITATION.cff` and should be updated with the DOI after publication.

## Licence

Analysis and figure-generation code is released under the MIT License. Numerical data retain the terms of their original sources where applicable.
