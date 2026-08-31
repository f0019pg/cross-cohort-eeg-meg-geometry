# Numerical source data

This directory contains compact numerical data used to regenerate the reported figures and audit the reported analyses.

- `main/`: arrays and participant-level values for the main figures.
- `supplementary/`: robustness arrays, stimulus mapping, held-out absolute-alignment table and six representative stimuli.
- `model_features/`: frozen feature matrices used by the reported adapter and control analyses.
- `checkpoints/`: final DINOv3 residual-adapter checkpoints for the three reported seeds.

Raw EEG and MEG participant data are not redistributed. Obtain them from the original public releases listed in the repository README.

The feature manifest records the exact checkpoint identifiers, preprocessing and hashes used for DINOv3, CLIP and SigLIP. `supplementary/adapter_architecture_baselines.npz` and `supplementary/nod_directional_retraining.npz` preserve the participant-by-fold values for the post-hoc analyses, including null outcomes.
