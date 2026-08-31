# Exploratory NOD directional retraining

Protocol fixed on 2026-08-31 before inspecting directional outcomes.

## Status and claim boundary

This analysis was requested during manuscript audit and is explicitly post-hoc. NOD outcomes had already been examined. It tests whether a target estimated from one NOD recording can guide alignment with the other recording under participant- and concept-disjoint evaluation; it is not an untouched external replication.

## Fixed design

- Participants: all 19 metadata-eligible participants with paired NOD EEG and MEG.
- Neural inputs: the native-bandwidth late class-level RDMs already sealed for the paired addendum (EEG 192–320 ms; MEG 180–300 ms).
- Participant folds: alternating participants form the teacher group and the complementary group is evaluated; assignments are then reversed.
- Concept folds: ImageNet classes are sorted within each released superclass and assigned round-robin to four folds. Three folds train the adapter and one disjoint fold is evaluated.
- Model inputs: frozen DINOv3 class-centroid features, with images averaged within class before training or evaluation.
- Directions: EEG teacher to held-out MEG participants and MEG teacher to held-out EEG participants.
- Adapter: reported DINOv3 residual adapter, 384→64→384, anchor weight 100, 400 epochs and seeds 20260722, 20260723 and 20260724.
- Optimization pairs: 10,000 training-class pairs sampled without replacement using seed 20260831 plus the concept-fold index. The same pairs are used for both directions.
- Outcome: participant-level Spearman alignment gain, adapted minus frozen, averaged across the four held-out concept folds.
- Inference: exact two-sided participant sign-flip test and 10,000 participant bootstrap resamples.

No fold, seed, pair sample, window or preprocessing choice is changed after outcome inspection.
