# Adapter architecture sensitivity analysis

Protocol fixed on 2026-08-31 before inspecting architecture-comparison outcomes.

## Status

This is a post-hoc sensitivity analysis requested during manuscript audit. It does not replace the preregistered or protocol-locked primary analyses.

## Fixed design

- Source data: Kaneshiro EEG and Cichy MEG data used in the reported adapter analysis.
- Backbone: frozen DINOv3 features.
- Target: equal-weight late EEG and MEG consensus RDM.
- Evaluation: the reported two participant folds and three category-balanced image folds.
- Optimization: 400 epochs, AdamW, anchor weight 100, and seeds 20260722, 20260723 and 20260724.
- Outcomes: participant-level mean held-out EEG and MEG alignment gain relative to the frozen representation.

## Architectures

1. Nonlinear residual adapter: LayerNorm, down projection, GELU and up projection.
2. Linear residual adapter: the same LayerNorm and projections without GELU. This has exactly the same trainable parameter count as the nonlinear adapter.
3. Diagonal feature reweighting: one positive multiplicative weight per frozen DINOv3 feature, optimized with the same target and anchor loss.

All three architectures are reported. No architecture is selected after outcome inspection.

## Random-target control

The existing category-preserving target-specificity analysis trains the reported adapter on 9,999 shuffled neural targets. It is the random-target control for this analysis and is more informative than a single arbitrary random target, because it preserves the category structure and target marginal distribution.
