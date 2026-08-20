# ODPT-HG 10% class-rank gate stability

Date: 2026-08-11

## Protocol

- Training data: canonical 10% split, two labeled Area 1 scenes and thirteen
  unlabeled Area 1 scenes.
- Area 3 was not instantiated or evaluated during calibration or stability
  training.
- Initialization: `experiments/odpt_hg/pretrain/checkpoints/final.pth`.
- Stability horizon: 35 epochs, seed 1, two GPUs, final-epoch-only checkpoint.

## Why the absolute gate was rejected

The original scalar 0.75 gate was outside the observed six-class
LayerNorm/GMM softmax range (calibrated maximum 0.587641). Class-specific
absolute thresholds initially restored nonzero coverage, but the dynamically
updated GMM rescaled confidence after the first unlabeled epoch:

| Run | Unlabeled samples/epoch | Epoch 5 coverage | Epoch 10 coverage | Epoch 15 coverage |
|---|---:|---:|---:|---:|
| `classgate-stability-c7b7236` | 390 | 13.24% | 0.19% | not continued |
| `balanced-stability-ea3c4b1` | 65 | 11.19% | 1.35% | 0.05% |
| `bounded-stability-52e77b2` | 13 | 14.41% | 0.20% | not continued |

All three diagnostic runs remained finite. The failure persisted even after
limiting an unlabeled epoch to about two DDP steps, so optimizer-step imbalance
was not the sole cause. Absolute confidence was therefore rejected as an
unstable gate coordinate for the dynamic GMM.

## Scale-invariant class-rank gate

The replacement keeps the labeled-Area-1 precision budget as a top fraction
within each agreeing predicted class:

`[pipeline=0.012, steel_frame=disabled, elbow_pipe=0.993,
valve_guardrail=disabled, gate_valve=0.090, tree_body=0.010]`

These fractions correspond to the accepted/predicted counts at the calibrated
absolute thresholds in `reports/calibration/gate_stability35_split10.json`.
They do not use Area 3.

## Accepted stability run

Run: `rankgate-stability-0ab9c71`

| Unsupervised epoch | Coverage | Loss | Finite-feature ratio |
|---:|---:|---:|---:|
| 5 | 24.76% | 1.3672 | 1.0000 |
| 10 | 16.19% | 1.3944 | 1.0000 |
| 15 | 10.53% | 1.3237 | 1.0000 |
| 20 | 13.10% | 1.1400 | 1.0000 |
| 25 | 11.35% | 1.3015 | 1.0000 |
| 30 | 11.86% | 1.2242 | 1.0000 |
| 35 | 11.58% | 1.2544 | 1.0000 |

Coverage no longer collapsed toward zero. Disabled classes had zero accepted
pseudo labels at every logged epoch. The supervised training mIoU peaked at
44.68% (epoch 28), compared with 35.96% in the earlier 35-epoch run whose
absolute 0.75 gate accepted no pseudo labels.

Result: stability gate passed; proceed to the full 100-epoch run and only
then evaluate the final checkpoint on Area 3.
