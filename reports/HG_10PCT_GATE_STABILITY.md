# ODPT-HG 10% pseudo-label gate stability run

- Date: 2026-08-11 (server UTC)
- Git commit used by the run: `cd64298a6aeb68afc2ec3d2f4c21a02310d10327`
- Run ID: `gate-stability-dae94d1`
- Epochs: 35
- Seed: 1
- GPUs: 2 x NVIDIA RTX 4090

## Protocol

- Labeled scenes: `Area_1_conferenceRoom_1`, `Area_1_conferenceRoom_5`.
- Unlabeled scenes: the other 13 Area 1 scenes; dataset labels returned to the
  trainer are all `255`.
- Area 3 validation/evaluation was disabled throughout this run.
- The existing HG pre-GMM checkpoint was loaded from
  `experiments/odpt_hg/pretrain/checkpoints/final.pth`.
- Pseudo-label acceptance required both augmented views to have confidence at
  least `0.75` and to predict the same class.

## Gate results

| Unsupervised epoch | Candidates | Accepted | Rejected | Coverage | Unsupervised loss | Finite feature ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 5  | 3,456,000 | 0 | 3,456,000 | 0.0000 | 0.0000 | 1.0000 |
| 10 | 3,456,000 | 0 | 3,456,000 | 0.0000 | 0.0000 | 1.0000 |
| 15 | 3,456,000 | 0 | 3,456,000 | 0.0000 | 0.0000 | 1.0000 |
| 20 | 3,456,000 | 0 | 3,456,000 | 0.0000 | 0.0000 | 1.0000 |
| 25 | 3,456,000 | 0 | 3,456,000 | 0.0000 | 0.0000 | 1.0000 |
| 30 | 3,456,000 | 0 | 3,456,000 | 0.0000 | 0.0000 | 1.0000 |
| 35 | 3,456,000 | 0 | 3,456,000 | 0.0000 | 0.0000 | 1.0000 |

The accepted/total accounting and reported coverage agree at every
unsupervised epoch. With no accepted points, the loss remained a
graph-connected differentiable zero and DDP training completed normally.

## Supervised training behavior

The supervised training mIoU was 29.33% at epoch 14, 30.55% at epoch 23,
34.94% at epoch 27, and peaked at 35.96% at epoch 32. Epochs 33 and 34 were
35.14% and 35.56%, respectively. This is materially more stable than the
invalid ungated run, whose supervised training mIoU stayed near 18-19% late
in training after low-confidence pseudo-label updates.

## Conclusion

The low-confidence confirmation-loop bug is fixed: rejected points no longer
enter the unsupervised CE, instance contrast, class-guided contrast, GMM, or
prototype update paths. However, the current GMM posterior never crosses the
joint 0.75 gate during the first 35 epochs, so this run is effectively
supervised-only during its nominal unsupervised epochs.

Do not start the official 100-epoch 10% run or any 20% run yet. The next
diagnostic should examine labeled-Area-1 fitting and GMM confidence
calibration/scale without using Area 3 to select a threshold. Re-running the
pre-GMM stage is not required solely because of this gate fix.

Checkpoint (diagnostic only):

`experiments/odpt_hg/10pct/checkpoints/gate-stability-dae94d1/final.pth`
