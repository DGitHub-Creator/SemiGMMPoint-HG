# ODPT-HG rank-gate full-run results: 10% and 20%

Date: 2026-08-11

Note: results were produced by the dataset-adapted SemiGMMPoint implementation (reproduction/adaptation for ODPT-HG), not the paper authors' official code.

## Protocol

- Run ID: `official-rankgate-0ab9c71`
- Seed: 1; epochs: 100; final-epoch-only checkpoint selection.
- Training/calibration: Area 1 only. Area 3 was read only after training.
- Test: all 2,757,311 valid points in Area 3 conference rooms 20/21/22.
- Initialization: existing `experiments/odpt_hg/pretrain/checkpoints/final.pth`.
  Pretraining was not repeated because the confidence gate and experiment
  protocol changes did not alter the pretraining objective or data.
- Rank-gate fractions: `[0.012, disabled, 0.993, disabled, 0.090, 0.010]`.

## Main results

| Budget | Labeled scenes | mIoU | mAcc | OA | Peak train mIoU |
|---|---:|---:|---:|---:|---:|
| 10% | 2 | 45.58% | 62.42% | 76.84% | 52.79% |
| 20% | 3 | 46.44% | 67.73% | 75.69% | 56.44% |
| 20% - 10% | +1 | +0.86 pp | +5.31 pp | -1.15 pp | +3.65 pp |

The unsupervised coverage of the full run remained finite and nonzero throughout:

- 10%: 6.22% to 22.86%, final 9.88%; finite-feature ratio 1.0000.
- 20%: 6.00% to 19.41%, final 7.61%; finite-feature ratio 1.0000.

## Per-class comparison

| Class | 10% IoU | 20% IoU | Delta | 10% Acc | 20% Acc | Delta |
|---|---:|---:|---:|---:|---:|---:|
| pipeline | 46.89% | 39.81% | -7.08 pp | 62.20% | 46.73% | -15.47 pp |
| steel frame | 16.66% | 18.22% | +1.56 pp | 24.15% | 24.47% | +0.32 pp |
| elbow pipe | 61.95% | 64.34% | +2.39 pp | 94.24% | 83.57% | -10.67 pp |
| valve guardrail | 7.54% | 16.00% | +8.46 pp | 15.49% | 62.65% | +47.16 pp |
| gate valve | 52.99% | 51.11% | -1.88 pp | 87.66% | 97.44% | +9.78 pp |
| Christmas tree body | 87.45% | 89.17% | +1.72 pp | 90.76% | 91.52% | +0.76 pp |

## Interpretation

The extra labeled scene improves class balance rather than point-weighted
accuracy. Valve-guardrail recall rises sharply, which drives the +5.31 pp mAcc
gain. It also over-segments pipeline: GT pipeline points predicted as
valve-guardrail increase from 64,588 (10%) to 175,423 (20%). That one confusion
explains much of the pipeline loss and the -1.15 pp OA change.

Elbow-pipe accuracy falls, but its IoU improves because false positives fall
more strongly: GT pipeline predicted as elbow drops from 80,662 to 38,238.
Gate-valve recall reaches 97.44%, while false positives from tree body and steel
frame rise, so its IoU falls slightly despite the recall gain.

Steel frame remains the weakest stable class (18.22% IoU at 20%). The next
experiment should target pipeline/guardrail separation and steel-frame
features, not globally increase pseudo-label coverage. In particular, enabling
the two currently disabled pseudo classes without a higher-precision mechanism
would be unsafe: labeled-set calibration accuracy was poor for valve guardrail
and only marginal for most steel-frame predictions.

## Relation to the earlier result

The earlier recorded 10% result (14.63% mIoU, 38.31% mAcc, 28.20% OA) used a
faulty pseudo-label protocol in which low-confidence points still entered the
unsupervised objective. The new 10% values improve by +30.95 pp mIoU, +24.11 pp
mAcc and +48.64 pp OA, but this is a protocol correction, not a fair
apples-to-apples model improvement claim.

## Reproducibility artifacts

- `experiments/odpt_hg/10pct/eval/official-rankgate-0ab9c71/`
- `experiments/odpt_hg/20pct/eval/official-rankgate-0ab9c71/`
- `experiments/odpt_hg/summary.csv`
- `experiments/odpt_hg/summary.txt`
- `reports/calibration/gate_stability35_split10.json`
- `reports/calibration/rankgate_stability35_split20.json`
- `reports/HG_10PCT_RANK_GATE_STABILITY.md`
