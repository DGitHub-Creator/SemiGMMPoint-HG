#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GudiePointContrastLoss edge-case tests (single process).

Covers the dimension-boundary crash that took down the official ODPT run:
the guide CE was fed `out = logits.squeeze()` which collapses [1, 1] -> []
when K == 1 (a single unique pseudo-label class), raising
`IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)`.

Run:  python tools/test_odpt_guide_loss_edges.py
"""
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO_ROOT)

from openpoints.loss.gudie_point_contrast_loss import GudiePointContrastLoss  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PASS, FAIL = 0, 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('  [PASS] %s %s' % (name, detail))
    else:
        FAIL += 1
        print('  [FAIL] %s %s' % (name, detail))


def make_inputs(B, N, C=64, num_classes=6, target_mode='multi', seed=0, require_grad=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    feats = torch.randn(B, C, N, generator=g, device=DEV)
    seg = torch.randn(B, num_classes, N, generator=g, device=DEV)
    if require_grad:
        feats.requires_grad_(True)
    t = torch.full((B, N), 255, dtype=torch.long, device=DEV)  # unlabeled views
    if target_mode == 'multi':
        t[:, :] = torch.randint(0, num_classes, (B, N), generator=g, device=DEV)
    elif target_mode == 'single_class':
        t[:, :] = 2
    elif target_mode == 'distinct':
        # every class appears <3 times -> deterministic guide K=0
        t[:, :] = torch.arange(N, device=DEV).unsqueeze(0).expand(B, N) % num_classes
    return feats, seg, seg.clone(), t, t.clone()


def run_case(name, B, N, seed=0, seg_fn=None, expect_error=None, label_error=None,
             target_mode='multi', expected_accepted=None, expected_coverage=None,
             confidence_threshold=0.75, class_acceptance_fraction=None):
    """seg_fn(seg) -> seg mutated for a specific pseudo-label structure."""
    global FAIL
    feats, seg1, seg2, t1, t2 = make_inputs(B, N, seed=seed, target_mode=target_mode)
    if label_error is not None:
        t1[:, 0] = label_error
        t2[:, 0] = label_error
    if seg_fn is not None:
        seg_fn(seg1)
        seg2.data.copy_(seg1)
    seg1.requires_grad_(True)
    seg2.requires_grad_(True)
    loss_fn = GudiePointContrastLoss(npos=4096, T=0.4, label_smoothing=0.1,
                                     is_guide=False, ignore_index=255,
                                     max_guide_iter=10, num_classes=6,
                                     confidence_threshold=confidence_threshold,
                                     class_acceptance_fraction=
                                     class_acceptance_fraction).to(DEV)
    loss_fn._epoch, loss_fn._iteration, loss_fn._rank = 1, 0, 0
    try:
        out = loss_fn(feats, feats, seg1, seg2, t1, t2)
    except SystemExit:
        print('  [FAIL] %s: diagnostic os._exit fired unexpectedly' % name)
        FAIL += 1
        return
    except Exception as e:
        if expect_error is not None and isinstance(e, expect_error):
            print('  [PASS] %s: expected error raised: %s: %s' % (name, type(e).__name__, str(e)[:140]))
            global PASS
            PASS += 1
            return
        print('  [FAIL] %s: unexpected %s: %s' % (name, type(e).__name__, str(e)[:200]))
        traceback.print_exc()
        FAIL += 1
        return
    if expect_error is not None:
        print('  [FAIL] %s: expected error %s but forward succeeded' % (name, expect_error.__name__))
        FAIL += 1
        return
    check('%s loss is scalar' % name, out.dim() == 0)
    check('%s loss is finite' % name, bool(torch.isfinite(out).item()), 'loss=%.4f' % out.item())
    try:
        out.backward()
        g1 = feats.grad
        g2 = seg1.grad
        check('%s backward ok' % name, True)
        check('%s feats grad finite' % name,
              g1 is not None and bool(torch.isfinite(g1).all().item()))
        check('%s seg grad finite' % name,
              g2 is not None and bool(torch.isfinite(g2).all().item()))
    except Exception as e:
        print('  [FAIL] %s: backward raised %s: %s' % (name, type(e).__name__, str(e)[:200]))
        FAIL += 1
    # CE contract held on the last guide step if any
    stats = loss_fn.pl_stats or {}
    check('%s pl_stats present' % name, isinstance(stats, dict))
    if expected_accepted is not None:
        check('%s accepted count' % name,
              stats.get('pseudo_gate_accepted') == expected_accepted,
              'got=%s expected=%s' %
              (stats.get('pseudo_gate_accepted'), expected_accepted))
    if expected_coverage is not None:
        check('%s coverage' % name,
              abs(stats.get('coverage', -1.0) - expected_coverage) < 1e-7,
              'got=%s expected=%s' %
              (stats.get('coverage'), expected_coverage))


def one_class_pseudo(seg):
    # all logits peak at class 2 -> pseudo labels collapse to a single class (K=1)
    seg.fill_(-3.0)
    seg[:, 2, :] = 3.0


def one_point_per_class(seg):
    # K=0 guide: every class present but with <3 points each -> min_iter=0
    pass  # handled by target_mode='multi' with tiny N


def multi_class_pseudo(seg):
    # several classes above the threshold -> K>1
    seg.fill_(-2.0)
    seg[:, 0, :] = 2.0
    seg[:, 3, :] = 2.0
    seg[:, 5, :] = 2.0


def low_confidence_pseudo(seg):
    # Uniform posterior: no point reaches the 0.75 gate.
    seg.zero_()


def exactly_one_confident_pseudo(seg):
    seg.zero_()
    seg[:, :, 0] = -8.0
    seg[:, 2, 0] = 8.0


def main():
    global PASS, FAIL
    print('GudiePointContrastLoss edge tests (device=%s)\n' % DEV)
    print('--- K=1 guide (the crash: single unique pseudo-label class) ---')
    run_case('K=1 guide (B=1)', 1, 8192, seg_fn=one_class_pseudo,
             target_mode='unlabeled', expected_accepted=4096,
             expected_coverage=1.0)
    run_case('K=1 guide (B=4)', 4, 8192, seg_fn=one_class_pseudo,
             target_mode='unlabeled', expected_accepted=16384,
             expected_coverage=1.0)

    print('--- confidence gate boundaries ---')
    run_case('accepted=0', 1, 32, seg_fn=low_confidence_pseudo,
             target_mode='unlabeled', expected_accepted=0,
             expected_coverage=0.0)
    run_case('accepted=1', 1, 32, seg_fn=exactly_one_confident_pseudo,
             target_mode='unlabeled', expected_accepted=1,
             expected_coverage=1.0 / 16.0)
    run_case('classwise accepted class', 1, 32, seg_fn=one_class_pseudo,
             target_mode='unlabeled', expected_accepted=16,
             expected_coverage=1.0,
             confidence_threshold=[None, None, 0.55, None, None, None])
    run_case('classwise disabled class', 1, 32, seg_fn=one_class_pseudo,
             target_mode='unlabeled', expected_accepted=0,
             expected_coverage=0.0,
             confidence_threshold=[None, None, None, None, None, None])
    run_case('rank gate top quarter', 1, 32, seg_fn=one_class_pseudo,
             target_mode='unlabeled', expected_accepted=4,
             expected_coverage=0.25,
             class_acceptance_fraction=[None, None, 0.25, None, None, None])
    run_case('rank gate disabled class', 1, 32, seg_fn=one_class_pseudo,
             target_mode='unlabeled', expected_accepted=0,
             expected_coverage=0.0,
             class_acceptance_fraction=[None, None, None, None, None, None])

    print('--- K=0 guide (every class <3 points -> CE skipped, differentiable zero) ---')
    run_case('K=0 guide (B=1, N=4 distinct)', 1, 4, target_mode='distinct')
    run_case('K=0 guide (B=4, N=8 distinct)', 4, 8, target_mode='distinct')

    print('--- K>1 ---')
    run_case('K>1 guide multi-class (B=4)', 4, 8192,
             seg_fn=multi_class_pseudo, target_mode='unlabeled')
    run_case('K>1 real labels (B=4)', 4, 8192, seed=3)

    print('--- NCE / contrast pair edges ---')
    run_case('NCE K=2 (tiny N)', 1, 4, seed=1)
    run_case('NCE K=4096 (B=1)', 1, 8192, seed=2)
    run_case('NCE K=4096 (B=4)', 4, 8192, seed=4)

    print('--- all labels 255 (pure pseudo-label path) ---')
    run_case('all-255 labels (B=4)', 4, 8192, seed=5,
             target_mode='unlabeled')

    print('--- non-contiguous inputs ---')
    feats, seg1, seg2, t1, t2 = make_inputs(2, 2048, seed=6)
    seg1 = seg1.transpose(1, 2).contiguous().transpose(1, 2)  # force non-contig
    seg2 = seg2.transpose(1, 2).contiguous().transpose(1, 2)
    loss_fn = GudiePointContrastLoss(npos=4096, T=0.4, is_guide=False,
                                     ignore_index=255, max_guide_iter=10, num_classes=6).to(DEV)
    loss_fn._epoch, loss_fn._iteration, loss_fn._rank = 1, 0, 0
    try:
        out = loss_fn(feats, feats, seg1, seg2, t1, t2)
        check('non-contiguous forward ok', bool(torch.isfinite(out).item()))
    except Exception as e:
        print('  [FAIL] non-contiguous forward raised %s: %s' % (type(e).__name__, str(e)[:160]))
        FAIL += 1

    print('--- invalid labels must raise ---')
    run_case('label=6 (out of range)', 1, 8192, label_error=6, expect_error=ValueError)
    run_case('label=-1 (out of range)', 1, 8192, label_error=-1, expect_error=ValueError)

    print('--- NaN input must raise ---')
    feats, seg1, seg2, t1, t2 = make_inputs(1, 8192, seed=7)
    seg1.data[0, 0, 0] = float('nan')
    seg2.data.copy_(seg1)
    loss_fn = GudiePointContrastLoss(npos=4096, T=0.4, is_guide=False,
                                     ignore_index=255, max_guide_iter=10, num_classes=6).to(DEV)
    loss_fn._epoch, loss_fn._iteration, loss_fn._rank = 1, 0, 0
    try:
        _ = loss_fn(feats, feats, seg1, seg2, t1, t2)
        print('  [FAIL] NaN input did not raise')
        FAIL += 1
    except ValueError as e:
        print('  [PASS] NaN input raised ValueError: %s' % str(e)[:140])
        PASS += 1
    except Exception as e:
        print('  [FAIL] NaN input raised %s (expected ValueError): %s'
              % (type(e).__name__, str(e)[:160]))
        FAIL += 1

    print('\n===== RESULT: %d passed, %d failed =====' % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
