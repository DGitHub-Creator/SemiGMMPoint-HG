#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibrate ODPT-HG confidence using labeled Area 1 scenes only.

This diagnostic never reads Area 3 and never changes a checkpoint.  It runs
the same full-scene voxel voting used by the standard evaluator, then reports
confidence/accuracy evidence for selecting or rejecting a pseudo-label gate.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from openpoints.dataset.build import build_transforms_from_cfg  # noqa: E402
from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.utils.ckpt_util import load_checkpoint  # noqa: E402
from openpoints.utils.config import EasyConfig  # noqa: E402
from tools.odpt_hg_eval import CLASS_NAMES, predict_scene  # noqa: E402


THRESHOLDS = np.asarray([0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
                        dtype=np.float64)


def read_scenes(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def safe_float(value):
    return None if not np.isfinite(value) else round(float(value), 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, required=True, choices=[10, 20])
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-mode', choices=['nearest_neighbor', 'multi_voxel'],
                    default='nearest_neighbor')
    args = ap.parse_args()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.test_mode = args.test_mode
    np.random.seed(1)
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    split_path = os.path.join(cfg.dataset.common.data_root, 'splits',
                              '%d.txt' % args.budget)
    scenes = read_scenes(split_path)
    assert scenes, 'empty labeled split: %s' % split_path
    assert all(scene.startswith('Area_1_') for scene in scenes), scenes
    assert os.path.isfile(args.checkpoint), args.checkpoint

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = build_model_from_cfg(cfg.model).cuda()
    load_checkpoint(model, args.checkpoint)
    model.eval()
    pipe_transform = build_transforms_from_cfg('val', cfg.datatransforms)
    gravity_dim = cfg.datatransforms.kwargs.gravity_dim

    confidences = []
    predictions = []
    targets = []
    per_scene = {}
    for scene in scenes:
        path = os.path.join(cfg.dataset.common.data_root, scene + '.pth')
        pred, target, logits = predict_scene(
            model, path, cfg, pipe_transform, gravity_dim,
            return_logits=True)
        valid = target != cfg.ignore_index
        probs = torch.softmax(logits, dim=1)
        conf = probs.max(dim=1).values.cpu().numpy()[valid]
        pred = pred[valid]
        target = target[valid]
        confidences.append(conf.astype(np.float32, copy=False))
        predictions.append(pred.astype(np.int16, copy=False))
        targets.append(target.astype(np.int16, copy=False))
        per_scene[scene] = {
            'points': int(valid.sum()),
            'accuracy': round(float((pred == target).mean()), 6),
            'mean_confidence': round(float(conf.mean()), 6),
            'max_confidence': round(float(conf.max()), 6),
        }
        print('%s points=%d acc=%.4f mean_conf=%.4f max_conf=%.4f' %
              (scene, valid.sum(), (pred == target).mean(), conf.mean(),
               conf.max()))
        del logits, probs
        torch.cuda.empty_cache()

    confidence = np.concatenate(confidences)
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    correct = prediction == target

    bins = np.linspace(0.0, 1.0, 21)
    bin_rows = []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        count = int(mask.sum())
        if count:
            acc = float(correct[mask].mean())
            mean_conf = float(confidence[mask].mean())
            ece += count / float(len(confidence)) * abs(acc - mean_conf)
        else:
            acc = mean_conf = float('nan')
        bin_rows.append({'lo': round(float(lo), 2), 'hi': round(float(hi), 2),
                         'count': count, 'accuracy': safe_float(acc),
                         'mean_confidence': safe_float(mean_conf)})

    threshold_rows = []
    for threshold in THRESHOLDS:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        row = {
            'threshold': round(float(threshold), 2),
            'accepted': count,
            'coverage': round(count / float(len(confidence)), 6),
            'accuracy': safe_float(correct[accepted].mean()) if count else None,
            'per_predicted_class': {},
        }
        for class_id, name in enumerate(CLASS_NAMES):
            class_mask = accepted & (prediction == class_id)
            class_count = int(class_mask.sum())
            row['per_predicted_class'][name] = {
                'accepted': class_count,
                'accuracy': safe_float(correct[class_mask].mean())
                if class_count else None,
            }
        threshold_rows.append(row)
        print('threshold=%.2f accepted=%d coverage=%.6f accuracy=%s' %
              (threshold, count, row['coverage'], row['accuracy']))

    result = {
        'dataset': 'ODPT-HG labeled Area 1 only',
        'budget': args.budget,
        'scenes': scenes,
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'test_mode': args.test_mode,
        'points': int(len(confidence)),
        'accuracy': round(float(correct.mean()), 6),
        'mean_confidence': round(float(confidence.mean()), 6),
        'max_confidence': round(float(confidence.max()), 6),
        'ece_20bin': round(float(ece), 6),
        'correct_confidence_percentiles': {
            str(q): round(float(np.percentile(confidence[correct], q)), 6)
            for q in (1, 5, 25, 50, 75, 95, 99)
        },
        'incorrect_confidence_percentiles': {
            str(q): round(float(np.percentile(confidence[~correct], q)), 6)
            for q in (1, 5, 25, 50, 75, 95, 99)
        },
        'per_scene': per_scene,
        'bins': bin_rows,
        'thresholds': threshold_rows,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'protocol_note': 'No Area 3 data was read or evaluated.',
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('CALIBRATION_OK out=%s ece=%.6f' % (args.out, ece))


if __name__ == '__main__':
    main()
