#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ODPT-HG per-split class weights audit artifact.

For each budget split (10/20) computes -- from the GT of the labeled scenes of
that split ONLY (never from the unlabeled pool, never from Area_3):

  * labeled scene list (the exact scenes used for the weights)
  * per-class point counts and frequencies
  * final class weights via get_class_weights() (w = 1/(count/sum + 0.02),
    normalized so the weights average to 1) -- identical to the runtime
    formula used by semi_gmmpoint_main.py

and writes experiments/odpt_hg/class_weights/split<BUDGET>.json together with
the split fingerprint (sha256) for auditability.

Usage:
  python tools/odpt_hg_class_weights.py            # both splits
  python tools/odpt_hg_class_weights.py --budget 10
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from openpoints.dataset.data_util import get_class_weights  # noqa: E402
from openpoints.dataset.s3dis.odpt import (  # noqa: E402
    load_odpt_scene,
    read_split_list,
)

HG_ROOT = '/path/to/odpt-hg-dataset'
DATA_ROOT = '/path/to/odpt-hg-data'
SPLIT_ROOT = os.path.join(DATA_ROOT, 'splits')
CLASSES = ['pipeline', 'steel_frame', 'elbow_pipe', 'valve_guardrail',
           'gate_valve', 'Christmas_tree_body']
OUT_DIR = os.path.join(REPO_ROOT, 'experiments', 'odpt_hg', 'class_weights')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def labeled_counts(data_root, scenes):
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for s in scenes:
        _, _, label = load_odpt_scene(data_root, s)
        counts += np.bincount(np.asarray(label).astype(np.int64),
                              minlength=len(CLASSES))
    return counts


def process_budget(budget):
    split_rel = os.path.join('splits', f'{budget}.txt')
    split_abs = os.path.join(data_root, split_rel)
    scenes = read_split_list(data_root, split_rel)
    counts = labeled_counts(data_root, scenes)
    total = int(counts.sum())
    freq = (counts / max(total, 1)).tolist()
    weights = get_class_weights(counts.astype(np.float32), normalize=True)
    weights = [round(float(w), 4) for w in weights.tolist()]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'split{budget}.json')
    record = {
        'dataset_name': 'ODPT-HG',
        'dataset_root': HG_ROOT,
        'data_root': data_root,
        'budget_pct': budget,
        'split_file': split_abs,
        'split_fingerprint': sha256_file(split_abs),
        'labeled_scenes': scenes,
        'num_labeled_scenes': len(scenes),
        'classes': CLASSES,
        'class_point_counts': counts.tolist(),
        'class_frequencies': [round(x, 8) for x in freq],
        'class_weights_normalized_mean1': weights,
        'weight_formula':
            'w_c = 1/(count_c/sum + 0.02); normalized: w *= C / sum(w)',
        'weights_source': 'GT of labeled split scenes only; '
                          'unlabeled pool and Area_3 GT NOT used',
    }
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f'dataset_name: ODPT-HG')
    print(f'budget: {budget}%   labeled scenes: {scenes}')
    print(f'class points    : {counts.tolist()}')
    print(f'class frequency : {[round(x, 8) for x in freq]}')
    print(f'class weights   : {weights}')
    print(f'written: {out_path}')
    return counts, weights


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, choices=[10, 20])
    ap.add_argument('--data-root', default=DATA_ROOT)
    args = ap.parse_args()
    data_root = args.data_root
    budgets = [args.budget] if args.budget else [10, 20]
    for b in budgets:
        process_budget(b)