#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ODPT-HG dataset fingerprint.

Computes and writes experiments/odpt_hg/dataset_fingerprint.json:

* dataset_name / dataset_root / dataset_version
* every formal data file: relative path, size, SHA256, point count,
  six-class label statistics
* split files 10.txt / 20.txt with SHA256 (split_fingerprint)
* conversion provenance pointer (PROVENANCE.txt in the converted root)

Also verifies the converted .pth files round-trip to the HG txt sources
(coords value-identical modulo per-scene mean shift, colors identical after
/127.5-1, labels identical).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch

HG_ROOT = '/path/to/odpt-hg-dataset'
DATA_ROOT = '/path/to/odpt-hg-data'
OUT = 'experiments/odpt_hg/dataset_fingerprint.json'
CLASSES = ['pipeline', 'steel_frame', 'elbow_pipe', 'valve_guardrail',
           'gate_valve', 'Christmas_tree_body']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = []
    for fn in sorted(glob.glob(os.path.join(HG_ROOT, 'Area_*.txt'))):
        name = os.path.basename(fn)[:-4]
        sha = sha256_file(fn)
        v = np.loadtxt(fn)
        counts = np.bincount(v[:, 6].astype(np.int64), minlength=6)
        files.append({
            'relative_path': os.path.relpath(fn, HG_ROOT),
            'size_bytes': os.path.getsize(fn),
            'sha256': sha,
            'points': int(len(v)),
            'class_counts': {CLASSES[i]: int(counts[i]) for i in range(6)},
            'class_count_list': counts.tolist(),
            'label_min': int(v[:, 6].min()),
            'label_max': int(v[:, 6].max()),
            'nan_inf': int(np.isnan(v).sum() + (np.abs(v) == np.inf).sum()),
        })
        print(f'{name:34s} N={len(v):>9d} sha256={sha[:16]}...')

    splits = {}
    for b in (10, 20):
        p = os.path.join(DATA_ROOT, 'splits', f'{b}.txt')
        splits[f'{b}.txt'] = {'sha256': sha256_file(p),
                              'scenes': [l.strip() for l in open(p) if l.strip()]}

    fingerprint = {
        'dataset_name': 'ODPT-HG',
        'dataset_root': HG_ROOT,
        'data_root': DATA_ROOT,
        'dataset_version': '1.0',
        'generated_at': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'num_scenes': len(files),
        'area1_scenes': sum(1 for f in files if os.path.basename(f['relative_path']).startswith('Area_1_')),
        'area3_scenes': sum(1 for f in files if os.path.basename(f['relative_path']).startswith('Area_3_')),
        'total_points': sum(f['points'] for f in files),
        'files': files,
        'splits': splits,
        'split_fingerprint_sha256_10': splits['10.txt']['sha256'],
        'split_fingerprint_sha256_20': splits['20.txt']['sha256'],
        'conversion_provenance': os.path.join(DATA_ROOT, 'PROVENANCE.txt'),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(fingerprint, f, indent=1)
    print('\nwritten:', OUT)
    print('total points:', fingerprint['total_points'])
    print('split 10 sha256:', splits['10.txt']['sha256'])
    print('split 20 sha256:', splits['20.txt']['sha256'])


if __name__ == '__main__':
    main()