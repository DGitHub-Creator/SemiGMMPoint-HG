#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ODPT protocol preflight check for a given label budget (10% or 20%).

Verifies (and prints) the scene-level semi-supervised protocol facts used by
the SemiGMMPoint ODPT experiments:

* split file equals the canonical copy under /path/to/s3dis/data_split/
* labeled / unlabeled / val / test scene counts for the budget
* labeled GT values stay in {0..5}
* the unlabeled dataset (ODPTPreS3DIS) never exposes ground truth (y == 255)
* the unlabeled pool excludes labeled and val scenes
* the test set is exactly Area_3 conferenceRoom 20/21/22

Exit code is 0 on success, 1 on any protocol violation.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from openpoints.dataset import data_util  # noqa: E402
from openpoints.dataset.s3dis.odpt import (  # noqa: E402
    ODPT_CLASSES,
    list_area_scenes,
    load_odpt_scene,
    normalize_scene_name,
    read_split_list,
)

DATA_ROOT = '/path/to/odpt-data'
CANON_SPLIT_ROOT = '/path/to/s3dis/data_split'
AREA1_TOTAL = 32183437  # sum of all Area_1 scene points (verified at setup time)
EXPECTED = {
    10: {'labeled': 2, 'unlabeled': 13, 'ratio': 0.1431},
    20: {'labeled': 3, 'unlabeled': 12, 'ratio': 0.2117},
}
VAL_SCENES = ['Area_1_conferenceRoom_3', 'Area_1_conferenceRoom_7']
TEST_SCENES = ['Area_3_conferenceRoom_20', 'Area_3_conferenceRoom_21',
               'Area_3_conferenceRoom_22']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, required=True, choices=[10, 20])
    ap.add_argument('--data-root', default=DATA_ROOT)
    args = ap.parse_args()
    budget = args.budget
    data_root = args.data_root
    exp = EXPECTED[budget]

    errors = []
    ok = lambda msg: print(f'[OK] {msg}')
    fail = lambda msg: errors.append(msg) or print(f'[FAIL] {msg}')

    # ---- split file consistency ------------------------------------------
    split_file = os.path.join('splits', f'{budget}.txt')
    local = read_split_list(data_root, split_file)
    canon_path = os.path.join(CANON_SPLIT_ROOT, f'{budget}.txt')
    assert os.path.exists(canon_path), canon_path
    canon = [normalize_scene_name(l) for l in open(canon_path) if normalize_scene_name(l)]
    if local == canon:
        ok(f'split {budget} matches canonical {canon_path}')
    else:
        fail(f'split {budget} differs from canonical copy: {local} vs {canon}')
    labeled_scenes = local

    all_area1 = list_area_scenes(data_root, 'Area_1')
    if len(all_area1) != 15:
        fail(f'expected 15 Area_1 scenes, got {len(all_area1)}')

    val_scenes = [s for s in VAL_SCENES if s in all_area1]
    if len(val_scenes) != 2:
        fail(f'val scenes {VAL_SCENES} not all present in Area_1')
    unlabeled_scenes = [s for s in all_area1 if s not in labeled_scenes]

    if len(labeled_scenes) == exp['labeled']:
        ok(f'labeled scenes = {len(labeled_scenes)}: {labeled_scenes}')
    else:
        fail(f'expected {exp["labeled"]} labeled scenes, got {len(labeled_scenes)}: {labeled_scenes}')
    if len(unlabeled_scenes) == exp['unlabeled']:
        ok(f'unlabeled scenes = {len(unlabeled_scenes)}: {unlabeled_scenes}')
    else:
        fail(f'expected {exp["unlabeled"]} unlabeled scenes, got {len(unlabeled_scenes)}')

    # ---- test set ----------------------------------------------------------
    area3 = list_area_scenes(data_root, 'Area_3')
    missing = [s for s in TEST_SCENES if s not in area3]
    extra = [s for s in area3 if s not in TEST_SCENES]
    if not missing and not extra and len(area3) == 3:
        ok(f'test set = {TEST_SCENES}')
    else:
        fail(f'test set mismatch: missing={missing} extra={extra}')

    # ---- labeled GT range --------------------------------------------------
    bad = 0
    for scene in labeled_scenes:
        _, _, label = load_odpt_scene(data_root, scene)
        if label.min() < 0 or label.max() > 5:
            bad += 1
            print(f'[FAIL] labeled scene {scene} has labels outside 0..5')
    if bad == 0:
        ok('labeled GT values all within 0..5')

    # ---- unlabeled GT isolation (dataset level) ----------------------------
    from openpoints.dataset.build import build_dataset_from_cfg
    from openpoints.transforms import build_transforms_from_cfg
    ds_cfg = {
        'NAME': 'ODPTPreS3DIS',
        'data_root': data_root,
        'split_file': split_file,
        'val_scenes': VAL_SCENES,
        'voxel_size': 0.04,
        'split': 'train',
        'mode': 'unlabeled',
        'voxel_max': 4096,
        'loop': 1,
        'presample': False,
    }
    ds = build_dataset_from_cfg(ds_cfg)
    leak = 0
    for i in range(len(ds)):
        sample = ds[i]
        d1 = sample[0] if isinstance(sample, (list, tuple)) else sample
        y = np.asarray(d1['y'])
        if (y != 255).any():
            leak += 1
            print(f'[FAIL] unlabeled sample {i} exposes labels != 255')
    if leak == 0:
        ok(f'unlabeled pool ({len(ds)} scenes) returns y == 255 only (no GT leak)')

    # ---- point ratio --------------------------------------------------------
    pts = 0
    for scene in labeled_scenes:
        coord, _, _ = load_odpt_scene(data_root, scene)
        pts += len(coord)
    ratio = pts / AREA1_TOTAL
    if abs(ratio - exp['ratio']) < 1e-4:
        ok(f'labeled point ratio = {ratio:.4f} (expected ~{exp["ratio"]})')
    else:
        fail(f'labeled point ratio = {ratio:.4f}, expected ~{exp["ratio"]}')

    print()
    print(f'protocol_summary budget={budget}% labeled={len(labeled_scenes)} '
          f'unlabeled={len(unlabeled_scenes)} validation=disabled test={len(area3)} '
          f'labeled_point_ratio={ratio:.4f}')
    if errors:
        print('PROTOCOL_CHECK_FAILED')
        sys.exit(1)
    print('PROTOCOL_CHECK_PASSED')
    sys.exit(0)


if __name__ == '__main__':
    main()
