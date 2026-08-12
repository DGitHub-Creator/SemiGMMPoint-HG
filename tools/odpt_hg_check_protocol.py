#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ODPT-HG protocol preflight check for a given label budget (10% or 20%).

Verifies (and prints) the scene-level semi-supervised protocol facts for the
HG dataset formal source (/path/to/odpt-hg-dataset):

* split file matches the audited canonical budget split
* labeled / unlabeled / test scene counts for the budget
* labeled GT values stay in {0..5}
* the unlabeled dataset (ODPTPreS3DISHG) never exposes ground truth
  (y == 255 for every sample; line "unlabeled batch labels unique = [255]")
* labeled ∩ unlabeled = empty; labeled ∪ unlabeled = all 15 Area_1 scenes
* train ∩ test = empty
* labeled point ratio recomputed from HG point counts (NOT reusing old 0.1431/
  0.2117 unless they match)
* per-split six-class point counts / coverage, per-labeled-scene distribution
* class 1 (steel_frame) coverage in the labeled set

Exit code is 0 on success, 1 on any protocol violation.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from openpoints.dataset.s3dis.odpt import (  # noqa: E402
    list_area_scenes,
    load_odpt_scene,
    normalize_scene_name,
    read_split_list,
)

HG_ROOT = '/path/to/odpt-hg-dataset'
DATA_ROOT = '/path/to/odpt-hg-data'
SPLIT_ROOT = os.path.join(DATA_ROOT, 'splits')
CANON_SPLIT_ROOT = '/path/to/s3dis/data_split'
TEST_SCENES = ['Area_3_conferenceRoom_20', 'Area_3_conferenceRoom_21',
               'Area_3_conferenceRoom_22']
CLASSES = ['pipeline', 'steel_frame', 'elbow_pipe', 'valve_guardrail',
           'gate_valve', 'Christmas_tree_body']


def counts_of(data_root, scenes):
    counts = np.zeros(6, dtype=np.int64)
    for s in scenes:
        _, _, label = load_odpt_scene(data_root, s)
        counts += np.bincount(np.asarray(label).astype(np.int64), minlength=6)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, required=True, choices=[10, 20])
    ap.add_argument('--data-root', default=DATA_ROOT)
    args = ap.parse_args()
    budget = args.budget
    data_root = args.data_root

    errors = []
    ok = lambda msg: print(f'[OK] {msg}')
    fail = lambda msg: errors.append(msg) or print(f'[FAIL] {msg}')

    all_area1 = list_area_scenes(data_root, 'Area_1')
    if len(all_area1) != 15:
        fail(f'expected 15 Area_1 scenes, got {len(all_area1)}')
    ok(f'HG Area_1 scenes = {len(all_area1)}')

    split_file = os.path.join('splits', f'{budget}.txt')
    labeled = read_split_list(data_root, split_file)
    canon_path = os.path.join(CANON_SPLIT_ROOT, f'{budget}.txt')
    canon = [normalize_scene_name(l) for l in open(canon_path) if normalize_scene_name(l)]
    if labeled == canon:
        ok(f'split {budget} matches canonical {canon_path} (scene identities audited)')
    else:
        fail(f'split {budget} differs from canonical copy: {labeled} vs {canon}')

    missing = [s for s in canon if s not in all_area1]
    if not missing:
        ok(f'all {len(canon)} canonical labeled scenes exist in HG')
    else:
        fail(f'canonical labeled scenes missing in HG: {missing}')

    unlabeled = [s for s in all_area1 if s not in labeled]
    inter = set(labeled) & set(unlabeled)
    if not inter:
        ok('labeled ∩ unlabeled = empty set')
    else:
        fail(f'labeled ∩ unlabeled = {inter}')
    if len(labeled) + len(unlabeled) == len(all_area1):
        ok(f'labeled ∪ unlabeled = all {len(all_area1)} Area_1 scenes')
    else:
        fail('labeled ∪ unlabeled != all train scenes')

    area3 = list_area_scenes(data_root, 'Area_3')
    missing_t = [s for s in TEST_SCENES if s not in area3]
    extra_t = [s for s in area3 if s not in TEST_SCENES]
    if not missing_t and not extra_t and len(area3) == 3:
        ok(f'test set = {TEST_SCENES}')
    else:
        fail(f'test set mismatch: missing={missing_t} extra={extra_t}')
    if set(labeled) & set(TEST_SCENES) or set(unlabeled) & set(TEST_SCENES):
        fail('train ∩ test = non-empty!')
    else:
        ok('train ∩ test = empty set')

    # ---- labeled GT range ----------------------------------------------
    bad = 0
    per_scene_counts = {}
    for scene in labeled:
        _, _, label = load_odpt_scene(data_root, scene)
        if label.min() < 0 or label.max() > 5:
            bad += 1
            print(f'[FAIL] labeled scene {scene} has labels outside 0..5')
        per_scene_counts[scene] = np.bincount(
            np.asarray(label).astype(np.int64), minlength=6).tolist()
    if bad == 0:
        ok('labeled GT values all within 0..5 (no 255, no negatives)')

    # ---- unlabeled GT isolation (dataset level) -------------------------
    from openpoints.dataset.build import build_dataset_from_cfg  # noqa: E402
    ds_cfg = {
        'NAME': 'ODPTPreS3DISHG',
        'data_root': data_root,
        'split_file': split_file,
        'voxel_size': 0.04,
        'split': 'train',
        'mode': 'unlabeled',
        'voxel_max': 4096,
        'loop': 1,
        'presample': False,
    }
    ds = build_dataset_from_cfg(ds_cfg)
    seen = set()
    leak = 0
    for i in range(len(ds)):
        sample = ds[i]
        d1 = sample[0] if isinstance(sample, (list, tuple)) else sample
        y = np.asarray(d1['y'])
        seen.update(np.unique(y).tolist())
        if (y != 255).any():
            leak += 1
    if leak == 0:
        ok(f'unlabeled pool ({len(ds)} scenes) returns y == 255 only (no GT leak)')
        print(f'      unlabeled batch labels unique = {sorted(seen)}')
    else:
        fail(f'{leak} unlabeled samples expose labels != 255')

    # ---- point ratio + per-split class stats ----------------------------
    area1_total = 0
    for s in all_area1:
        coord, _, _ = load_odpt_scene(data_root, s)
        area1_total += len(coord)
    labeled_pts = 0
    for s in labeled:
        coord, _, _ = load_odpt_scene(data_root, s)
        labeled_pts += len(coord)
    ratio = labeled_pts / area1_total
    old_ratio = 0.1431 if budget == 10 else 0.2117
    same = abs(ratio - old_ratio) < 1e-4
    msg = f'labeled point ratio = {ratio:.4f} (recomputed from HG)'
    print(f'[INFO] ' + msg + (f'; matches old value {old_ratio}' if same
                              else f'; old value {old_ratio} NOT reused'))
    if same:
        ok(f'old ratio {old_ratio} confirmed by HG re-computation')
    else:
        fail(f'old ratio {old_ratio} invalidated by HG re-computation ({ratio:.4f})')

    lbl_counts = counts_of(data_root, labeled)
    unl_counts = counts_of(data_root, unlabeled)
    test_counts = counts_of(data_root, TEST_SCENES)
    print(f'[INFO] six-class point counts: labeled={lbl_counts.tolist()}')
    print(f'[INFO] six-class point counts: unlabeled={unl_counts.tolist()}')
    print(f'[INFO] six-class point counts: test={test_counts.tolist()}')
    for name, c in [('labeled', lbl_counts), ('unlabeled', unl_counts),
                    ('test', test_counts)]:
        missing_cls = [CLASSES[i] for i in range(6) if c[i] == 0]
        status = 'ALL 6 CLASSES PRESENT' if not missing_cls else \
            f'MISSING: {missing_cls}'
        print(f'[INFO] {name} set covers: {status}')
    if budget in (10, 20) and lbl_counts[1] == 0:
        fail('class 1 (steel_frame) has ZERO points in the labeled set -> '
             'steel_frame IoU=0 would come from annotation coverage, not the model')
    else:
        ok(f'class 1 (steel_frame) present in labeled set ({lbl_counts[1]:d} pts)')

    print('\nper-labeled-scene class distribution:')
    for s in labeled:
        c = per_scene_counts[s]
        print(f'  {s:28s} N={sum(c):>9d} ' +
              '  '.join(f'{CLASSES[i][:8]}={c[i]:>7d}' for i in range(6)))

    print()
    print(f'protocol_summary budget={budget}% labeled={len(labeled)} '
          f'unlabeled={len(unlabeled)} validation=disabled test={len(area3)} '
          f'labeled_point_ratio={ratio:.4f}')
    if errors:
        print('PROTOCOL_CHECK_FAILED')
        sys.exit(1)
    print('PROTOCOL_CHECK_PASSED')
    sys.exit(0)


if __name__ == '__main__':
    main()