#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ODPT-HG official evaluation: fixed final checkpoint, exactly the 3 Area_3
scenes of the HG dataset.

Same protocol as tools/odpt_eval.py but:
  * reads data_root / test_scenes / labeled ratio from the HG config & split
  * records dataset_name=ODPT-HG, dataset_root, dataset_version,
    dataset_fingerprint & split_fingerprint (sha256) in the results

Metrics: IoU / Acc per class (rows=GT), mIoU, mAcc, OA over one global
confusion matrix over the 3 test scenes.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from openpoints.dataset.build import build_transforms_from_cfg  # noqa: E402
from openpoints.dataset.data_util import get_features_by_keys, voxelize  # noqa: E402
from openpoints.dataset.s3dis.odpt import list_area_scenes  # noqa: E402
from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.utils.config import EasyConfig  # noqa: E402
from openpoints.utils.ckpt_util import load_checkpoint  # noqa: E402
from torch_scatter import scatter  # noqa: E402

CLASS_NAMES = ['pipeline', 'steel_frame', 'elbow_pipe',
               'valve_guardrail', 'gate_valve', 'Christmas_tree_body']
HG_ROOT = '/path/to/odpt-hg-dataset'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_scene(path, cfg):
    data = torch.load(path, map_location='cpu')
    coord, feat, label = data[0], data[1], data[2]
    coord = np.asarray(coord, dtype=np.float32)
    feat = np.clip((np.asarray(feat, dtype=np.float32) + 1) / 2., 0, 1).astype(np.float32)
    label = np.asarray(label).reshape(-1).astype(np.int64)
    coord -= coord.min(0)
    idx_points = []
    reverse_idx_part = reverse_idx_sort = None
    voxel_idx = None
    voxel_size = cfg.dataset.common.get('voxel_size', None)
    if voxel_size is not None:
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max() + 1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle]
            reverse_idx_part = np.argsort(idx_shuffle, axis=0)
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


def predict_scene(model, data_path, cfg, pipe_transform, gravity_dim,
                  return_logits=False):
    model.eval()
    coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort = \
        load_scene(data_path, cfg)
    if label is not None:
        label = torch.from_numpy(label)
    len_part = len(idx_points)
    nearest_neighbor = len_part == 1
    all_item_logit = []
    with torch.no_grad():
        for idx_subcloud in tqdm(range(len(idx_points)), leave=False, desc='eval subcloud'):
            if not (nearest_neighbor and idx_subcloud > 0):
                idx_part = idx_points[idx_subcloud]
                coord_part = coord[idx_part]
                coord_part -= coord_part.min(0)
                feat_part = feat[idx_part]
                data = {'pos': coord_part}
                if feat_part is not None:
                    data['x'] = feat_part
                if pipe_transform is not None:
                    data = pipe_transform(data)
                if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                    data['heights'] = torch.from_numpy(
                        coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
                if not cfg.dataset.common.get('variable', False):
                    if 'x' in data.keys():
                        data['x'] = data['x'].unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)
                data['x'] = get_features_by_keys(data, cfg.feature_keys)
                out = model(data)
                logits = out[1] if len(out) == 3 else out[0]
            all_item_logit.append(logits)
    all_logits = torch.cat(all_item_logit, dim=0)
    if not cfg.dataset.common.get('variable', False):
        all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    if not nearest_neighbor:
        idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
        all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
    else:
        all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx_sort]
    pred = all_logits.argmax(dim=1).cpu().numpy()
    label_np = label.cpu().numpy() if label is not None else None
    if return_logits:
        return pred, label_np, all_logits
    return pred, label_np


def metrics_from_cm(cm):
    gt = cm.sum(1)
    pred = cm.sum(0)
    tp = np.diag(cm)
    denom = (gt + pred - tp).astype(float)
    ious = np.where(denom > 0, tp / denom, 0.0)
    accs = np.where(gt > 0, tp / gt.astype(float), 0.0)
    oa = tp.sum() / max(gt.sum(), 1)
    return float(oa), float(accs.mean()), float(ious.mean()), ious, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, required=True, choices=[10, 20])
    ap.add_argument('--cfg', type=str, required=True, help='absolute path to the semi config yaml')
    ap.add_argument('--checkpoint', type=str, required=True, help='absolute path to final.pth')
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=100, help='expected training epochs (100 official)')
    ap.add_argument('--smoke', action='store_true', help='allow non-final checkpoints (smoke mode)')
    args = ap.parse_args()

    budget = args.budget
    assert os.path.exists(args.checkpoint), f'checkpoint not found: {args.checkpoint}'
    os.makedirs(args.outdir, exist_ok=True)

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    test_scenes = list(cfg.dataset.common.test_scenes)
    assert len(test_scenes) == 3, f'expected 3 test scenes, got {test_scenes}'
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    data_root = cfg.dataset.common.data_root

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_epoch = ckpt.get('epoch', None)
    if ckpt_epoch != args.epochs:
        if args.smoke:
            print(f'[WARN] checkpoint epoch = {ckpt_epoch} (expected {args.epochs}); '
                  f'smoke mode allows it')
        else:
            print(f'[ERROR] checkpoint epoch = {ckpt_epoch}, expected {args.epochs} '
                  f'(final checkpoint). Refusing to evaluate a non-final checkpoint.')
            sys.exit(1)

    model = build_model_from_cfg(cfg.model).cuda()
    load_checkpoint(model, args.checkpoint)
    model.eval()

    pipe_transform = build_transforms_from_cfg('val', cfg.datatransforms)
    gravity_dim = cfg.datatransforms.kwargs.gravity_dim

    global_cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    per_scene = {}
    for scene in test_scenes:
        path = os.path.join(data_root, scene + '.pth')
        assert os.path.exists(path), f'missing test scene: {path}'
        pred, label = predict_scene(model, path, cfg, pipe_transform, gravity_dim)
        assert label is not None, f'no GT for test scene {scene}'
        assert len(pred) == len(label), f'{scene}: pred {len(pred)} != gt {len(label)}'
        assert pred.min() >= 0 and pred.max() <= cfg.num_classes - 1
        valid = label != 255
        if not valid.all():
            print(f'[WARN] {scene}: {int((~valid).sum())} GT=255 points filtered')
        lab = label[valid]
        pred_v = pred[valid]
        assert lab.min() >= 0 and lab.max() <= cfg.num_classes - 1
        for c in range(cfg.num_classes):
            global_cm[c, :] += np.bincount(pred_v[lab == c], minlength=cfg.num_classes)
        per_scene[scene] = {'valid_points': int(len(lab)), 'pred_points': int(len(pred))}
        print(f'  {scene}: valid points {len(lab)}, pred points {len(pred)}')

    total_valid = sum(v['valid_points'] for v in per_scene.values())
    assert global_cm.sum() == total_valid

    oa, macc, miou, ious, accs = metrics_from_cm(global_cm)
    print()
    print(f'Global confusion matrix ({global_cm.sum()} points):')
    print(global_cm)
    for i, name in enumerate(CLASS_NAMES):
        print(f'  {name:20s} IoU={ious[i]*100:6.2f}%  Acc={accs[i]*100:6.2f}%')
    print(f'  OA={oa*100:.2f}%  mAcc={macc*100:.2f}%  mIoU={miou*100:.2f}%')

    split_path = os.path.join(data_root, 'splits', f'{budget}.txt')
    data_fingerprint = os.path.join('experiments', 'odpt_hg', 'dataset_fingerprint.json')
    result = {
        'method': 'SemiGMMPoint',
        'dataset_name': 'ODPT-HG',
        'dataset_root': HG_ROOT,
        'data_root': data_root,
        'dataset_version': '1.0',
        'dataset_fingerprint': data_fingerprint,
        'split_fingerprint': sha256_file(split_path),
        'budget_alias': f'{budget}%',
        'labeled_scenes': int(len([l for l in open(split_path) if l.strip()])),
        'test_area': 'Area_3',
        'test_scenes': list(test_scenes),
        'per_scene_points': per_scene,
        'checkpoint': os.path.abspath(args.checkpoint),
        'epoch': int(ckpt_epoch),
        'seed': int(cfg.seed),
        'OA': round(float(oa), 4),
        'mAcc': round(float(macc), 4),
        'mIoU': round(float(miou), 4),
        'per_class_accuracy': {CLASS_NAMES[i]: round(float(accs[i]), 4) for i in range(6)},
        'per_class_IoU': {CLASS_NAMES[i]: round(float(ious[i]), 4) for i in range(6)},
        'confusion_matrix': global_cm.tolist(),
        'smoke': args.smoke,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, 'metrics.txt'), 'w') as f:
        f.write(f'SemiGMMPoint HG {budget}% (Area_3, {global_cm.sum()} valid points)\n')
        f.write(f'  dataset_name: ODPT-HG\n')
        f.write(f'  dataset_root: {HG_ROOT}\n')
        f.write(f'  dataset_fingerprint: {data_fingerprint}\n')
        f.write(f'  split_fingerprint: {result["split_fingerprint"]}\n')
        f.write(f'  checkpoint: {os.path.abspath(args.checkpoint)}\n')
        f.write(f'  epoch: {ckpt_epoch}, seed: {cfg.seed}\n')
        f.write(f'  OA: {oa*100:.2f}%  mAcc: {macc*100:.2f}%  mIoU: {miou*100:.2f}%\n')
        for i, name in enumerate(CLASS_NAMES):
            f.write(f'  IoU_{name}: {ious[i]*100:.2f}%  Acc_{name}: {accs[i]*100:.2f}%\n')
        f.write('confusion matrix (rows=GT, cols=Pred):\n')
        for i in range(6):
            f.write(f'  GT{i}  ' + ' '.join(f'{global_cm[i, j]:>10d}' for j in range(6)) + '\n')
    with open(os.path.join(args.outdir, 'confusion_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['row=GT', 'col=Pred'] + CLASS_NAMES)
        for i in range(6):
            w.writerow([CLASS_NAMES[i], i] + list(global_cm[i]))
    with open(os.path.join(args.outdir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['method', 'dataset_name', 'budget_alias', 'checkpoint', 'epoch',
                    'seed', 'OA', 'mAcc', 'mIoU'])
        w.writerow(['SemiGMMPoint', 'ODPT-HG', f'{budget}%',
                    os.path.abspath(args.checkpoint), int(ckpt_epoch), int(cfg.seed),
                    round(float(oa), 4), round(float(macc), 4), round(float(miou), 4)])
    print(f'\nresults written to {args.outdir}')
    print('EVAL_OK')


if __name__ == '__main__':
    main()
