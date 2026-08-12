#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregates official (non-smoke) ODPT-HG eval results into
experiments/odpt_hg/summary.* (dataset_name=ODPT-HG only).
"""
import csv
import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO_ROOT, 'experiments', 'odpt_hg')
BUDGETS = [10, 20]
FIELDS = ['method', 'dataset_name', 'budget_alias', 'dataset_root',
          'dataset_version', 'labeled_scenes', 'seed', 'checkpoint',
          'mIoU', 'mAcc', 'OA', 'split_fingerprint']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='official',
                        help='eval subdirectory shared by the 10%% and 20%% runs')
    args = parser.parse_args()
    if not args.run_id or args.run_id in ('.', '..') or \
            os.path.basename(args.run_id) != args.run_id:
        parser.error('--run-id must be one non-empty path component')
    return args


def load_official(budget, run_id):
    p = os.path.join(BASE, f'{budget}pct', 'eval', run_id, 'metrics.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    args = parse_args()
    rows = []
    for b in BUDGETS:
        r = load_official(b, args.run_id)
        if r is None:
            rows.append({k: ('NOT_AVAILABLE' if k not in ('method', 'dataset_name',
                                                          'budget_alias')
                             else ('SemiGMMPoint' if k == 'method' else
                                   ('ODPT-HG' if k == 'dataset_name' else f'{b}%')))
                         for k in FIELDS})
        else:
            rows.append({k: r.get(k, 'NOT_AVAILABLE') for k in FIELDS})

    os.makedirs(BASE, exist_ok=True)
    with open(os.path.join(BASE, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def fmt(b, key):
        r = load_official(b, args.run_id)
        if r is None:
            return 'NOT_AVAILABLE'
        return f'{r[key]*100:.2f}%'

    lines = [
        'SemiGMMPoint ODPT-HG summary (Area_3, official runs only)',
        f'run_id: {args.run_id}',
        'dataset_name: ODPT-HG',
        'dataset_root: /path/to/odpt-hg-dataset',
        'dataset_fingerprint: experiments/odpt_hg/dataset_fingerprint.json',
        f'SemiGMMPoint HG 10% mIoU = {fmt(10, "mIoU")}',
        f'SemiGMMPoint HG 10% mAcc = {fmt(10, "mAcc")}',
        f'SemiGMMPoint HG 20% mIoU = {fmt(20, "mIoU")}',
        f'SemiGMMPoint HG 20% mAcc = {fmt(20, "mAcc")}',
    ]
    with open(os.path.join(BASE, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
