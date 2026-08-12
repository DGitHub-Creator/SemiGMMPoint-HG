#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregates official (non-smoke) eval results into experiments/odpt/summary.*.

Only reads eval/<RUN_ID=official>/metrics.json for each budget. Missing or
not-yet-produced results are reported as NOT_AVAILABLE (never 0, never stale).
"""
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO_ROOT, 'experiments', 'odpt')
BUDGETS = [10, 20]
FIELDS = ['method', 'budget_alias', 'labeled_scenes', 'labeled_point_ratio',
          'seed', 'checkpoint', 'mIoU', 'mAcc', 'OA']


def load_official(budget):
    p = os.path.join(BASE, f'{budget}pct', 'eval', 'official', 'metrics.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    rows = []
    for b in BUDGETS:
        r = load_official(b)
        if r is None:
            rows.append({'method': 'SemiGMMPoint', 'budget_alias': f'{b}%',
                         'labeled_scenes': 2 if b == 10 else 3,
                         'labeled_point_ratio': 0.1431 if b == 10 else 0.2117,
                         'seed': 'NOT_AVAILABLE', 'checkpoint': 'NOT_AVAILABLE',
                         'mIoU': 'NOT_AVAILABLE', 'mAcc': 'NOT_AVAILABLE',
                         'OA': 'NOT_AVAILABLE'})
        else:
            rows.append({k: r.get(k, 'NOT_AVAILABLE') for k in FIELDS})

    os.makedirs(BASE, exist_ok=True)
    with open(os.path.join(BASE, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def fmt(b):
        r = load_official(b)
        if r is None:
            return 'NOT_AVAILABLE'
        return f'{r["mIoU"]*100:.2f}%'

    def fmt_a(b):
        r = load_official(b)
        if r is None:
            return 'NOT_AVAILABLE'
        return f'{r["mAcc"]*100:.2f}%'

    lines = [
        'SemiGMMPoint summary (Area_3, official runs only)',
        f'SemiGMMPoint 10% mIoU = {fmt(10)}',
        f'SemiGMMPoint 10% mAcc = {fmt_a(10)}',
        f'SemiGMMPoint 20% mIoU = {fmt(20)}',
        f'SemiGMMPoint 20% mAcc = {fmt_a(20)}',
    ]
    with open(os.path.join(BASE, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
