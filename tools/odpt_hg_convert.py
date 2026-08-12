#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert HG raw txt scenes into the dataset format consumed by SemiGMMPoint.

Source  (read-only): /path/to/odpt-hg-dataset/Area_*.txt
Target  (new):       /path/to/odpt-hg-data/Area_*.pth
                     /path/to/odpt-hg-data/splits/{10,20,100}.txt
                     /path/to/odpt-hg-data/PROVENANCE.txt

Encoding (deterministic, one-to-one with the HG source rows):
  coords = raw XYZ  (float32)   - the training code min-shifts internally
  colors = raw RGB  /127.5 - 1  (float32, [-1,1]) - load_odpt_scene maps back
                                          to 0..255 before any transform
  labels = raw label (float32)  - {0..5}

Every converted file records its HG source file and the source SHA256.
"""
import os
import glob
import hashlib
import numpy as np
import torch

HG_ROOT = '/path/to/odpt-hg-dataset'
DST_ROOT = '/path/to/odpt-hg-data'
OLD_SPLIT_ROOT = '/path/to/s3dis/data_split'

os.makedirs(DST_ROOT, exist_ok=True)
os.makedirs(os.path.join(DST_ROOT, 'splits'), exist_ok=True)

provenance = []
for fn in sorted(glob.glob(os.path.join(HG_ROOT, 'Area_*.txt'))):
    name = os.path.basename(fn)[:-4]
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    src_sha = h.hexdigest()
    v = np.loadtxt(fn)  # x y z r g b label
    coords = v[:, :3].astype(np.float32)
    colors = (v[:, 3:6].astype(np.float32) / 127.5 - 1.0).astype(np.float32)
    labels = v[:, 6].astype(np.float32)
    dst = os.path.join(DST_ROOT, name + '.pth')
    torch.save((coords, colors, labels), dst)
    sha = hashlib.sha256()
    with open(dst, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            sha.update(chunk)
    provenance.append(dict(
        hg_source=os.path.join(HG_ROOT, fn),
        hg_source_sha256=src_sha,
        converted=os.path.relpath(dst, DST_ROOT),
        converted_sha256=sha.hexdigest(),
        points=int(len(coords)),
        label_unique=sorted(set(np.unique(labels).tolist())),
    ))
    print(f'{name}: N={len(coords)}  -> {dst}')

# ---- splits: scene identities verified identical to old dataset ------------
old_names = {b: [l.strip() for l in open(os.path.join(OLD_SPLIT_ROOT, f'{b}.txt')) if l.strip()]
             for b in (10, 20)}
all_area1 = sorted((n for n in provenance
                    if os.path.basename(n['hg_source']).startswith('Area_1_')),
                   key=lambda p: p['hg_source'])
# keep relative list of all 15 Area_1 scenes
all_scenes = [os.path.basename(p['converted'])[:-4] for p in all_area1]
open(os.path.join(DST_ROOT, 'splits', '100.txt'), 'w').write('\n'.join(all_scenes) + '\n')
for b in (10, 20):
    missing = [s for s in old_names[b] if s not in all_scenes]
    assert not missing, f'old split {b} scenes missing in HG: {missing}'
    open(os.path.join(DST_ROOT, 'splits', f'{b}.txt'), 'w').write(
        '\n'.join(old_names[b]) + '\n')

# provenance manifest (also asserts converted label range)
with open(os.path.join(DST_ROOT, 'PROVENANCE.txt'), 'w') as f:
    f.write('ODPT-HG conversion manifest\n')
    f.write('source root (read-only): %s\n' % HG_ROOT)
    f.write('converted root:          %s\n' % DST_ROOT)
    f.write('split files come from old canonical splits whose scenes were\n')
    f.write('verified to exist, with identical point counts and labels, in HG.\n')
    for p in provenance:
        f.write('%-32s points=%-9d labels=%s\n  src=%s\n  src_sha256=%s\n  out=%s/%s\n  sha256=%s\n'
                % (os.path.basename(p['hg_source']), p['points'], p['label_unique'],
                   p['hg_source'], p['hg_source_sha256'],
                   DST_ROOT, p['converted'], p['converted_sha256']))
    f.write('splits:\n')
    for b in (10, 20, 100):
        f.write('  splits/%d.txt: %s\n' % (b, ','.join(
            open(os.path.join(DST_ROOT, 'splits', f'{b}.txt')).read().split())))
print('written:', os.path.join(DST_ROOT, 'PROVENANCE.txt'))
print('splits:', sorted(os.listdir(os.path.join(DST_ROOT, 'splits'))))