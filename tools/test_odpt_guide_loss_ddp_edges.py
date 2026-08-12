#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DDP edge test for GudiePointContrastLoss: different ranks see different K.

Covers the DDP-deadlock / unused-parameter hazards of the K=0/K=1 fixes:
  phase A: rank0 guide K=1 (single class)  vs  rank1 guide K>1
  phase B: rank0 accepts zero low-confidence pseudo points vs rank1 accepts
           high-confidence pseudo points
Each phase runs forward -> backward -> optimizer.step under real DDP
(all-reduce on gradients) and asserts no hang, finite grads, finite params.

Run:  python tools/test_odpt_guide_loss_ddp_edges.py
"""
import os
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO_ROOT)

from openpoints.loss.gudie_point_contrast_loss import GudiePointContrastLoss  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self, C=64, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(3, C)
        self.fc2 = nn.Linear(C, num_classes)

    def forward(self, pos):  # pos: [B, N, 3]
        f = torch.relu(self.fc1(pos))            # [B, N, C]
        seg = self.fc2(f)                         # [B, N, 6]
        return f.transpose(1, 2), seg.transpose(1, 2)  # [B, C, N], [B, 6, N]


def build_inputs(rank, phase, B=4, N=2048, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed + rank * 100 + phase)
    t = torch.full((B, N), 255, dtype=torch.long, device='cuda')
    pos = torch.randn(B, N, 3, generator=g, device='cuda')
    return pos, t


def steer_logits(rank, phase_idx, seg):
    # keep only the steering factors the model cannot learn by itself
    with torch.no_grad():
        if phase_idx == 1 and rank == 0:
            # Joint confidence is 1/6, so the real 0.75 gate accepts zero.
            seg.zero_()
        elif rank == 0:
            # single dominant class everywhere -> all argmax = class 2 -> guide K=1
            seg.fill_(-3.0)
            seg[:, 2, :] = 3.0
        else:
            # three spatial regions with different dominant classes -> K>1
            seg.fill_(-3.0)
            n = seg.shape[2]
            seg[:, 0, : n // 3] = 3.0
            seg[:, 3, n // 3: 2 * n // 3] = 3.0
            seg[:, 5, 2 * n // 3:] = 3.0


def run(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['NCCL_DEBUG'] = 'WARN'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % port,
                            world_size=world_size, rank=rank)
    ok = True
    try:
        model = TinyModel().cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        loss_fn = GudiePointContrastLoss(npos=4096, T=0.4, is_guide=False,
                                         ignore_index=255, max_guide_iter=10,
                                         num_classes=6).cuda()
        for phase_idx, phase in enumerate(('A', 'B')):
            pos, t = build_inputs(rank, phase_idx)
            feats, seg = model(pos)
            steer_logits(rank, phase_idx, seg)
            loss_fn._epoch, loss_fn._iteration, loss_fn._rank = 1, 0, rank
            out = loss_fn(feats, feats, seg, seg, t, t.clone())
            assert out.dim() == 0, 'loss not scalar'
            assert torch.isfinite(out).item(), 'loss not finite: %s' % out.item()
            loss = out / world_size
            loss.backward()
            grad_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all().item())
                              for p in model.parameters())
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            opt.step()
            opt.zero_grad()
            params_finite = all(bool(torch.isfinite(p).all().item())
                                for p in model.parameters())
            stats = loss_fn.pl_stats or {}
            print('[rank%d phase%s] loss=%.4f grads_finite=%s params_finite=%s '
                  'guide_points=%s guide_skipped=%s nce_skipped=%s'
                  % (rank, phase, out.item(), grad_finite, params_finite,
                     stats.get('accepted_guide_points'),
                     stats.get('guide_ce_skipped'), stats.get('nce_skipped')))
            assert grad_finite, 'non-finite gradients'
            assert params_finite, 'non-finite parameters'
            # verify the intended K per phase/rank actually occurred
            if phase == 'A':
                if rank == 0:
                    assert stats.get('accepted_guide_points') == 40, \
                        'expected K=1 (40 guide points), got %s' % stats
                else:
                    assert stats.get('accepted_guide_points') == 80, \
                        'expected K=2 (80 guide points), got %s' % stats
            else:
                if rank == 0:
                    assert stats.get('accepted_guide_points') == 0 and \
                        stats.get('guide_ce_skipped', 0) > 0, \
                        'expected K=0 (guide skipped), got %s' % stats
                    assert stats.get('pseudo_gate_accepted') == 0 and \
                        stats.get('coverage') == 0.0, \
                        'expected accepted=0 and coverage=0, got %s' % stats
                else:
                    assert stats.get('accepted_guide_points') == 80, \
                        'expected K=2 (80 guide points), got %s' % stats
        dist.barrier()
    except Exception:
        traceback.print_exc()
        ok = False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if not ok:
        os._exit(1)


def main():
    world_size = 2
    port = int(os.environ.get('TEST_PORT', 29501))
    print('DDP edge test: world_size=%d' % world_size)
    mp.spawn(run, nprocs=world_size, args=(world_size, port))
    print('DDP edge test PASSED')


if __name__ == '__main__':
    main()
