# -*- coding: utf-8 -*-
"""Finite-value diagnostics for SemiGMMPoint.

assert_finite reports the FIRST tensor that becomes non-finite with full
context (epoch / iteration / rank / stage / tensor stats / counts), then
terminates ALL DDP ranks safely (dist.destroy_process_group + os._exit) so
that no rank is left hanging inside a collective op.

This is a diagnostic gate, not a fix: a non-finite tensor must be root-caused
and the producing operation fixed. Only enabled when cfg.num_debug_gmm=True.
"""
import os
import sys
import logging

import torch

_DISABLED = False
_ENABLED = False


def configure_finite_check(enabled):
    """Enable/disable the global finite check (called once from the entry script)."""
    global _ENABLED
    _ENABLED = bool(enabled)


def _report(tag, t, ctx):
    lines = [
        '=' * 80,
        'FINITE_CHECK FAILURE (first non-finite tensor)',
        '  tensor      : %s' % tag,
        '  shape       : %s' % (tuple(t.shape),),
        '  dtype       : %s' % t.dtype,
        '  numel       : %d' % t.numel(),
    ]
    for k, v in ctx.items():
        lines.append('  %-12s: %s' % (k, v))
    ft = t.float() if not t.is_floating_point() else t
    lines.append('  NaN count   : %d' % int(torch.isnan(ft).sum().item()))
    lines.append('  +Inf count  : %d' % int((ft == float('inf')).sum().item()))
    lines.append('  -Inf count  : %d' % int((ft == float('-inf')).sum().item()))
    lines.append('  min         : %s' % ('%.6g' % ft.min().item() if t.numel() else 'n/a'))
    lines.append('  max         : %s' % ('%.6g' % ft.max().item() if t.numel() else 'n/a'))
    lines.append('  mean        : %s' % ('%.6g' % ft.mean().item() if t.numel() else 'n/a'))
    lines.append('  std         : %s' % ('%.6g' % ft.std().item() if t.numel() > 1 else 'n/a'))
    msg = '\n'.join(lines)
    logging.error(msg)
    print(msg, flush=True)
    try:
        torch.distributed.barrier()  # let the message flush on every rank
    except Exception:
        pass
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    os._exit(1)


def assert_finite(t, tag, epoch=None, iteration=None, rank=None, **ctx):
    """Raise-abort when `t` contains NaN/Inf.

    `ctx` may carry scene names, augmentation params, GMM state summaries, loss
    and lr values, GMM update round, etc. and is printed in the report.
    """
    if _DISABLED or not _ENABLED:
        return
    if t is None:
        return
    if t.numel() == 0:
        return
    if not t.is_floating_point():
        return
    info = {'epoch': epoch, 'iteration': iteration, 'rank': rank}
    info.update(ctx)
    if not torch.isfinite(t).all():
        _report(tag, t, info)


def assert_grads_finite(model, tag, epoch=None, iteration=None, rank=None, **ctx):
    """Check every parameter gradient is finite and return the global grad norm."""
    if _DISABLED or not _ENABLED:
        return
    total_norm = 0.0
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        total_norm += p.grad.data.float().norm().item() ** 2
        if not torch.isfinite(p.grad.data).all():
            bad.append(name)
    total_norm = total_norm ** 0.5
    if bad:
        _report('%s gradient (first bad: %s)' % (tag, bad[0]),
                model.get_parameter(bad[0]).grad,
                dict(epoch=epoch, iteration=iteration, rank=rank,
                     bad_params=bad[:20], global_grad_norm='%.6g' % total_norm, **ctx))
    return total_norm


def assert_params_finite(model, tag, epoch=None, iteration=None, rank=None, **ctx):
    """Check every model parameter is finite after optimizer.step()."""
    if _DISABLED or not _ENABLED:
        return
    bad = [n for n, p in model.named_parameters()
           if not torch.isfinite(p.data).all()]
    if bad:
        _report('%s parameter (first bad: %s)' % (tag, bad[0]),
                dict(model.named_parameters())[bad[0]].data,
                dict(epoch=epoch, iteration=iteration, rank=rank,
                     bad_params=bad[:20], **ctx))
