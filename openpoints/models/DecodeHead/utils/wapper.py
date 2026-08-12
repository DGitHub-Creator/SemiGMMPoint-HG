import warnings

import torch.nn as nn
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True,
           D=2):
    if warning and size is not None and align_corners and mode not in ('nearest', 'area'):
        warnings.warn(
            'size is not None and align_corners is True, may cause mis-aligned results. '
            'Set warning=False to disable this warning.')
    if input.dim() == 5 and D == 1 and mode in ('bilinear', 'linear'):
        mode = 'trilinear'
    if mode in ('nearest', 'area'):
        align_corners = None
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)
