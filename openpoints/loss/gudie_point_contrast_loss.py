import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import LOSS
import numpy as np

from openpoints.utils.finite_check import assert_finite, configure_finite_check

class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        # No bare squeeze: a K=1 logits tensor must keep the class dimension
        # [1, 1] so CrossEntropy sees out.ndim==2. Flattening the batch dim is
        # the only transform allowed here.
        if x.ndim == 2:
            x = x.reshape(-1, x.shape[-1])
        loss = self.criterion(x, label)
        return loss

@LOSS.register_module()
class GudiePointContrastLoss(nn.Module):
    def __init__(self, npos, T, label_smoothing=0.1, isNorm=True, weight=None,
                 is_guide=False, ignore_index=255, max_guide_iter=10,
                 num_classes=None, confidence_threshold=0.75,
                 class_acceptance_fraction=None):
        super(GudiePointContrastLoss, self).__init__()
        self.T = T
        self.npos = npos
        self.isNorm = isNorm
        self.is_guide = is_guide
        self.ignore_index = ignore_index
        self.max_guide_iter = max_guide_iter
        if isinstance(confidence_threshold, (list, tuple)):
            self.confidence_thresholds = tuple(
                None if value is None else float(value)
                for value in confidence_threshold)
            if not self.confidence_thresholds:
                raise ValueError('confidence_threshold list must not be empty')
            invalid = [value for value in self.confidence_thresholds
                       if value is not None and not 0.0 <= value <= 1.0]
            if invalid:
                raise ValueError('confidence thresholds must be in [0, 1] or '
                                 'null (disabled), got %s' % (invalid,))
        else:
            value = float(confidence_threshold)
            if not 0.0 <= value <= 1.0:
                raise ValueError('confidence_threshold must be in [0, 1], got %s'
                                 % confidence_threshold)
            self.confidence_thresholds = value
        if class_acceptance_fraction is None:
            self.class_acceptance_fractions = None
        else:
            if not isinstance(class_acceptance_fraction, (list, tuple)):
                raise ValueError('class_acceptance_fraction must be a list')
            self.class_acceptance_fractions = tuple(
                None if value is None else float(value)
                for value in class_acceptance_fraction)
            invalid = [value for value in self.class_acceptance_fractions
                       if value is not None and not 0.0 <= value <= 1.0]
            if invalid:
                raise ValueError('class acceptance fractions must be in [0, 1] '
                                 'or null (disabled), got %s' % (invalid,))
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._epoch = None
        self._iteration = None
        self._rank = None
        self.pl_stats = None
        self.nfinite = None
        self._cov_acc = 0.0
        self._acc_acc = 0.0
        self._n_views = 0
        self._diag = {}

    @staticmethod
    def _normalize(feat):
        # F.normalize clamps the norm with eps (1e-12), so a zero row stays 0
        # instead of becoming NaN via 0/0 in a raw division.
        return F.normalize(feat, p=2, dim=1)

    def _reset_diag(self):
        self._cov_acc = 0.0
        self._acc_acc = 0.0
        self._n_views = 0
        self._diag = {
            'nce_skipped': 0,
            'accepted_guide_points': 0,
            'accepted_guide_items': 0,
            'guide_ce_skipped': 0,
            'used_pseudo': False,
            'per_class_counts': [],
            'last_out_shape': None,
            'last_labels_shape': None,
            # pseudo-label diagnostics (aggregated over the batch item loop):
            # confidence = max softmax prob of the pseudo/guide labels.
            'pseudo_total': 0,
            'pseudo_per_class': None,      # [C] ints
            'pseudo_accepted_per_class': None,  # [C] ints
            'pseudo_conf_hist': None,      # [10] ints, bins of [0,1)
            'pseudo_gate_accepted': 0,
            'pseudo_gate_rejected': 0,
        }

    def _check_labels(self, t, C, tag):
        """Labels must be in [0, C) or the ignore index; anything else is a
        hard error (no silent fixing)."""
        if t.dtype != torch.long:
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: {tag} dtype={t.dtype} '
                f'(expected torch.long)')
        valid = (t == self.ignore_index) | ((t >= 0) & (t < C))
        if not bool(valid.all().item()):
            bad = t[~valid].unique().tolist()
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: {tag} contains labels '
                f'out of range: {bad} (valid: [0, {C}) or {self.ignore_index})')

    def _check_ce(self, out, labels, tag):
        """CrossEntropy contract: out must be [K, C], labels must be [K] long
        with values in [0, C). Never relies on a bare squeeze()."""
        if out.ndim != 2:
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: {tag} out.ndim={out.ndim} '
                f'(expected 2, shape={tuple(out.shape)}); a K=1 squeeze collapsed '
                f'the class dimension')
        if out.shape[0] != labels.shape[0]:
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: {tag} shape mismatch '
                f'out={tuple(out.shape)} labels={tuple(labels.shape)}')
        if labels.dtype != torch.long:
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: {tag} labels dtype='
                f'{labels.dtype} (expected torch.long)')
        if not bool(torch.isfinite(out).all().item()):
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: {tag} logits contain '
                f'NaN/Inf (shape={tuple(out.shape)})')
        if not bool(((labels >= 0) & (labels < out.shape[1])).all().item()):
            bad = labels[(labels < 0) | (labels >= out.shape[1])].unique().tolist()
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: {tag} labels out of '
                f'class range: {bad} (C={out.shape[1]})')

    @property
    def _where(self):
        return 'epoch=%s iter=%s rank=%s' % (self._epoch, self._iteration, self._rank)

    def forward(self,_feat1,_feat2,seg_pred1,seg_pred2,target1,target2):
        self._reset_diag()

        # ---- entry validation -------------------------------------------------
        if _feat1.ndim != 3 or _feat2.ndim != 3:
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: expected 3D features, '
                f'got _feat1={tuple(_feat1.shape)} _feat2={tuple(_feat2.shape)}')
        if seg_pred1.ndim != 3 or seg_pred2.ndim != 3:
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: expected 3D logits, '
                f'got seg_pred1={tuple(seg_pred1.shape)} seg_pred2={tuple(seg_pred2.shape)}')
        B = _feat1.shape[0]
        C = seg_pred1.shape[1]
        assert _feat1.shape[0] == _feat2.shape[0] == seg_pred1.shape[0] == seg_pred2.shape[0]
        if torch.isnan(seg_pred1).any() or torch.isnan(seg_pred2).any():
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: NaN in seg_pred '
                f'({tuple(seg_pred1.shape)} / {tuple(seg_pred2.shape)})')
        self._check_labels(target1, C, 'target1')
        self._check_labels(target2, C, 'target2')

        # The unlabeled dataset deliberately returns ignore_index everywhere.
        # PyTorch versions differ on the mean reduction of an all-ignored CE;
        # make the intended result explicit and keep both prediction graphs
        # connected for DDP.
        if bool((target1 != self.ignore_index).any().item()):
            ce_loss = self.criterion(seg_pred1, target1)
        else:
            ce_loss = (seg_pred1.sum() + seg_pred2.sum()) * 0.0

        # Build the point-selection masks once and reuse them in every
        # unsupervised term.  A pseudo point is accepted only when both views
        # are confident and agree on the class.  Ground-truth labels, when
        # present (supervised/unit-test path), are not confidence-filtered.
        selections = []
        if self.class_acceptance_fractions is not None and \
                len(self.class_acceptance_fractions) != C:
            raise ValueError(
                'GudiePointContrastLoss[%s]: got %d class fractions for C=%d'
                % (self._where, len(self.class_acceptance_fractions), C))
        if isinstance(self.confidence_thresholds, tuple):
            if len(self.confidence_thresholds) != C:
                raise ValueError(
                    'GudiePointContrastLoss[%s]: got %d class thresholds for C=%d'
                    % (self._where, len(self.confidence_thresholds), C))
            threshold_values = [float('inf') if value is None else value
                                for value in self.confidence_thresholds]
        else:
            threshold_values = [self.confidence_thresholds] * C
        class_thresholds = torch.as_tensor(
            threshold_values, dtype=seg_pred1.dtype, device=seg_pred1.device)
        for index in range(B):
            half = target1[index].numel() // 2
            labels1 = target1[index][:half]
            labels2 = target2[index][:half]
            is_unlabeled = bool((labels1 == self.ignore_index).all().item())
            if is_unlabeled:
                self._diag['used_pseudo'] = True
                probs1 = F.softmax(seg_pred1[index][:, :half].transpose(0, 1), dim=-1)
                probs2 = F.softmax(seg_pred2[index][:, :half].transpose(0, 1), dim=-1)
                conf1, labels1 = probs1.detach().max(dim=-1)
                conf2, labels2 = probs2.detach().max(dim=-1)
                joint_conf = torch.minimum(conf1, conf2)
                agreement = labels1 == labels2
                if self.class_acceptance_fractions is None:
                    point_thresholds = class_thresholds[labels1]
                    accepted = ((conf1 >= point_thresholds) &
                                (conf2 >= point_thresholds) & agreement)
                else:
                    # Dynamic GMM updates rescale absolute softmax confidence.
                    # Preserve the labeled-set calibrated budget by selecting
                    # the most confident agreeing points within each predicted
                    # class instead of comparing across classes/scales.
                    accepted = torch.zeros_like(agreement)
                    for class_idx, fraction in enumerate(
                            self.class_acceptance_fractions):
                        if fraction is None or fraction <= 0.0:
                            continue
                        candidates = torch.nonzero(
                            agreement & (labels1 == class_idx),
                            as_tuple=False).flatten()
                        if candidates.numel() == 0:
                            continue
                        keep = min(
                            int(candidates.numel()),
                            max(1, int(np.ceil(
                                fraction * int(candidates.numel())))))
                        chosen = torch.topk(
                            joint_conf[candidates], k=keep, largest=True,
                            sorted=False).indices
                        accepted[candidates[chosen]] = True

                diag = self._diag
                if diag['pseudo_conf_hist'] is None:
                    diag['pseudo_per_class'] = [0] * int(C)
                    diag['pseudo_accepted_per_class'] = [0] * int(C)
                    diag['pseudo_conf_hist'] = [0] * 10
                hist = np.histogram(joint_conf.cpu().numpy(), bins=10,
                                    range=(0.0, 1.0))[0]
                raw_classes = labels1.cpu().numpy()
                for c in np.unique(raw_classes):
                    diag['pseudo_per_class'][int(c)] += int((raw_classes == c).sum())
                accepted_classes = labels1[accepted].cpu().numpy()
                for c in np.unique(accepted_classes):
                    diag['pseudo_accepted_per_class'][int(c)] += int(
                        (accepted_classes == c).sum())
                for bin_idx in range(10):
                    diag['pseudo_conf_hist'][bin_idx] += int(hist[bin_idx])
                total = int(accepted.numel())
                naccepted = int(accepted.sum().item())
                diag['pseudo_total'] += total
                diag['pseudo_gate_accepted'] += naccepted
                diag['pseudo_gate_rejected'] += total - naccepted
                self._cov_acc += naccepted / max(float(total), 1.0)
                self._acc_acc += naccepted / max(float(total), 1.0)
                self._n_views += 1
            else:
                accepted = ((labels1 != self.ignore_index) &
                            (labels2 != self.ignore_index) &
                            (labels1 == labels2))
            selections.append((labels1, labels2, accepted, is_unlabeled))

        # ---- NCE contrast term ------------------------------------------------
        # Make features [B, N, C] so rows are points.  Infer the point axis
        # from the segmentation logits, not from C < N (that heuristic fails
        # for deliberately tiny edge-case batches).
        npoints = seg_pred1.shape[2]
        if _feat1.shape[2] == npoints:
            _feat1 = _feat1.transpose(1, 2).contiguous()
            _feat2 = _feat2.transpose(1, 2).contiguous()
        elif _feat1.shape[1] != npoints:
            raise ValueError(
                f'GudiePointContrastLoss[{self._where}]: cannot align feature '
                f'points {_feat1.shape} with logits {seg_pred1.shape}')
        N = _feat1.shape[1]
        # differentiable zero accumulator: keeps the graph connected even when
        # no valid contrast pair can be formed (K=0), avoiding DDP unused-param
        # issues and empty-mean errors.
        loss = (_feat1.sum() + _feat2.sum()) * 0.0
        for index in range(B):
            feat1 = _feat1[index][:N//2]
            feat2 = _feat2[index][:N//2]

            _, _, accepted, is_unlabeled = selections[index]
            if is_unlabeled:
                feat1 = feat1[accepted]
                feat2 = feat2[accepted]

            if self.isNorm:
                feat1 = self._normalize(feat1)
                feat2 = self._normalize(feat2)
            q = feat1
            k = feat2

            if self.npos < q.shape[0]:
                sampled_inds = np.random.choice(q.shape[0], self.npos, replace=False)
                q = q[sampled_inds]
                k = k[sampled_inds]
            npos = q.shape[0]
            if npos < 2:
                # K<2: cannot form positive/negative contrast pairs; contribute
                # a differentiable zero term. Never fabricate pairs by copying.
                self._diag['nce_skipped'] += 1
                continue
            logits = torch.mm(q, k.transpose(1, 0)) # [npos, npos]
            out = torch.div(logits, self.T)
            labels = torch.arange(npos, device=out.device)
            self._check_ce(out, labels, 'nce')
            loss = loss + self.criterion(out, labels)
        loss = loss / B  # keep the original mean-over-batch definition

        # ---- class-guided contrast term ---------------------------------------
        self_loss = _feat1.sum() * 0.0  # differentiable zero accumulator
        for idxx in range(B):
            feat1 = _feat1[idxx][:N//2]
            feat2 = _feat2[idxx][:N//2]
            feat1 = self._normalize(feat1)
            feat2 = self._normalize(feat2)

            tq1, tq2, accepted, _ = selections[idxx]
            tq1 = tq1[accepted]
            tq2 = tq2[accepted]
            feat1 = feat1[accepted]
            feat2 = feat2[accepted]
            qs_idx = tq1.argsort()
            tq1 = tq1[qs_idx]
            tq2 = tq2[qs_idx]
            feat1 = feat1[qs_idx]
            feat2 = feat2[qs_idx]
            q_unique, count = tq1.unique(return_counts=True)
            self._diag['per_class_counts'].append(
                [(int(q_unique[i].item()), int(count[i].item())) for i in range(q_unique.numel())])

            # K=0: no valid pseudo/guide points at all -> differentiable zero.
            if count.numel() == 0:
                self._diag['guide_ce_skipped'] += 1
                continue
            min_iter = min(torch.min(count).item()//3, self.max_guide_iter)
            if min_iter < 1:
                # every class has <3 points: not enough for guide sampling.
                self._diag['guide_ce_skipped'] += 1
                continue
            loss_item = 0
            for iter_num in range(min_iter):
                uniform = torch.distributions.Uniform(0, 1).sample([len(count)]).to(target1.device)
                off = torch.floor(uniform*count).long()
                uniform2 = torch.distributions.Uniform(0, 1).sample([len(count)]).to(target1.device)
                off2 = torch.floor(uniform2*count).long()

                cums = torch.cat([torch.tensor([0], device=count.device), torch.cumsum(count, dim=0)[0:-1]], dim=0)

                _q = feat1[off+cums]
                _k = feat2[off2+cums].clone().detach()
                npos = _q.shape[0]  # one point per class; K>=1 by min_iter>=1
                logits = torch.mm(_q, _k.transpose(1, 0)) # [K, K]
                out = torch.div(logits, self.T)
                labels = torch.arange(npos, device=out.device)
                # K=1 keeps out as [1, 1] (class dim preserved, no squeeze).
                self._check_ce(out, labels, 'guide')
                loss_item += self.criterion(out, labels)
            loss_item /= min_iter
            self_loss = self_loss + loss_item
            self._diag['accepted_guide_points'] += min_iter * (tq1.unique().numel())
            self._diag['accepted_guide_items'] += 1
            self._diag['last_out_shape'] = tuple(out.shape)
            self._diag['last_labels_shape'] = tuple(labels.shape)
        self_loss = self_loss / B

        totle_loss = self_loss + 0.1*loss + 0.02*ce_loss
        # per-component finiteness (report before the total)
        assert_finite(ce_loss, 'guide loss ce component', epoch=self._epoch,
                      iteration=self._iteration, rank=self._rank)
        assert_finite(loss, 'guide loss nce component', epoch=self._epoch,
                      iteration=self._iteration, rank=self._rank)
        assert_finite(self_loss, 'guide loss guide component', epoch=self._epoch,
                      iteration=self._iteration, rank=self._rank)
        assert_finite(totle_loss, 'guide loss total', epoch=self._epoch,
                      iteration=self._iteration, rank=self._rank)

        # ---- diagnostics ------------------------------------------------------
        if self._n_views > 0:
            self.pl_stats = {'coverage': self._cov_acc / self._n_views,
                             'accepted_ratio': self._acc_acc / self._n_views}
        else:
            self.pl_stats = {'coverage': 0.0, 'accepted_ratio': 0.0}
        self.pl_stats.update({
            'accepted_guide_points': self._diag['accepted_guide_points'],
            'accepted_guide_items': self._diag['accepted_guide_items'],
            'guide_ce_skipped': self._diag['guide_ce_skipped'],
            'nce_skipped': self._diag['nce_skipped'],
            'used_pseudo': self._diag['used_pseudo'],
            'pseudo_total': self._diag['pseudo_total'],
            'pseudo_per_class': self._diag['pseudo_per_class'],
            'pseudo_accepted_per_class': self._diag[
                'pseudo_accepted_per_class'],
            'pseudo_conf_hist': self._diag['pseudo_conf_hist'],
            'pseudo_gate_accepted': self._diag['pseudo_gate_accepted'],
            'pseudo_gate_rejected': self._diag['pseudo_gate_rejected'],
            'gate_thresholds': list(self.confidence_thresholds)
            if isinstance(self.confidence_thresholds, tuple)
            else [self.confidence_thresholds] * C,
            'gate_fractions': list(self.class_acceptance_fractions)
            if self.class_acceptance_fractions is not None else None,
        })
        self.nfinite = _feat1.reshape(-1, _feat1.shape[-1]).isfinite().all(dim=-1).float().mean().item()
        return totle_loss
