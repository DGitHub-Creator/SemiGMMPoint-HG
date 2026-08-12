"""HG (ODPT-HG) dataset classes for SemiGMMPoint.

Scene-level semi-supervised protocol, same as ODPTS3DIS/ODPTPreS3DIS but:

* class names follow the HG class_statistics manifest:
  0 pipeline, 1 steel_frame, 2 elbow_pipe, 3 valve_guardrail,
  4 gate_valve, 5 Christmas_tree_body
* num_per_class (-> class weights) is recomputed **at runtime from the real GT
  of the labeled scenes of the current split only** (never from the full set),
  satisfying "class weights must come from labeled GT of the split".

Everything else (loading, crop_pc, dual-view unlabeled protocol, y=255 label
replacement) is inherited from odpt.py unchanged.
"""
import logging

import numpy as np

from ..build import DATASETS
from .odpt import (
    ODPTPreS3DIS,
    ODPTS3DIS,
    list_area_scenes,
    load_odpt_scene,
    read_split_list,
)

HG_CLASSES = ['pipeline', 'steel_frame', 'elbow_pipe', 'valve_guardrail',
              'gate_valve', 'Christmas_tree_body']


def _label_counts(data_root, scenes):
    counts = np.zeros(6, dtype=np.int64)
    for s in scenes:
        _, _, label = load_odpt_scene(data_root, s)
        counts += np.bincount(np.asarray(label).astype(np.int64), minlength=6)
    return counts


def _set_hg_num_per_class(dataset, data_root, split_file, mode):
    """Recompute per-split class point counts / weights from labeled GT only.

    Compliance:
      * 'labeled' : counts come from the GT of the labeled split scenes only
                    (scene list is logged) -> feeds the supervised loss
                    weights. Never touches the unlabeled pool or Area_3.
      * 'test'    : Area_3 GT counts, statistics for the eval report only.
      * else (unlabeled/pretrain): GT is NEVER read (num_per_class=None);
                    nothing derived from unlabeled GT exists.
    """
    if mode == 'labeled':
        labeled = read_split_list(data_root, split_file)
        counts = _label_counts(data_root, labeled)
        dataset.num_per_class = counts.astype(np.int32)
        logging.info(
            'ODPT-HG: class weights recomputed from labeled split GT only '
            '(split %s, %d scenes %s): %s' % (split_file, len(labeled),
                                              sorted(labeled),
                                              counts.tolist()))
        dataset._labeled_counts = counts
    elif mode == 'test':
        test = list_area_scenes(data_root, 'Area_3')
        counts = _label_counts(data_root, test)
        dataset.num_per_class = counts.astype(np.int32)
        logging.info('ODPT-HG test GT counts (Area_3): %s' % counts.tolist())
    else:
        # unlabeled / pretrain: no GT read at all (labels are replaced by
        # zeros-on-disk + 255 in the batch). num_per_class must stay None so
        # any GT-derived weighting is impossible.
        dataset.num_per_class = None
        logging.info(
            'ODPT-HG: mode=%s -> GT labels NOT read, num_per_class=None '
            '(no class weights can be derived from this set)' % mode)


@DATASETS.register_module()
class ODPTS3DISHG(ODPTS3DIS):
    classes = HG_CLASSES
    num_classes = 6

    def __init__(self, dataset_root=None, *args, **kwargs):
        # dataset_root (raw HG txt root) is documentation-only; all reading
        # happens through data_root's converted .pth files.
        super().__init__(*args, **kwargs)
        mode = kwargs.get('mode', 'labeled')
        _set_hg_num_per_class(self, self.data_root, kwargs.get('split_file'),
                              mode)
        if mode == 'unlabeled':
            logging.info('ODPT-HG protocol: unlabeled batch labels unique = [255]')


@DATASETS.register_module()
class ODPTPreS3DISHG(ODPTPreS3DIS):
    classes = HG_CLASSES
    num_classes = 6

    def __init__(self, dataset_root=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = kwargs.get('mode', 'unlabeled')
        _set_hg_num_per_class(self, self.data_root, kwargs.get('split_file'),
                              mode if mode != 'pretrain' else 'unlabeled')
        logging.info('ODPT-HG protocol: unlabeled batch labels unique = [255]')