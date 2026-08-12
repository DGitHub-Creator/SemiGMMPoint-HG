import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, crop_pc_pre
from ..build import DATASETS

ODPT_CLASSES = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window']
ODPT_NUM_CLASSES = 6
ODPT_NUM_PER_CLASS = np.array([4188864, 2954756, 4376810, 749576, 1371500, 18541931], dtype=np.int32)


def normalize_scene_name(name):
    """'Area_1_conferenceRoom_1.pth\\n' -> 'Area_1_conferenceRoom_1'"""
    return str(name).strip()


def read_split_list(data_root, split_file):
    """Read a scene list file (one scene stem per line). Returns matched existing scenes."""
    if split_file is None:
        return []
    split_path = split_file if os.path.isabs(split_file) else os.path.join(data_root, split_file)
    assert os.path.exists(split_path), f'split file not found: {split_path}'
    names = [normalize_scene_name(line) for line in open(split_path, 'r') if normalize_scene_name(line)]
    existing = sorted(set(os.listdir(data_root)))
    matched = [n for n in names if n + '.pth' in existing]
    assert matched, f'no scene in split file {split_path} matches files in {data_root}'
    return matched


def list_area_scenes(data_root, area_prefix='Area_1'):
    """All scene stems in data_root whose file name starts with area_prefix."""
    return sorted(n[:-4] for n in os.listdir(data_root)
                  if n.endswith('.pth') and n.startswith(area_prefix))


def load_odpt_scene(data_root, scene_name, with_label=True):
    """Load one ODPT scene .pth (torch tuple: coord, color, label).

    coord: (N,3) float32, mean-centered (crop_pc shifts by min internally).
    color: (N,3) float32 in [-1,1] -> converted to [0,255].
    label: (N,) values in {0..5}.

    with_label=False: do NOT read the GT label array from disk (used by the
    unsupervised/pretrain branches, which must never touch real labels). The
    returned label is a zeros placeholder of the same length; callers replace
    it with 255 (ignore) before use, so no GT value ever reaches the loss.
    """
    path = os.path.join(data_root, scene_name + '.pth')
    coord, color, label = torch.load(path, map_location='cpu')
    coord = np.asarray(coord, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    label = np.asarray(label).reshape(-1).astype(np.long)
    color = np.clip((color + 1.0) * 127.5, 0.0, 255.0).astype(np.float32)
    if not with_label:
        label = np.zeros(coord.shape[0], dtype=np.long)
    return coord, color, label


@DATASETS.register_module()
class ODPTS3DIS(Dataset):
    """ODPT 6-class S3DIS subset.

    Scene-level semi-supervised protocol:
      labeled train  : scenes listed in split_file (e.g. splits/10.txt, splits/20.txt)
      unlabeled train: all Area_1 scenes not in split_file and not in val_scenes
      val            : fixed Area_1 scenes given by val_scenes (Area_3 must NOT be used)
      test           : Area_3 scenes given by test_scenes (or all Area_3 files)
    Colors are stored in [0,255]; labels in {0..5}.
    """
    classes = ODPT_CLASSES
    num_classes = ODPT_NUM_CLASSES
    num_per_class = ODPT_NUM_PER_CLASS
    class2color = {'ceiling': [0, 255, 0],
                   'floor':   [0, 0, 255],
                   'wall':    [0, 255, 255],
                   'beam':    [255, 255, 0],
                   'column':  [255, 0, 255],
                   'window':  [100, 100, 255]}
    cmap = [*class2color.values()]
    gravity_dim = 2

    def __init__(self,
                 data_root: str = 'data/odpt_semigmm',
                 split_file: str = None,
                 mode: str = 'labeled',
                 split: str = 'train',
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 val_scenes=None,
                 test_scenes=None,
                 ):
        super().__init__()
        self.data_root = data_root
        assert os.path.isdir(data_root), f'data_root not found: {data_root}'
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle
        all_area1 = list_area_scenes(data_root, 'Area_1')
        assert all_area1, f'no Area_1 scenes under {data_root}'
        if mode == 'unlabeled':
            labeled = read_split_list(data_root, split_file)
            self.data_list = [s for s in all_area1 if s not in labeled]
            logging.info('ODPTS3DIS audit: unlabeled pool = %d Area_1 scenes %s; '
                         'labels replaced by ignore (255) in __getitem__; '
                         'no Area_3 scene instantiated'
                         % (len(self.data_list), sorted(self.data_list)))
        elif mode == 'val':
            val = [normalize_scene_name(s) for s in (val_scenes or [])]
            self.data_list = [s for s in all_area1 if s in val]
            assert self.data_list, 'val_scenes must be Area_1 scenes, got %s' % (val,)
            logging.info('ODPTS3DIS audit: WARNING validation dataset constructed with GT '
                         '(val scenes %s) — protocol requires validation=disabled'
                         % sorted(self.data_list))
        elif mode == 'test':
            test = [normalize_scene_name(s) for s in (test_scenes or [])]
            self.data_list = [s for s in test if os.path.exists(os.path.join(data_root, s + '.pth'))]
            if not self.data_list:
                self.data_list = list_area_scenes(data_root, 'Area_3')
            assert self.data_list, 'no test scenes found under %s' % data_root
            logging.info('ODPTS3DIS audit: test set = %d Area_3 scenes %s (only used by '
                         'the official post-training evaluator)'
                         % (len(self.data_list), sorted(self.data_list)))
        else:  # labeled train
            self.data_list = read_split_list(data_root, split_file)
            assert self.data_list, 'labeled split file contains no Area_1 scenes'
            logging.info('ODPTS3DIS audit: labeled train = %d Area_1 scenes %s from split file %s'
                         % (len(self.data_list), sorted(self.data_list), split_file))
        self.data_idx = np.arange(len(self.data_list))
        logging.info(f"\nTotally {len(self.data_idx)} samples in {mode}/{split} set")
        assert len(self.data_idx) > 0

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        coord, color, label = load_odpt_scene(self.data_root, self.data_list[data_idx])
        label = label.reshape(-1, 1).astype(np.float32)
        coord, color, label = crop_pc(
            coord, color, label, self.split, self.voxel_size, self.voxel_max,
            downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
        label = label.squeeze(-1).astype(np.long)
        data = {'pos': coord, 'x': color, 'y': label}
        if self.transform is not None:
            data = self.transform(data)
        if 'heights' not in data.keys():
            data['heights'] = torch.from_numpy(
                coord[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32))
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop


@DATASETS.register_module()
class ODPTPreS3DIS(Dataset):
    """ODPT unlabeled dataset (dual-view, no GT).

    Labels are replaced by 255 (ignore) so that no ground-truth leaks into the
    unsupervised branch; the point-contrast loss falls back to pseudo labels.
    Each scene is split into 3 equal parts; view1 = part1+part2, view2 = part1+part3.
    """
    classes = ODPT_CLASSES
    num_classes = ODPT_NUM_CLASSES
    num_per_class = ODPT_NUM_PER_CLASS
    class2color = ODPTS3DIS.class2color
    cmap = [*class2color.values()]
    gravity_dim = 2

    def __init__(self,
                 data_root: str = 'data/odpt_semigmm',
                 split_file: str = None,
                 mode: str = 'unlabeled',
                 split: str = 'train',
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 val_scenes=None,
                 test_scenes=None,
                 ):
        super().__init__()
        self.data_root = data_root
        assert os.path.isdir(data_root), f'data_root not found: {data_root}'
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle
        all_area1 = list_area_scenes(data_root, 'Area_1')
        assert all_area1, f'no Area_1 scenes under {data_root}'
        if mode == 'pretrain':
            self.data_list = [s for s in all_area1]
            logging.info('ODPTPreS3DIS pretrain pool: %d scenes (all Area_1, split file ignored)'
                         % len(self.data_list))
        else:
            labeled = read_split_list(data_root, split_file)
            self.data_list = [s for s in all_area1 if s not in labeled]
            logging.info('ODPTPreS3DIS audit: unlabeled pool = %d Area_1 scenes %s; '
'labels replaced by ignore (255) in __getitem__ before returning; '
                         'no Area_3 scene instantiated'
                         % (len(self.data_list), sorted(self.data_list)))
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} unlabeled samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        coord, color, label = load_odpt_scene(
            self.data_root, self.data_list[data_idx], with_label=False)
        label = label.reshape(-1, 1).astype(np.float32)
        coord, color, label = crop_pc_pre(
            coord, color, label, self.split, self.voxel_size, self.voxel_max,
            downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
        num_pointpair = len(label) // 3
        coord1 = np.concatenate((coord[:num_pointpair], coord[num_pointpair:num_pointpair * 2]), axis=0)
        color1 = np.concatenate((color[:num_pointpair], color[num_pointpair:num_pointpair * 2]), axis=0)
        label1 = np.concatenate((label[:num_pointpair], label[num_pointpair:num_pointpair * 2]), axis=0)
        coord2 = np.concatenate((coord[:num_pointpair], coord[num_pointpair * 2:num_pointpair * 3]), axis=0)
        color2 = np.concatenate((color[:num_pointpair], color[num_pointpair * 2:num_pointpair * 3]), axis=0)
        label2 = np.concatenate((label[:num_pointpair], label[num_pointpair * 2:num_pointpair * 3]), axis=0)
        label1 = np.full(label1.shape, 255, dtype=np.long).squeeze(-1)
        label2 = np.full(label2.shape, 255, dtype=np.long).squeeze(-1)
        if 'train' not in self.split:
            data1 = {'pos': coord1, 'x': color1, 'y': label1}
            if self.transform is not None:
                data1 = self.transform(data1)
            if 'heights' not in data1.keys():
                data1['heights'] = torch.from_numpy(
                    coord1[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32))
            return data1
        data1 = {'pos': coord1, 'x': color1, 'y': label1}
        data2 = {'pos': coord2, 'x': color2, 'y': label2}
        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
        if 'heights' not in data1.keys():
            data1['heights'] = torch.from_numpy(
                coord1[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32))
        if 'heights' not in data2.keys():
            data2['heights'] = torch.from_numpy(
                coord2[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32))
        return [data1, data2]

    def __len__(self):
        return len(self.data_idx) * self.loop
