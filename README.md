# SemiGMMPoint-HG

This repository is a **reproduction and dataset-adapted version** of the paper **SemiGMMPoint: Semi-supervised Point Cloud Segmentation based on Gaussian Mixture Models**, modified for the custom ODPT / ODPT-HG datasets. It is NOT the official code released by the paper authors.

> 说明：本仓库是基于原论文方法、为适配自定义 ODPT / ODPT-HG 数据集而修改后的复现与适配版本，并非论文作者发布的官方代码。

## Project Overview

- Semi-supervised 3D point cloud segmentation framework based on Gaussian Mixture Models
- Supports multiple backbone models: PointNet++, PointNeXt-L, etc.
- Supports S3DIS, ScanNetv2, ShapeNetPart, SemanticKITTI datasets

## Requirements

This codebase was tested with the following environment configurations. The corresponding version of the installation library is required. You must install these environments independently according to the official documentation:

- Python 3.7.11
- CUDA 11.4
- Pytorch 1.9.0
- GCC version 7.5.0
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) 0.4.3
- [MMCV](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html) 2.0.0

Other necessary installation libraries embedded in the source code include the following parts:

- grid_subsampling library
- pointnet++ library
- point transformer library
- chamfer_dist library

## Usage

### 1. Install Dependencies

Install Dependent Package:

```shell
pip install -r requirements.txt
```

if you have any problem with the above command, you can Install some key installation packages by:

```shell
pip install torch_sparse==0.6.12
pip install torch_points3d==1.3.0
pip install tensorboard timm termcolor tensorboardX
```

### 2. Install Optional Dependencies

Make sure you have installed `gcc` and `cuda`, and `nvcc` can work.

- Install cpp extensions, the pointnet++ library：

```shell
cd openpoints/cpp/pointnet2_batch
python setup.py install
```

- Install point transformer library

```shell
cd openpoints/cpp/pointops/
python setup.py install
```

### 3. Datasets Preparation

##### S3DIS

Please refer to https://github.com/yanx27/Pointnet_Pointnet2_pytorch for S3DIS preprocessing.

##### ScanNetv2

Please refer to https://github.com/facebookresearch/PointContrast for ScanNetV2 preprocessing.

##### ShapeNetPart

Please refer to https://github.com/lulutang0608/Point-BERT for ShapeNetPart preprocessing.

##### SemanticKITTI

Please refer to https://github.com/PRBonn/semantic-kitti-api for SemanticKITTI preprocessing

> The location for modifying the dataset path is: `SemiGMMPoint\cfgs\...\Name.cfg` and `dataset` data structure.

### 4. Training and Testing Examples

The configuration files of model and training parameter are in `SemiGMMPoint\cfgs\...`

Example of modifying classifier parameters in configuration files:

```yaml
decoder_params:
    embed_dim: 64
    num_components: 5
    factor_n: 1
    factor_c: 1
    factor_p: 1,
    mem_size: 36000
    max_sample_size: 40
    update_GMM_interval: 5
    update_loop: 5
```

The modification of other runtime parameters, including epochs, batch size, learning rate, and so on, is still implemented in the configuration file.

#### With the GMM classifier

- PointNet++

Setting dataset root path in `train_gmmseg_semseg.py`:

```python
root = ''
```

Use the following code for training:

```shell
cd pointnet2based/
python train_gmm_semseg.py --model gmm_pointnet2_seg_msg --test_area 5 --log_dir gmm_pointnet2_sem_seg
```

Use the following code for testing:

```
cd pointnet2based/
python test_semseg.py --test_area 5 --log_dir gmm_pointnet2_sem_seg
```

- PointNeXt-L

Setting dataset root path and other settings in `cfgs\scannet\gmmseg_pointnext-l.yaml`:

Use the following code for training

```shell
cd SemiGMMPoint/examples/segmentation
python examples/segmentation/gmm_main.py --cfg cfgs/scannet/gmm_pointnext-l.yaml
```

Use the following code for testing:

```shell
python examples/segmentation/gmm_main.py --cfg cfgs/scannet/gmm_pointnext-l.yaml mode=test dataset.test.split=test no_label=True pretrained_path=<YOUR_CHECKPOINT_PATH>
```

#### SemiGMMPoint in semi-supervised settings

Set data root paths for annotated and unlabeled datasets and other necessary parameters in the configuration file, such as `10r_semi_gmm_pointnext-l.yaml`

Use the following code for training:

```shell
cd SemiGMMPoint/examples/segmentation
python examples/segmentation/semi_gmmpoint_main.py --cfg cfgs/scannet/10r_semi_gmm_pointnext-l.yaml
```

Use the following code for testing:

```shell
python examples/segmentation/semi_gmmpoint_main.py --cfg cfgs/scannet/10r_semi_gmm_pointnext-l.yaml mode=test dataset.test.split=test no_label=True pretrained_path=<YOUR_CHECKPOINT_PATH>
```

### Model Checkpoints and Logs

*Checkpoints will be made public online after paper is accepted.

## References

The code framework is based on [openpoints](https://github.com/guochengqian/PointNeXt).

Some experiments involve open source repositories:

- https://github.com/guochengqian/PointNeXt
- https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- https://github.com/lulutang0608/Point-BERT
- https://github.com/facebookresearch/PointContrast/
- https://github.com/dvlab-research/Stratified-Transformer
