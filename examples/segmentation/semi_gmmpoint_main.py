"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""
import os
import sys
import warnings

# ---- warning governance (before any third-party import) --------------------
# MMCV 2.0 release announcement, tqdm/torch deprecations, etc.: informational
# only for this long-running training process; ignore to keep logs clean.
warnings.filterwarnings("ignore")

import __init__
import argparse, yaml,  logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.utils.finite_check import configure_finite_check, assert_finite, assert_grads_finite, assert_params_finite
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg

warnings.simplefilter(action='ignore', category=FutureWarning)


def build_pbar(iterable, cfg, desc=None, total=None):
    """tqdm helpers: rank-0 only, ascii-safe output for non-interactive logs."""
    if total is None:
        total = len(iterable) if hasattr(iterable, '__len__') else None
    if cfg.get('tqdm_rank0_only', False) and cfg.rank != 0:
        return tqdm(iterable, total=total, desc=desc,
                    ascii=True, dynamic_ncols=False, disable=True)
    return tqdm(iterable, total=total, desc=desc,
                ascii=cfg.get('tqdm_ascii', True), dynamic_ncols=False)


    

def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True, area=5):
    ious_table = [f'{item:.2f}' for item in ious]
    header = ['method', 'Area', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.cfg_basename, str(area), f'{oa:.2f}', f'{macc:.2f}',
            f'{miou:.2f}'] + ious_table + [str(best_epoch), cfg.run_dir,
                                           wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def generate_data_list(cfg):
    if 'odpt' in cfg.dataset.common.NAME.lower():
        raw_root = cfg.dataset.common.data_root
        data_list = sorted(os.path.join(raw_root, item) for item in os.listdir(raw_root)
                           if item.endswith('.pth') and 'Area_3' in item)
        assert data_list, f'no Area_3 test scenes under {raw_root}'
    elif 's3dis' in cfg.dataset.common.NAME.lower():
        # Modify this location
        cfg.dataset.common.data_root = ""
        raw_root =  ""
        data_list = sorted(os.listdir(raw_root))
        data_list = [os.path.join(raw_root, item) for item in data_list if
                     'Area_{}'.format(cfg.dataset.common.test_area) in item]
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data_list = glob.glob(os.path.join(cfg.dataset.common.data_root, cfg.dataset.test.split, "*.pth"))
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        if cfg.dataset.test.split == 'val':
            split_no = 1
        else:
            split_no = 2
        data_list = get_semantickitti_file_list(os.path.join(cfg.dataset.common.data_root, 'sequences'),
                                                str(cfg.dataset.test.test_id + 11))[split_no]
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    return data_list


def load_data(data_path, cfg):
    label, feat = None, None
    if 'odpt' in cfg.dataset.common.NAME.lower():
        data = torch.load(data_path)  # tuple (coord, color[-1,1], label)
        coord, feat, label = data[0], data[1], data[2]
        feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)
    elif 's3dis' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat = data[0], data[1]
        if cfg.dataset.test.split != 'test':
           label = data[2]
        else:
            label = None
        feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        coord = load_pc_kitti(data_path[0])
        if cfg.dataset.test.split != 'test':
            label = load_label_kitti(data_path[1], remap_lut_read)
    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part,reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points  
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max()+1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle] # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0) # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        # NCCL device mapping: bind the process to its local GPU BEFORE any
        # collective/barrier (avoids the "best-guess GPU" warning and hang risk).
        torch.cuda.set_device(gpu)
        device = torch.device('cuda', gpu)
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier(device_ids=[gpu])
    else:
        device = torch.device('cuda', cfg.rank)

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.distributed:
        logging.info(
            'NCCL rank=%d global | local gpu=%d | device=%s (%s) | pid=%d' % (
                cfg.rank, gpu, device,
                torch.cuda.get_device_name(gpu), os.getpid()))
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # NaN/Inf diagnostics: configurable via `num_debug_gmm: true` in the yaml.
    configure_finite_check(cfg.get('num_debug_gmm', False))

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank,find_unused_parameters=True)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # full-state resume: model + optimizer + scheduler + epoch + GMM buffers.
    # The DDP wrapper is removed first so state_dict keys match; then re-wrapped.
    if cfg.resume_from is not None:
        if not os.path.isfile(cfg.resume_from):
            logging.error('resume_from checkpoint not found: %s' % cfg.resume_from)
            sys.exit(1)
        raw_model = model.module if hasattr(model, 'module') else model
        resume_checkpoint(cfg, raw_model, optimizer, scheduler,
                          pretrained_path=cfg.resume_from)
        logging.info('RESUMED from %s (start_epoch=%d)' % (cfg.resume_from, cfg.start_epoch))
    else:
        logging.info('Training from %s' % cfg.mode)


    # build dataset
    # ODPT protocol: when True, validation is fully disabled (no val Dataset/
    # DataLoader constructed, no best-val checkpoint selection). Official ODPT
    # entries force this via CLI (--disable_validation True).
    disable_validation = cfg.get('disable_validation', False)
    if disable_validation:
        # ODPT protocol: validation is fully disabled -> no val Dataset, no val
        # DataLoader, no val_scenes file access, no best-val bookkeeping.
        val_loader = None
        num_classes = cfg.num_classes
        cfg.classes = np.arange(num_classes)
        cfg.cmap = None
        logging.info('ODPT protocol: validation dataset NOT constructed '
                     '(validation=disabled)')
    else:
        val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                               cfg.dataset,
                                               cfg.dataloader,
                                               datatransforms_cfg=cfg.datatransforms,
                                               split='val',
                                               distributed=cfg.distributed
                                               )
        logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
        num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
        if num_classes is not None:
            assert cfg.num_classes == num_classes
        cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.arange(num_classes)
        cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None
    logging.info(f"number of classes of the dataset: {num_classes}")
    validate_fn = validate if 'sphere' not in cfg.dataset.common.NAME.lower() else validate_sphere

    model_module = model.module if hasattr(model, 'module') else model
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
        else:
            if cfg.mode == 'val':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=1)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {val_oa:.2f} {val_macc:.2f} {val_miou:.2f}, '
                        f'\niou per cls is: {val_ious}')
                return val_miou
            elif cfg.mode == 'test':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                data_list = generate_data_list(cfg)
                logging.info(f"length of test dataset: {len(data_list)}")
                test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)

                if test_miou is not None:
                    with np.printoptions(precision=2, suppress=True):
                        logging.info(
                            f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {test_oa:.2f} {test_macc:.2f} {test_miou:.2f}, '
                            f'\niou per cls is: {test_ious}')
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                    write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg)
                return test_miou

            elif 'encoder' in cfg.mode:
                logging.info(f'pretrained_path={cfg.pretrained_path}')
                logging.info('pretrained_exists=%s' % os.path.isfile(cfg.pretrained_path))
                load_checkpoint(model_module.encoder, cfg.pretrained_path, cfg.get('pretrained_module', None))
                loaded_params = sum(p.numel() for p in model_module.encoder.parameters())
                total_params = sum(p.numel() for p in model_module.parameters())
                logging.info('loaded encoder parameters=%d (%.4f of total %.0f)'
                             % (loaded_params, loaded_params / max(total_params, 1),
                                total_params))
            else:
                logging.info(f'pretrained_path={cfg.pretrained_path}')
                logging.info('pretrained_exists=%s' % os.path.isfile(cfg.pretrained_path))
                load_checkpoint(model, cfg.pretrained_path, cfg.get('pretrained_module', None))
                loaded_params = sum(p.numel() for p in model_module.parameters())
                enc_params = sum(p.numel() for p in model_module.encoder.parameters())
                total_params = sum(p.numel() for p in model_module.parameters())
                logging.info('loaded model parameters=%d (encoder=%d) = %.4f of total %.0f'
                             % (loaded_params, enc_params,
                                loaded_params / max(total_params, 1), total_params))
    else:
        logging.info('Training from scratch')

    # exit(0)
    if 'freeze_blocks' in cfg.mode:
        for p in model_module.encoder.blocks.parameters():
            p.requires_grad = False

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    un_train_loader = build_dataloader_from_cfg(cfg.unbatch_size,
                                             cfg.undataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    logging.info(f"length of Unsupervised training dataset: {len(un_train_loader.dataset)}")

    if 'odpt' in cfg.dataset.common.NAME.lower():
        logging.info(
            'ODPT protocol: labeled train %d scenes; unlabeled pool %d scenes; '
            'validation=%s; Area_3 is NOT used during training.'
            % (len(train_loader.dataset.data_list), len(un_train_loader.dataset.data_list),
               'disabled' if disable_validation else 'enabled'))

    cfg.criterion_args.weight = None
    if cfg.get('cls_weighed_loss', False):
        num_per_class = getattr(train_loader.dataset, 'num_per_class', None)
        if num_per_class is not None:
            cfg.criterion_args.weight = get_class_weights(num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` is None on the labeled dataset; '
                         'class weights unavailable -> unweighted CE')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    uncriterion = build_criterion_from_cfg(cfg.uncriterion_args).cuda()

    # ===> start training
    if cfg.rank == 0:
        logging.info('ODPT protocol: validation=DISABLED '
                     '(checkpoint_selection=FINAL_EPOCH_ONLY, test_area_evaluation=DID_NOT_RUN)')

    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if disable_validation:
        # No validation anywhere: no best-val state, no val-based checkpoint
        # selection, and the epoch log carries training metrics only.
        val_miou, val_macc, val_oa = None, None, None
        best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
        last_epoch_unsup = False
    else:
        val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
        best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        if epoch > cfg.max_unsupervised and epoch % cfg.unbranch_freq==0:
            last_epoch_unsup = True
            train_loss, train_miou, train_macc, train_oa, _, _ = \
                un_train_one_epoch(model, un_train_loader, uncriterion, optimizer, scheduler, scaler, epoch, cfg)

        else:
            last_epoch_unsup = False
            train_loss, train_miou, train_macc, train_oa, _, _ = \
                train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg)

        is_best = False
        if not disable_validation and epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
                        f'\nmious: {val_ious}')

        lr = optimizer.param_groups[0]['lr']
        if disable_validation:
            # ODPT protocol mode: training metrics only. Unavailable metrics are
            # reported as N/A, never as placeholder -1/0 values.
            if last_epoch_unsup:
                last_stats = getattr(uncriterion, 'pl_stats_last', None) \
                    or uncriterion.pl_stats or {}
                logging.info(
                    f'Epoch {epoch} LR {lr:.6f} supervised_train_mIoU=N/A '
                    f'unsup_loss {train_loss:.4f} pseudo_label_ratio '
                    f'{last_stats.get("accepted_ratio", float("nan")):.4f} '
                    f'coverage {last_stats.get("coverage", float("nan")):.4f}')
            else:
                logging.info(f'Epoch {epoch} LR {lr:.6f} supervised_train_mIoU {train_miou:.2f}')
        else:
            logging.info(f'Epoch {epoch} LR {lr:.6f} '
                         f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            if not disable_validation:
                writer.add_scalar('best_val', best_val, epoch)
                writer.add_scalar('val_miou', val_miou, epoch)
                writer.add_scalar('macc_when_best', macc_when_best, epoch)
                writer.add_scalar('oa_when_best', oa_when_best, epoch)
                writer.add_scalar('val_macc', val_macc, epoch)
                writer.add_scalar('val_oa', val_oa, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val, 'seed': cfg.seed},
                            is_best=is_best
                            )
            is_best = False

    if disable_validation:
        logging.info('ODPT protocol: training finished at epoch %d; validation never ran '
                     '(validation=disabled), final checkpoint is the result.'
                     % cfg.epochs)
    else:
        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
                f'\niou per cls is: {ious_when_best}')

    if 'odpt' in cfg.dataset.common.NAME.lower():
        # ---- ODPT protocol: no Area_3 evaluation during training -----------
        # The official protocol evaluates the FINAL (last-epoch) checkpoint only,
        # and only after training. The fixed final checkpoint is written to
        # --odpt_final_ckpt (used by scripts/odpt/run_*_eval.sh).
        if cfg.get('odpt_final_ckpt', None) is not None and cfg.rank == 0:
            os.makedirs(os.path.dirname(cfg.odpt_final_ckpt), exist_ok=True)
            torch.save({
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'epoch': cfg.epochs,
            }, cfg.odpt_final_ckpt)
            logging.info('ODPT protocol: final checkpoint (epoch %d) saved to %s'
                         % (cfg.epochs, cfg.odpt_final_ckpt))
        logging.info('ODPT protocol: training finished at epoch %d; official Area_3 evaluation '
                     'is performed only after training, on the final checkpoint only.' % cfg.epochs)
    elif cfg.world_size < 2:  # do not support multi gpu testing
        # test
        load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'.csv')
        if 'sphere' in cfg.dataset.common.NAME.lower():
            test_miou, test_macc, test_oa, test_ious, test_accs = validate_sphere(model, val_loader, cfg)
        else:
            data_list = generate_data_list(cfg)
            test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)
        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
                f'\niou per cls is: {test_ious}')
        if writer is not None:
            writer.add_scalar('test_miou', test_miou, epoch)
            writer.add_scalar('test_macc', test_macc, epoch)
            writer.add_scalar('test_oa', test_oa, epoch)
        write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True)
        logging.info(f'save results in {cfg.csv_path}')
        if cfg.use_voting:
            load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
            set_random_seed(cfg.seed)
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=20,
                                                                         data_transform=data_transform)
            if writer is not None:
                writer.add_scalar('val_miou20', val_miou, cfg.epochs + 50)

            ious_table = [f'{item:.2f}' for item in val_ious]
            data = [cfg.cfg_basename, 'True', f'{val_oa:.2f}', f'{val_macc:.2f}', f'{val_miou:.2f}'] + ious_table + [
                str(best_epoch), cfg.run_dir]
            with open(cfg.csv_path, 'w', encoding='UT8') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    else:
        logging.warning('Testing using multiple GPUs is not allowed for now. Running testing after this training is required.')
    if writer is not None:
        writer.close()



def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg):
    logging.info("sup training ... "+ "-"*100)
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = build_pbar(enumerate(train_loader), cfg, desc='Sup-Train', total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1) #([B, 24000])
        """ debug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        end of debug """
        data['x'] = get_features_by_keys(data, cfg.feature_keys) #([B, 4, 24000])  data['pos']=[B,24000,3]
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            _,logits,_ = model(data) #torch.Size([2, 13, 24000])
            loss = criterion(logits, target) if 'mask' not in cfg.criterion_args.NAME.lower() \
                else criterion(logits, target, data['mask'])

        assert_finite(loss, 'sup loss (step=%d)' % num_iter,
                      epoch=epoch, iteration=idx, rank=cfg.rank)
        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            assert_grads_finite(model, 'post-backward grads', epoch=epoch, iteration=idx, rank=cfg.rank)

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            assert_params_finite(model, 'post-step params', epoch=epoch, iteration=idx, rank=cfg.rank)
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    return loss_meter.avg, miou, macc, oa, ious, accs

def un_train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg):
    logging.info("unsup training ... "+ "-"*100)
    loss_meter = AverageMeter()
    # Unsupervised branch: there are no ground-truth labels (the unlabeled pool
    # has none), so classification accuracy is not computable -> reported as N/A.
    # Instead we track pseudo-label statistics produced by the contrast criterion.
    pl_meter = AverageMeter()   # accepted pseudo-labels ratio
    cov_meter = AverageMeter()  # coverage (points with pseudo-label candidates)
    nfinite_meter = AverageMeter()  # ratio of finite per-point features
    guide_pts_meter = AverageMeter()  # accepted guide points per step
    guide_skip_meter = AverageMeter()  # steps where guide CE was skipped (K=0)
    npoints_pl = 0          # pseudo-labeled points seen this epoch
    per_class_pl = np.zeros(cfg.num_classes, dtype=np.int64)
    accepted_per_class_pl = np.zeros(cfg.num_classes, dtype=np.int64)
    conf_hist_pl = np.zeros(10, dtype=np.int64)   # confidence bins of [0,1)
    gate_acc_pl = 0         # points accepted by the configured gate
    gate_rej_pl = 0         # points rejected by the configured gate
    gate_thresholds = None
    gate_fractions = None
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = build_pbar(enumerate(train_loader), cfg, desc='Unsup-Train', total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        keys = data[0].keys() if callable(data[0].keys) else data[0].keys
        for key in keys:
            data[0][key] = data[0][key].cuda(non_blocking=True)

        num_iter += 1


        keys = data[1].keys() if callable(data[1].keys) else data[1].keys
        for key in keys:
            data[1][key] = data[1][key].cuda(non_blocking=True)

        target1 = data[0]['y'].squeeze(-1)
        target2 = data[1]['y'].squeeze(-1)


        data[0]['x'] = get_features_by_keys(data[0], cfg.feature_keys)
        data[0]['y'] =None

        data[1]['x'] = get_features_by_keys(data[1], cfg.feature_keys)
        data[1]['y'] =None

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):

            _,logits1,feats1 = model(data[0])
            _,logits2,feats2 = model(data[1])
            if hasattr(criterion, '_epoch'):
                criterion._epoch = epoch
                criterion._iteration = idx
                criterion._rank = cfg.rank
            loss = criterion(feats1,feats2, logits1, logits2,target1,target2)

        assert_finite(loss, 'unsup loss (step=%d)' % num_iter,
                      epoch=epoch, iteration=idx, rank=cfg.rank)
        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            assert_grads_finite(model, 'post-backward grads', epoch=epoch, iteration=idx, rank=cfg.rank)

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            assert_params_finite(model, 'post-step params', epoch=epoch, iteration=idx, rank=cfg.rank)
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # pseudo-label / finiteness diagnostics (per-step, cheapest first)
        if hasattr(criterion, 'pl_stats') and criterion.pl_stats is not None:
            pl_meter.update(criterion.pl_stats.get('accepted_ratio', 0.), 1)
            cov_meter.update(criterion.pl_stats.get('coverage', 0.), 1)
            guide_pts_meter.update(criterion.pl_stats.get('accepted_guide_points', 0), 1)
            guide_skip_meter.update(criterion.pl_stats.get('guide_ce_skipped', 0), 1)
            pc = criterion.pl_stats.get('pseudo_per_class')
            hi = criterion.pl_stats.get('pseudo_conf_hist')
            if criterion.pl_stats.get('pseudo_total', 0) > 0:
                npoints_pl += criterion.pl_stats['pseudo_total']
                if pc is not None and len(pc) == cfg.num_classes:
                    per_class_pl += np.asarray(pc, dtype=np.int64)
                accepted_pc = criterion.pl_stats.get(
                    'pseudo_accepted_per_class')
                if accepted_pc is not None and \
                        len(accepted_pc) == cfg.num_classes:
                    accepted_per_class_pl += np.asarray(
                        accepted_pc, dtype=np.int64)
                if hi is not None and len(hi) == 10:
                    conf_hist_pl += np.asarray(hi, dtype=np.int64)
                gate_acc_pl += criterion.pl_stats.get('pseudo_gate_accepted', 0)
                gate_rej_pl += criterion.pl_stats.get('pseudo_gate_rejected', 0)
                gate_thresholds = criterion.pl_stats.get('gate_thresholds')
                gate_fractions = criterion.pl_stats.get('gate_fractions')
            criterion.pl_stats_last = dict(criterion.pl_stats)
            criterion.pl_stats = None
        if hasattr(criterion, 'nfinite') and criterion.nfinite is not None:
            nfinite_meter.update(criterion.nfinite, 1)
            criterion.nfinite = None

        loss_meter.update(loss.item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc N/A "
                                 f"PL {pl_meter.val:.2f} Cov {cov_meter.val:.2f}")
    if not isinstance(cm.value, torch.Tensor) or cm.value.sum() == 0:
        miou, macc, oa, ious, accs = -1., -1., -1., [], []
    else:
        miou, macc, oa, ious, accs = cm.all_metrics()
    logging.info(
        f'unsup Epoch {epoch}: loss {loss_meter.avg:.4f}, '
        f'pseudo-label ratio {pl_meter.avg:.4f}, coverage {cov_meter.avg:.4f}, '
        f'finite-feature ratio {nfinite_meter.avg:.4f}, '
        f'accepted guide points {guide_pts_meter.sum:.0f} per iter, '
        f'guide CE skipped items {guide_skip_meter.sum:.0f}, Acc N/A (no GT)')
    if npoints_pl > 0:
        per_cls_ratio = per_class_pl / float(npoints_pl)
        logging.info(
            f'unsup Epoch {epoch} pseudo-label stats: points={npoints_pl} '
            f'per-class={per_class_pl.tolist()} ratio='
            f'{[round(float(x), 4) for x in per_cls_ratio]} '
            f'accepted-per-class={accepted_per_class_pl.tolist()} '
            f'conf_hist={conf_hist_pl.tolist()} '
            f'gate_thresholds={gate_thresholds} '
            f'gate_fractions={gate_fractions} '
            f'gate_accepted={gate_acc_pl} gate_rejected={gate_rej_pl}')
    else:
        logging.info(f'unsup Epoch {epoch} pseudo-label stats: no pseudo points')
    return loss_meter.avg, miou, macc, oa, ious, accs



@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=1, data_transform=None):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        data['y'] = None
        _,logits,_ = model(data)
        if 'mask' not in cfg.criterion_args.NAME or cfg.get('use_maks', False):
            cm.update(logits.argmax(dim=1), target)
        else:
            mask = data['mask'].bool()
            cm.update(logits.argmax(dim=1)[mask], target[mask])

    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs


@torch.no_grad()
def validate_sphere(model, val_loader, cfg, num_votes=1, data_transform=None):
    """
    validation for sphere sampled input points with mask.
    in this case, between different batches, there are overlapped points.
    thus, one point can be evaluated multiple times.
    In this validate_mask, we will avg the logits.
    """
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    if cfg.get('visualize', False):
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    all_logits, idx_points = [], []
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        logits = model(data)
        all_logits.append(logits)
        idx_points.append(data['input_inds'])
    all_logits = torch.cat(all_logits, dim=0).transpose(1, 2).reshape(-1, cfg.num_classes)
    idx_points = torch.cat(idx_points, dim=0).flatten()

    if cfg.distributed:
        dist.all_reduce(all_logits), dist.all_reduce(idx_points)

    # average overlapped predictions to subsampled points
    all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')


    all_logits = all_logits.argmax(dim=1)
    val_points_labels = torch.from_numpy(val_loader.dataset.clouds_points_labels[0]).squeeze(-1).to(all_logits.device)
    val_points_projections = torch.from_numpy(val_loader.dataset.projections[0]).to(all_logits.device).long()
    val_points_preds = all_logits[val_points_projections]

    del all_logits, idx_points
    torch.cuda.empty_cache()

    cm.update(val_points_preds, val_points_labels)
    miou, macc, oa, ious, accs = cm.all_metrics()

    if cfg.get('visualize', False):
        dataset_name = cfg.dataset.common.NAME.lower()
        coord = val_loader.dataset.clouds_points[0]
        colors = val_loader.dataset.clouds_points_colors[0].astype(np.float32)
        gt = val_points_labels.cpu().numpy().squeeze()
        pred = val_points_preds.cpu().numpy().squeeze()
        gt = cfg.cmap[gt, :]
        pred = cfg.cmap[pred, :]
        # output pred labels
        # save per room
        rooms = val_loader.dataset.clouds_rooms[0]

        for idx in tqdm(range(len(rooms)-1), desc='save visualization'):
            start_idx, end_idx = rooms[idx], rooms[idx+1]
            write_obj(coord[start_idx:end_idx], colors[start_idx:end_idx],
                        os.path.join(cfg.vis_dir, f'input-{dataset_name}-{idx}.obj'))
            # output ground truth labels
            write_obj(coord[start_idx:end_idx], gt[start_idx:end_idx],
                        os.path.join(cfg.vis_dir, f'gt-{dataset_name}-{idx}.obj'))
            # output pred labels
            write_obj(coord[start_idx:end_idx], pred[start_idx:end_idx],
                        os.path.join(cfg.vis_dir, f'{cfg.cfg_basename}-{dataset_name}-{idx}.obj'))
    return miou, macc, oa, ious, accs


@torch.no_grad()
def test(model, data_list, cfg, num_votes=1):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    dataset_name = cfg.dataset.common.NAME.lower()
    len_data = len(data_list)

    cfg.save_path = cfg.get('save_path', f'results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}')
    if 'semantickitti' in cfg.dataset.common.NAME.lower():
        cfg.save_path = os.path.join(cfg.save_path, str(cfg.dataset.test.test_id + 11), 'predictions')
    os.makedirs(cfg.save_path, exist_ok=True)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim
    nearest_neighbor = cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor'
    for cloud_idx, data_path in enumerate(data_list):
        logging.info(f'Test [{cloud_idx}]/[{len_data}] cloud')
        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx  = load_data(data_path, cfg)
        if label is not None:
            label = torch.from_numpy(label.astype(np.int).squeeze()).cuda(non_blocking=True)

        len_part = len(idx_points)
        nearest_neighbor = len_part == 1
        pbar = tqdm(range(len(idx_points)))
        for idx_subcloud in pbar:
            pbar.set_description(f"Test on {cloud_idx}-th cloud [{idx_subcloud}]/[{len_part}]]")
            if not (nearest_neighbor and idx_subcloud>0):
                idx_part = idx_points[idx_subcloud]
                coord_part = coord[idx_part]
                coord_part -= coord_part.min(0)

                feat_part =  feat[idx_part] if feat is not None else None
                data = {'pos': coord_part}
                if feat_part is not None:
                    data['x'] = feat_part
                if pipe_transform is not None:
                    data = pipe_transform(data)
                if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                    if 'semantickitti' in cfg.dataset.common.NAME.lower():
                        data['heights'] = torch.from_numpy((coord_part[:, gravity_dim:gravity_dim + 1] - coord_part[:, gravity_dim:gravity_dim + 1].min()).astype(np.float32)).unsqueeze(0)
                    else:
                        data['heights'] = torch.from_numpy(coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
                if not cfg.dataset.common.get('variable', False):
                    if 'x' in data.keys():
                        data['x'] = data['x'].unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                else:
                    data['o'] = torch.IntTensor([len(coord)])
                    data['batch'] = torch.LongTensor([0] * len(coord))

                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)
                data['x'] = get_features_by_keys(data, cfg.feature_keys)
                # virtual_target = torch.ones([data['x'].size()[0],data['x'].size()[2]]).long()
                # data['y'] = virtual_target.cuda(non_blocking=True)
                _,logits,_ = model(data)
                """visualization in debug mode. !!! visulization is not correct, should remove ignored idx.
                from openpoints.dataset.vis3d import vis_points, vis_multi_points
                vis_multi_points([coord, coord_part], labels=[label.cpu().numpy(), logits.argmax(dim=1).squeeze().cpu().numpy()])
                """

            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        if not cfg.dataset.common.get('variable', False):
            all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        if not nearest_neighbor:
            # average merge overlapped multi voxels logits to original point set
            idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
            all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
        else:
            # interpolate logits by nearest neighbor
            all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]
        pred = all_logits.argmax(dim=1)
        if label is not None:
            cm.update(pred, label)
        """visualization in debug mode
        from openpoints.dataset.vis3d import vis_points, vis_multi_points
        vis_multi_points([coord, coord], labels=[label.cpu().numpy(), all_logits.argmax(dim=1).squeeze().cpu().numpy()])
        """
        if cfg.visualize:
            gt = label.cpu().numpy().squeeze() if label is not None else None
            pred = pred.cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :] if gt is not None else None
            pred = cfg.cmap[pred, :]
            # output pred labels
            if 's3dis' in dataset_name:
                file_name = f'{dataset_name}-Area{cfg.dataset.common.test_area}-{cloud_idx}'
            else:
                file_name = f'{dataset_name}-{cloud_idx}'

            write_obj(coord, feat,
                      os.path.join(cfg.vis_dir, f'input-{file_name}.obj'))
            # output ground truth labels
            if gt is not None:
                write_obj(coord, gt,
                        os.path.join(cfg.vis_dir, f'gt-{file_name}.obj'))
            # output pred labels
            write_obj(coord, pred,
                      os.path.join(cfg.vis_dir, f'{cfg.cfg_basename}-{file_name}.obj'))

        if cfg.get('save_pred', False):
            if 'semantickitti' in cfg.dataset.common.NAME.lower():
                pred = pred + 1
                pred = pred.cpu().numpy().squeeze()
                pred = pred.astype(np.uint32)
                upper_half = pred >> 16  # get upper half for instances
                lower_half = pred & 0xFFFF  # get lower half for semantics (lower_half.shape) (100k+, )
                lower_half = remap_lut_write[lower_half]  # do the remapping of semantics
                pred = (upper_half << 16) + lower_half  # reconstruct full label
                pred = pred.astype(np.uint32)
                frame_id = data_path[0].split('/')[-1][:-4]
                store_path = os.path.join(cfg.save_path, frame_id + '.label')
                pred.tofile(store_path)
            elif 'scannet' in cfg.dataset.common.NAME.lower():
                pred = pred.cpu().numpy().squeeze()
                label_int_mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
                pred=np.vectorize(label_int_mapping.get)(pred)
                save_file_name=data_path.split('/')[-1].split('_')
                save_file_name=save_file_name[0]+'_'+save_file_name[1]+'.txt'
                save_file_name=os.path.join(cfg.save_path,save_file_name)
                np.savetxt(save_file_name, pred, fmt="%d")

        if label is not None:
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'[{cloud_idx}]/[{len_data}] cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                    f'\niou per cls is: {ious}')
            all_cm.value += cm.value

    if 'scannet' in cfg.dataset.common.NAME.lower():
        logging.info(f" Please select and zip all the files (DON'T INCLUDE THE FOLDER) in {cfg.save_path} and submit it to"
                     f" Scannet Benchmark https://kaldir.vc.in.tum.de/scannet_benchmark/. ")

    if label is not None:
        tp, union, count = all_cm.tp, all_cm.union, all_cm.count
        if cfg.distributed:
            dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        miou, macc, oa, ious, accs = get_mious(tp, union, count)
        return miou, macc, oa, ious, accs, all_cm
    else:
        return None, None, None, None, None, None


def main_cleanup(gpu, cfg):
    """mp.spawn entry: always tear down the process group (also on exceptions)
    so DataLoader/mp.spawn children exit cleanly and the resource_tracker
    'leaked semaphore' warning does not appear on normal exits."""
    try:
        main(gpu, cfg)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='full-state checkpoint path (model+optimizer+scheduler+GMM) to resume training from')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml
    cfg.resume_from = args.resume_from

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main_cleanup, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
