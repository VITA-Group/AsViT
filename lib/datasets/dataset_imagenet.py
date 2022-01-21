import yaml
import os
from pathlib import Path
this_dir = Path(__file__).parent.absolute()

from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.utils import *
import ml_collections


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/transforms_factory.py
def get_imagenet_dataset(data_path=None, config_file="%s/default.yaml"%this_dir, no_aug=False, img_size=None, batch_size=None, workers=None):
    with open(config_file, 'r') as f:
        args = ml_collections.ConfigDict(yaml.safe_load(f))
    data_config = resolve_data_config(vars(args), verbose=False)
    if data_path is not None:
        args.data = data_path
    data_config['no_aug'] = no_aug
    if img_size is not None:
        data_config['input_size'] = img_size
    if batch_size is not None:
        data_config['batch_size'] = batch_size
        args.batch_size = batch_size

    if workers is None:
        workers = args.workers
    else:
        assert isinstance(workers, int) and workers >= 0

    args.prefetcher = not args.no_prefetcher

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # Dataset & Dataloader
    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir):
        _logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train, input_size=data_config['input_size'], batch_size=args.batch_size, is_training=True, use_prefetcher=args.prefetcher, no_aug=args.no_aug, re_prob=args.reprob,
        re_mode=args.remode, re_count=args.recount, re_split=args.resplit, scale=args.scale, ratio=args.ratio, hflip=args.hflip, vflip=args.vflip, color_jitter=args.color_jitter,
        auto_augment=args.aa, num_aug_splits=num_aug_splits, interpolation=train_interpolation, mean=data_config['mean'], std=data_config['std'], num_workers=workers,
        distributed=args.distributed, collate_fn=collate_fn, pin_memory=args.pin_mem, use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            _logger.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = Dataset(eval_dir)

    loader_eval = create_loader(
        dataset_eval, input_size=data_config['input_size'], batch_size=args.validation_batch_size_multiplier * args.batch_size, is_training=False, use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'], mean=data_config['mean'], std=data_config['std'], num_workers=workers, distributed=args.distributed,
        crop_pct=data_config['crop_pct'], pin_memory=args.pin_mem,
    )

    return dataset_train, loader_train, dataset_eval, loader_eval
