"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import munch
import os
import shutil
import warnings
import json
from util import Logger, Reporter
from typing import Dict


_DEVICE_NAMES = ('cpu', 'cuda')
_DEVICE_DEFAULT = 'cpu'
_DATASET_NAMES = ('CUB', 'ILSVRC', 'OpenImages', 'SYNTHETIC')
_ARCHITECTURE_NAMES = ('vgg16', 'resnet50', 'inception_v3')
_ARCHITECTURE_DEFAULT = 'vgg16'
_ARCHITECTURE_TYPE_NAMES = ('auto', 'vanilla')
_ARCHITECTURE_TYPE_DEFAULT = 'auto'
_METHOD_NAMES = ('basic', 'cam', 'adl', 'acol', 'spg', 'has', 'cutmix', 'minmaxcam')
_SPLITS = ('train', 'val', 'test')
_BBOX_METRIC_NAMES = ('MaxBoxAcc', 'MaxBoxAccV2', 'MaxBoxAccV3')
_BBOX_METRIC_DEFAULT = 'MaxBoxAccV2'
_CAM_METHOD_NAMES = ('cam', 'gradcam', 'scorecam')
_CAM_METHOD_DEFAULT = 'cam'
_LR_SCHEDULER_NAMES = ('StepLR', "MultiStepLR")
_LR_SCHEDULER_DEFAULT = 'StepLR'

def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_bbox_metric(args):
    if args.bbox_metric == 'MaxBoxAcc':
        args.multi_contour_eval = False
        args.multi_iou_eval = False
        args.multi_gt_eval = False
        warnings.warn("Bbox metric MaxBoxAcc is deprecated. Use MaxBoxAccV2 or MaxBoxAccV3")
    elif args.bbox_metric == 'MaxBoxAccV2':
        args.multi_contour_eval = True
        args.multi_iou_eval = True
        args.multi_gt_eval = False
    elif args.bbox_metric == 'MaxBoxAccV3':
        args.multi_contour_eval = True
        args.multi_iou_eval = True
        args.multi_gt_eval = True
    else:
        raise ValueError

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_architecture_type(architecture_type, wsol_method):
    if architecture_type == 'auto':
        if wsol_method in ('cam', 'has', 'cutmix', 'minmaxcam'):
            architecture_type = 'cam'
        elif wsol_method in ('acol', 'adl', 'spg'):
            architecture_type = wsol_method
        else:
            architecture_type = 'vanilla'
    elif architecture_type == 'vanilla' and wsol_method not in ('basic'):
        raise ValueError
    return architecture_type


def configure_data_paths(args, tags=None):
    if tags is None:
        tags = list()
    train = val = test = os.path.join(args.data_root, args.dataset_name, *tags)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def configure_mask_root(args, tags=None):
    if tags is None:
        tags = list()
    mask_root = os.path.join(args.mask_root, args.dataset_name, *tags) #'OpenImages')
    return mask_root


def configure_log_folder(args):
    log_folder = os.path.join(args.log_folder, args.experiment_name)
    if os.path.isdir(log_folder):
        if args.override_cache:
            shutil.rmtree(log_folder, ignore_errors=True)
        # else:
        #     raise RuntimeError("Experiment with the same name exists: {}"
        #                        .format(log_folder))
    os.makedirs(log_folder, exist_ok=True)
    return log_folder


def configure_log(args):
    log_file_name = os.path.join(args.log_folder, 'log.log')
    Logger(log_file_name)
    return log_file_name


def configure_reporter(args):
    reporter = Reporter # importlib.import_module('util').Reporter
    reporter_log_root = os.path.join(args.log_folder, 'reports')
    if not os.path.isdir(reporter_log_root):
        os.makedirs(reporter_log_root)
    return reporter, reporter_log_root


def configure_pretrained_path(args):
    pretrained_path = args.pretrained_path
    return pretrained_path


def check_dependency(args):
    if args.dataset_name == 'CUB':
        if args.num_val_sample_per_class >= 6:
            raise ValueError("num-val-sample must be <= 5 for CUB.")
    if args.dataset_name == 'OpenImages':
        if args.num_val_sample_per_class >= 26:
            raise ValueError("num-val-sample must be <= 25 for OpenImages.")

def configure_parse(load_config=True):
    parser = argparse.ArgumentParser()

    # Config
    parser.add_argument('--config', type=str, help='Configuration JSON file path with saved arguments')

    # Util
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_name', type=str, default='test_case')
    parser.add_argument('--log_folder', type=str, default='train_log', help='log folder')
    parser.add_argument('--scoremap_folder', type=str,
                        default='scoremaps',
                        help="The root folder for score maps to be evaluated.")
    parser.add_argument('--xai_folder', type=str, default='xai', help='xai folder')
    parser.add_argument('--override_cache', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--device', type=str, default=_DEVICE_DEFAULT, choices=_DEVICE_NAMES)

    # Data
    parser.add_argument('--dataset_name', type=str, default='SYNTHETIC',
                        choices=_DATASET_NAMES)
    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='data/dataset',
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default='data/metadata')
    parser.add_argument('--mask_root', metavar='/PATH/TO/MASKS',
                        default='data/maskdata',
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=0,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')

    # Setting
    parser.add_argument('--architecture', default=_ARCHITECTURE_DEFAULT,
                        choices=_ARCHITECTURE_NAMES,
                        help='model architecture: ' +
                             ' | '.join(_ARCHITECTURE_NAMES) +
                             f' (default: {_ARCHITECTURE_DEFAULT})')
    parser.add_argument('--architecture_type', default=_ARCHITECTURE_TYPE_DEFAULT,
                        choices=_ARCHITECTURE_TYPE_NAMES, help='model architecture type')
    parser.add_argument('--cam_method', default=_CAM_METHOD_DEFAULT,
                        choices=_CAM_METHOD_NAMES,
                        help='CAM method used to generate scoremaps')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Use pre-trained model.')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path of pre-trained model.')
    parser.add_argument('--cam_curve_interval', type=float, default=0.01,#.001,
                        help='CAM curve interval')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='input crop size')
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])
    parser.add_argument('--eval_checkpoint_type', type=str, default='last',
                        choices=('best', 'last'))
    parser.add_argument('--bbox_metric', type=str, default=_BBOX_METRIC_DEFAULT,
                        choices=_BBOX_METRIC_NAMES)
    # Common hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini-batch size (default: 64), this is the total'
                             'batch size of all GPUs on the current node when'
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', default=0.001, type=float, #default=0.01, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_scheduler', default=_LR_SCHEDULER_DEFAULT,
                        choices=_LR_SCHEDULER_NAMES, help='Learning rate scheduler')
    parser.add_argument('--lr_decay_frequency', type=int, default=30,
                        help='How frequently do we decay the learning rate?')
    parser.add_argument('--lr_scheduler_multisteplr_milestones', type=int, action='extend', default=list())
    parser.add_argument('--lr_classifier_ratio', type=float, default=10,
                        help='Multiplicative factor on the classifier layer.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)

    # Method-specific hyperparameters
    parser.add_argument('--wsol_method', type=str, default='cam',
                        choices=_METHOD_NAMES)
    parser.add_argument('--has_grid_size', type=int, default=4)
    parser.add_argument('--has_drop_rate', type=float, default=0.5)
    parser.add_argument('--acol_threshold', type=float, default=0.7)
    parser.add_argument('--spg_threshold_1h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_1l', type=float, default=0.01,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2h', type=float, default=0.5,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2l', type=float, default=0.05,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3h', type=float, default=0.7,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3l', type=float, default=0.1,
                        help='SPG threshold')
    parser.add_argument('--adl_drop_rate', type=float, default=0.75,
                        help='ADL dropout rate')
    parser.add_argument('--adl_threshold', type=float, default=0.9,
                        help='ADL gamma, threshold ratio '
                             'to maximum value of attention map')
    parser.add_argument('--cutmix_beta', type=float, default=1.0,
                        help='CutMix beta')
    parser.add_argument('--cutmix_prob', type=float, default=1.0,
                        help='CutMix Mixing Probability'),
    # see minmaxcam paper
    parser.add_argument('--minmaxcam_class_set_size', type=int, default=5,
                        help='MinMaxCam class set size'),
    # see minmaxcam paper
    parser.add_argument('--minmaxcam_batch_set_size', type=int, default=12,
                        help='MinMaxCam batch set size'),
    # see minmaxcam paper
    parser.add_argument('--minmaxcam_frr_weight', type=int, default=10,
                        help='MinMaxCam Full Region Regularization Weight'),
    # see minmaxcam paper
    parser.add_argument('--minmaxcam_crr_weight', type=int, default=1,
                        help='MinMaxCam Common Region Regularization Weight'),
    # tags
    parser.add_argument('--dataset_name_suffix', type=str, default='',
                        help='Suffix = <tag1><tag2><tag3> used to partition SYNTHETIC dataset. '
                             'tag1 = <choice o (overlapping) | d (disjunct)'
                             'tag2 = <n_instances: 0..4>'
                             'tag3 = <choice b (background) | t (transparent'),
    parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--train_augment', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--wsol', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--xai', type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    if load_config and args.config is not None:
        # Load stored arguments
        args_config = configure_load(args.config)
        # Update with provided arguments. Provided arguments take priority over shared arguments.
        args = parser.parse_args(namespace=argparse.Namespace(**args_config))
    return args

def configure_save(filename: str, args: Dict):
    with open(filename, "w") as fp:
        json.dump(args, fp, sort_keys=True, indent=2)

def configure_load(filename: str) -> Dict:
    with open(filename, "r") as fp:
        return json.load(fp)

def get_configs():
    args = configure_parse()
    check_dependency(args)

    tags_encoded = []
    if args.dataset_name_suffix:
        tags_encoded.append('_'.join(list(args.dataset_name_suffix)))
    args.log_folder = configure_log_folder(args)
    args.scoremap_root = os.path.join(args.log_folder, args.scoremap_folder)
    args.xai_root = os.path.join(args.log_folder, args.xai_folder)
    args.log_path = configure_log(args)
    configure_bbox_metric(args)

    args.architecture_type = get_architecture_type(args.architecture_type, args.wsol_method)
    args.data_paths = configure_data_paths(args, tags=tags_encoded)
    args.metadata_root = os.path.join(args.metadata_root, args.dataset_name, *tags_encoded)
    args.mask_root = configure_mask_root(args, tags=tags_encoded)
    args.reporter, args.reporter_log_root = configure_reporter(args)
    args.pretrained_path = configure_pretrained_path(args)
    args.spg_thresholds = ((args.spg_threshold_1h, args.spg_threshold_1l),
                           (args.spg_threshold_2h, args.spg_threshold_2l),
                           (args.spg_threshold_3h, args.spg_threshold_3l))

    return args
