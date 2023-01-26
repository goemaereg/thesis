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

from typing import Mapping, Any, List
import cv2
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from config import get_configs
from data_loaders import get_data_loader, configure_metadata, get_image_sizes, get_bounding_boxes
from inference import CAMComputer
from evaluation import MaskEvaluator, compute_bboxes_from_scoremaps, normalize_scoremap, xai_save_cams
from util import string_contains_any, t2n  # , t2n
import wsol
# import wsol.method
import itertools
import tqdm
import mlflow
from munch import Munch
import sys
from wsol.method import AcolMethod, ADLMethod, BaseMethod, CAMMethod, CutMixMethod, HASMethod, MinMaxCAMMethod, SPGMethod

wsol_methods = {
    'acol': AcolMethod,
    'adl': ADLMethod,
    'basic': BaseMethod,
    'cam': CAMMethod,
    'cutmix': CutMixMethod,
    'has': HASMethod,
    'minmaxcam': MinMaxCAMMethod,
    'spg': SPGMethod
}

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    def __init__(self, name, higher_is_better=True):
        self.name = name
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] #if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)

    def state_dict(self):
        fetched_keys = ['value_per_epoch']
        return {key: getattr(self, key) for key in fetched_keys}

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if not isinstance(state_dict, Mapping):
            raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))
        error_msgs: List[str] = []
        if strict:
            missing_keys = []
            unexpected_keys = []
            expected_keys = ['value_per_epoch']
            for key in expected_keys:
                if key not in state_dict:
                    missing_keys.append(key)
            for key in state_dict:
                if key not in expected_keys:
                    unexpected_keys.append(key)
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        for key, val in state_dict.items():
            setattr(self, key, val)
        if len(self.value_per_epoch) > 0:
            self.current_value = self.value_per_epoch[-1]
            self.best_value = self.best_function(self.value_per_epoch)
            self.best_epoch = self.value_per_epoch.index(self.best_value)


def accelerator_get():
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    device = 'cpu'
    return device


class Trainer(object):
    _DEVICE = accelerator_get()
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pt' #'{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification']
    _BEST_CRITERION_METRIC = 'classification'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
        "SYNTHETIC": 9
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.', 'conv6.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self):
        self.epoch = 0
        self.args = get_configs()
        model_params = dict(
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold,
       )
        model_kwargs = model_params | dict(
            architecture_type=self.args.architecture_type,
            dataset = self.args.dataset_name,
            pretrained=self.args.pretrained,
            pretrained_path=self.args.pretrained_path,
            large_feature_map=self.args.large_feature_map,
            num_classes=self._NUM_CLASSES_MAPPING[self.args.dataset_name],
        )
        optimizer_params = dict(
            learning_rate=self.args.lr,
            learning_rate_features=self.args.lr,
            learning_rate_classifier=self.args.lr * self.args.lr_classifier_ratio,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True
        )
        set_random_seed(self.args.seed)
        # print(self.args)
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.model = self._set_model(**model_kwargs)
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self._DEVICE)#.cuda()
        self.mse_loss = nn.MSELoss(reduction='mean').to(self._DEVICE)#.cuda()
        batch_set_size = None
        class_set_size = None
        if self.args.wsol_method == 'minmaxcam':
            batch_set_size = self.args.minmaxcam_batch_set_size
            class_set_size = self.args.minmaxcam_class_set_size

        self.optimizer = self._set_optimizer(**optimizer_params)
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            class_set_size=class_set_size,
            batch_set_size=batch_set_size,
            train_augment=self.args.train_augment
        )
        method_args = vars(self.args) | {'model': self.model, 'device': self._DEVICE, 'optimizer': self.optimizer}
        self.wsol_method = wsol_methods[self.args.wsol_method](**method_args)
        self.cam_method = self.args.cam_method
        # MLFlow logging
        # artifacts
        mlflow.log_artifact('requirements.txt')
        mlflow.log_artifact(self.args.config, 'config')
        state = vars(self.args)
        state['data_paths'] = vars(self.args.data_paths)
        del state['reporter']
        if self.args.pretrained_path is None:
            del state['pretrained_path']
        mlflow.log_dict(state, 'state/config.json')
        info = dict(
            loss_fn_class=self.cross_entropy_loss.__class__.__name__,
            loss_fn_minmax=self.mse_loss.__class__.__name__
        )
        mlflow.log_text(' '.join(sys.argv), 'state/command.txt')
        mlflow.log_dict(info, 'state/info.json')
        # hyper parameters
        params = model_params | optimizer_params
        params |= dict(
            epochs=self.args.epochs,
            batch_size=self.args.batch_size
        )
        mlflow.log_params(params)
        # tags
        tags = dict(
            architecture=self.args.architecture,
            architecture_type=self.args.architecture_type,
            wsol_method=self.args.wsol_method,
            cam_method=self.args.cam_method,
            experiment=self.args.experiment_name,
            dataset=self.args.dataset_name,
            dataset_spec=self.args.dataset_name_suffix,
            model=self.model.__class__.__name__,
            optimizer=self.optimizer.__class__.__name__,
            pretrained=self.args.pretrained,
            large_feature_map=self.args.large_feature_map,
            num_classes=self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        )
        mlflow.set_tags(tags)

    def _set_performance_meters(self):
        if self.args.dataset_name in ('SYNTHETIC', 'OpenImages'):
            metric = 'PxAP'
            self._EVAL_METRICS += [metric]
        if self.args.dataset_name in ('SYNTHETIC', 'ILSVRC', 'CUB'):
            metric = self.args.bbox_metric
            self._EVAL_METRICS += [metric]
            self._EVAL_METRICS += [f'{metric}_IOU_{threshold}'
                                   for threshold in self.args.iou_threshold_list]
        self._BEST_CRITERION_METRIC = self._EVAL_METRICS[2]

        eval_dict = {
            split: {
                metric: PerformanceMeter(name=f'{split}_{metric}',
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self, **kwargs):
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](**kwargs)
        model = model.to(self._DEVICE) # model.cuda()
        # print(model)
        return model

    def _set_optimizer(self, **kwargs):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_features.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_classifiers.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_features.append(parameter)
        opt = Munch(kwargs)
        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': opt.learning_rate_features},
            {'params': param_classifiers, 'lr': opt.learning_rate_classifier}],
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        return optimizer

    # def _wsol_training(self, images, targets):
    #     if (self.args.wsol_method == 'cutmix' and
    #             self.args.cutmix_prob > np.random.rand(1) and
    #             self.args.cutmix_beta > 0):
    #         images, target_a, target_b, lam = wsol.method.cutmix(
    #             images, targets, self.args.cutmix_beta)
    #         output_dict = self.model(images)
    #         logits = output_dict['logits']
    #         loss = (self.cross_entropy_loss(logits, target_a) * lam +
    #                 self.cross_entropy_loss(logits, target_b) * (1. - lam))
    #         return logits, loss
    #
    #     if self.args.wsol_method == 'has':
    #         images = wsol.method.has(images, self.args.has_grid_size,
    #                                  self.args.has_drop_rate)
    #
    #     output_dict = self.model(images, targets)
    #     logits = output_dict['logits']
    #
    #     if self.args.wsol_method in ('acol', 'spg'):
    #         loss = wsol.method.__dict__[self.args.wsol_method].get_loss(
    #             output_dict, targets, spg_thresholds=self.args.spg_thresholds)
    #     else:
    #         loss = self.cross_entropy_loss(logits, targets)
    #
    #     return logits, loss

    # def vgg16_minmaxcam_regularization_loss(self, images, labels):
    #     for param in self.model.features.parameters():
    #         param.requires_grad = False
    #     for param in self.model.conv6.parameters():
    #         param.requires_grad = False
    #     for param in self.model.fc.parameters():
    #         param.requires_grad = True
    #
    #     images = images.to(self._DEVICE)  # .cuda()
    #     # Compute CAMs from B(I) with I=input image
    #     result_orig = self.model(images, labels,
    #                              return_cam=True, clone_cam=False)
    #     cams = result_orig['cams']
    #     # cams = t2n(cams)
    #     cams = cams.unsqueeze(1)
    #     # rescale cams to images input size
    #     resize = tuple(images.shape[2:])
    #     cams_resized = F.interpolate(cams, size=resize, mode='bilinear')
    #     # normalize cams
    #     cams_min = torch.amin(cams_resized, dim=(2, 3), keepdim=True)
    #     cams_max = torch.amax(cams_resized, dim=(2, 3), keepdim=True)
    #     cams_normalized = torch.zeros_like(cams_resized)
    #     if ((not torch.isnan(cams_resized).any()) and
    #         (not torch.equal(cams_min, cams_max))):
    #         cams_normalized = (cams_resized - cams_min)/(cams_max - cams_min)
    #
    #     # Compute B(I * cams_normalized)
    #     result_mask = self.model(images * cams_normalized)
    #     # x = self.model.features(images * cams_normalized)
    #     # x = self.model.conv6(x)
    #     # out_extra_masked = self.model.relu(x)
    #
    #     # compute features_i
    #     features_i = result_mask['avgpool_flat']
    #     # compute features_o
    #     features_o = result_orig['avgpool_flat']
    #
    #     # compute losses
    #     loss_crr = 0.0
    #     loss_frr = 0.0
    #     # compute ss and bs per target
    #     # ss and bs can be less than minmaxcam_class_set_size and minmaxcam_batch_set_size for small datasets
    #     # ss = self.args.minmaxcam_class_set_size
    #     # bs = self.args.minmaxcam_batch_set_size
    #     bs = np.unique(labels).shape[0]
    #     starts = np.nonzero(np.r_[1, np.diff(labels)])[0]
    #     stops = np.nonzero(np.r_[0, np.diff(labels), 1])[0]
    #     for start, stop in zip(starts, stops):
    #         loss_crr_ss = 0.0
    #         f_i_pair_num = 0
    #         features_i_ss = features_i[start:stop]
    #         features_o_ss = features_o[start:stop]
    #         ss = stop - start
    #         for j, k in itertools.combinations(range(ss), 2):
    #             f_i_pair_num += 1
    #             feature_i_ss_j = features_i_ss[j].unsqueeze(0)
    #             feature_i_ss_k = features_i_ss[k].unsqueeze(0)
    #             loss_crr_ss += self.mse_loss(feature_i_ss_j, feature_i_ss_k)
    #         loss_frr += self.mse_loss(features_i_ss, features_o_ss)
    #         loss_crr += loss_crr_ss / f_i_pair_num
    #     loss_crr /= bs
    #     loss_frr /= bs
    #     return loss_crr, loss_frr

    # def _train(self, epoch, split):
    #     loader = self.loaders[split]
    #
    #     total_loss = 0.0
    #     num_correct = 0
    #     num_images = 0
    #
    #     for images, targets, _ in loader:
    #         images = images.to(self._DEVICE) # images.cuda()
    #         targets = targets.to(self._DEVICE) #.cuda()
    #
    #         # minmaxcam stage I
    #         if self.args.wsol_method == 'minmaxcam':
    #             for param in self.model.features.parameters():
    #                 param.requires_grad = True
    #             for param in self.model.conv6.parameters():
    #                 param.requires_grad = True
    #             for param in self.model.fc.parameters():
    #                 param.requires_grad = True
    #
    #         self.model.train()
    #         logits, loss = self._wsol_training(images, targets)
    #         pred = logits.argmax(dim=1)
    #
    #         total_loss += loss.item() * images.size(0)
    #         num_correct += (pred == targets).sum().item()
    #         num_images += images.size(0)
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         # minmaxcam stage II
    #         if self.args.wsol_method == 'minmaxcam':
    #             for param in self.model.features.parameters():
    #                 param.requires_grad = False
    #             for param in self.model.conv6.parameters():
    #                 param.requires_grad = False
    #             for param in self.model.fc.parameters():
    #                 param.requires_grad = True
    #
    #             self.optimizer.zero_grad()
    #             self.model.eval()
    #
    #             loss_crr, loss_frr = \
    #                 self.vgg16_minmaxcam_regularization_loss(images, targets)
    #
    #             self.model.train()
    #             loss_all_p2 = self.args.minmaxcam_crr_weight * loss_crr + \
    #                           self.args.minmaxcam_frr_weight * loss_frr
    #             try:
    #                 loss_all_p2.backward()
    #             except RuntimeError as e:
    #                 print(e)
    #                 print(loss_all_p2)
    #             self.optimizer.step()
    #
    #     loss_average = total_loss / float(num_images)
    #     # if loss_average > 1000:
    #     #     print(loss_average)
    #     classification_acc = num_correct / float(num_images) # * 100
    #
    #     self.performance_meters[split]['classification'].update(
    #         classification_acc)
    #     self.performance_meters[split]['loss'].update(loss_average)
    #
    #     mlflow_metrics = {f'{split}_loss': loss_average, f'{split}_accuracy': classification_acc}
    #     mlflow.log_metrics(mlflow_metrics, step=epoch)
    #
    #     return dict(classification_acc=classification_acc,
    #                 loss=loss_average)

    def train(self, epoch, split):
        self.model.train()
        loader = self.loaders[split]
        total_loss = 0.0
        num_correct = 0
        num_images = 0
        for images, targets, _ in loader:
            images = images.to(self._DEVICE)
            targets = targets.to(self._DEVICE)
            logits, loss = self.wsol_method.train(images, targets)
            pred = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)
        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) # * 100
        self.performance_meters[split]['classification'].update(classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)
        mlflow_metrics = {f'{split}_loss': loss_average, f'{split}_accuracy': classification_acc}
        mlflow.log_metrics(mlflow_metrics, step=epoch)
        return dict(classification_acc=classification_acc, loss=loss_average)


    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(
                        split, metric, current_performance))
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_loss(self, loader):
        total_loss = 0.0
        num_images = 0
        with torch.no_grad():
            for images, targets, image_ids in loader:
                images = images.to(self._DEVICE) #.cuda()
                targets = targets.to(self._DEVICE) #.cuda()
                output_dict = self.model(images)
                logits = output_dict['logits']
                loss = self.cross_entropy_loss(logits, targets)
                total_loss += loss.item() * images.size(0)
                num_images += images.size(0)
        loss_average = total_loss / float(num_images)
        return loss_average

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0
        with torch.no_grad():
            for images, targets, image_ids in loader:
                images = images.to(self._DEVICE) #.cuda()
                targets = targets.to(self._DEVICE) #.cuda()
                output_dict = self.model(images)
                pred = output_dict['logits'].argmax(dim=1)
                num_correct += (pred == targets).sum().item()
                num_images += images.size(0)
            classification_acc = num_correct / float(num_images) # * 100
        return classification_acc

    def evaluate(self, epoch, split, save_xai=False, save_cams=False, log=False):
        # print("Evaluate epoch {}, split {}".format(epoch, split))
        self.model.eval()
        loss = self._compute_loss(loader=self.loaders[split])
        self.performance_meters[split]['loss'].update(loss)
        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)
        metadata_root = os.path.join(self.args.metadata_root, split)
        cam_computer = CAMComputer(
            model=self.model,
            loader=self.loaders[split],
            metadata_root=metadata_root,
            mask_root=self.args.mask_root,
            scoremap_root=self.args.scoremap_root,
            cam_method=self.cam_method,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            multi_gt_eval=self.args.multi_gt_eval,
            device = self._DEVICE,
            bbox_metric=self.args.bbox_metric,
            log=log
        )
        metrics = cam_computer.compute_and_evaluate_cams(save_cams=save_cams)
        for metric, value in metrics.items():
            self.performance_meters[split][metric].update(value)
        if self.args.xai and save_xai:
            metadata = configure_metadata(metadata_root)
            xai_save_cams(xai_root=self.args.xai_root,
                          metadata=metadata,
                          data_root=self.args.data_paths[split],
                          scoremap_root=self.args.scoremap_root,
                          evaluator=cam_computer.evaluator,
                          multi_contour_eval=self.args.multi_contour_eval,
                          log=True)

        mlflow_metrics = { f'{split}_loss': loss, f'{split}_accuracy': accuracy}
        mlflow_metrics |= { f'{split}_{metric}':value  for metric, value in metrics.items() }
        mlflow.log_metrics(mlflow_metrics, step=epoch)


    def _torch_save_model(self, filename):
        meters_state_dict = {
            split: {
                metric: self.performance_meters[split][metric].state_dict()
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        torch.save({'epoch': self.epoch,
                    'meters_state_dict': meters_state_dict,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint_best_criterion(self, epoch, split):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('best'))

    def save_checkpoint_current_epoch(self, epoch):
        self._torch_save_model(
            self._CHECKPOINT_NAME_TEMPLATE.format('last'))

    def save_checkpoint(self, epoch, split):
        self.save_checkpoint_best_criterion(epoch, split)
        self.save_checkpoint_current_epoch(epoch)

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.epoch = checkpoint['epoch']
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for split, metrics in checkpoint['meters_state_dict'].items():
                    for metric, meters in metrics.items():
                        self.performance_meters[split][metric].load_state_dict(meters)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        # else:
        #     raise IOError("No checkpoint {}.".format(checkpoint_path))

    def report_train(self, train_performance, epoch, split='train'):
        reporter_log_root = os.path.join(self.args.reporter_log_root, split)
        if not os.path.isdir(reporter_log_root):
            os.makedirs(reporter_log_root)
        reporter_instance = self.reporter(reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/classification'.format(split=split),
            val=train_performance['classification_acc'])
        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=train_performance['loss'])
        reporter_instance.write()

    def report(self, epoch, split):
        reporter_log_root = os.path.join(self.args.reporter_log_root, split)
        if not os.path.isdir(reporter_log_root):
            os.makedirs(reporter_log_root)
        reporter_instance = self.reporter(reporter_log_root, epoch)
        for metric in self._EVAL_METRICS:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(
                key='{split}/{metric}_best'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write()

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.lr_decay_frequency == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1


def main():
    trainer = Trainer()
    print("===========================================================")
    print(f"Accelerator: {accelerator_get()}")
    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    if trainer.args.train:
        print("===========================================================")
        print("Start training {} epochs ...".format(trainer.args.epochs))
        epochs_range = range(trainer.epoch, trainer.args.epochs, 1)
        tq0 = tqdm.tqdm(epochs_range, total=len(epochs_range), desc='training epochs')
        for epoch in tq0:
            trainer.adjust_learning_rate(epoch)
            train_performance = trainer.train(epoch, split='train')
            trainer.report_train(train_performance, epoch, split='train')
            last_epoch = (epoch == (trainer.args.epochs - 1))
            trainer.evaluate(epoch, split='val', save_xai=last_epoch, save_cams=last_epoch, log=last_epoch)
            # trainer.print_performances()
            trainer.report(epoch, split='val')
            trainer.epoch += 1
            trainer.save_checkpoint(epoch, split='val')
    else:
        print("===========================================================")
        print("Final epoch evaluation on val set ...")
        trainer.evaluate(trainer.args.epochs, split='val', save_xai=True, save_cams=True, log=True)
    print("===========================================================")
    print("Final epoch evaluation on test set ...")
    trainer.evaluate(trainer.args.epochs, split='test', save_xai=True, save_cams=True, log=True)
    trainer.print_performances()
    trainer.report(trainer.args.epochs, split='test')
    trainer.save_performances()
    # MLFlow logging
    mlflow.pytorch.log_model(trainer.model, 'model', pip_requirements='requirements.txt')


if __name__ == '__main__':
    with mlflow.start_run():
        main()


