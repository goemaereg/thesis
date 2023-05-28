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
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim
from config import get_configs
from data_loaders import get_data_loader, configure_metadata
from inference import CAMComputer
from util import string_contains_any, Logger
import wsol
import tqdm
import mlflow
from munch import Munch
import sys
from wsol.method import AcolMethod, ADLMethod, BaseMethod, CutMixMethod, HASMethod, MinMaxCAMMethod, SPGMethod

wsol_methods = {
    # 'acol': AcolMethod,
    # 'adl': ADLMethod,
    'cam': BaseMethod,
    'minmaxcam': MinMaxCAMMethod,
    'gradcam': BaseMethod,
    'gradcam++': BaseMethod,
    'scorecam': BaseMethod,
    # 'cutmix': CutMixMethod,
    # 'has': HASMethod,
    # 'spg': SPGMethod
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


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, min_epochs=5):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.epochs = 0
        self.best_loss = None
        self.early_stop = False

    @property
    def stop(self):
        return self.early_stop and self.epochs >= self.min_epochs

    def __call__(self, val_loss):
        self.epochs += 1
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pt' #'{}_checkpoint.pth.tar'
    _EVAL_METRICS = ['loss', 'classification']
    _BEST_CRITERION_METRIC = 'classification'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
        "SYNTHETIC": 9
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],#, 'conv6.'], original code is without conv6 because that layer needs same lr as classifier
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self, args):
        self.epoch = 0
        self.args = args
        self.splits = self._set_splits(args)
        model_params = dict(
            architecture=self.args.architecture,
            architecture_type=self.args.architecture_type,
            dataset_name = self.args.dataset_name,
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
        self.performance_meters = self._set_performance_meters()
        self.early_stopping = self._set_early_stopping()
        self.reporter = self.args.reporter
        self.device = self._set_device()
        self.model = self._set_model(**model_params)
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)#.cuda()
        self.mse_loss = nn.MSELoss(reduction='mean').to(self.device)#.cuda()
        batch_set_size = None
        class_set_size = None
        if self.args.wsol_method == 'minmaxcam':
            batch_set_size = self.args.minmaxcam_batch_set_size
            class_set_size = self.args.minmaxcam_class_set_size

        self.optimizer = self._set_optimizer(**optimizer_params)
        self.lr_scheduler = None
        self.loaders = get_data_loader(
            splits=self.splits,
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
            train_augment=self.args.train_augment,
            dataset_name=self.args.dataset_name
        )
        method_args = vars(self.args) | {'model': self.model, 'device': self.device, 'optimizer': self.optimizer}
        self.wsol_method = wsol_methods[self.args.wsol_method](**method_args)

        # MLFlow logging
        # artifacts
        mlflow.log_artifact('requirements.txt')
        mlflow.log_artifact(self.args.config, 'config')
        state: dict[str, Any] = vars(self.args)
        state['data_paths'] = vars(self.args.data_paths)
        del state['reporter']
        if self.args.pretrained_path is None:
            del state['pretrained_path']
        # sort by key
        state = dict(sorted(state.items()))
        mlflow.log_dict(state, 'state/config.json')
        mlflow.log_text(' '.join(sys.argv), 'state/command.txt')
        # hyper parameters
        params = model_params | optimizer_params
        params |= dict(
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            iter_max = self.args.iter_max,
            iter_stop_prob_delta=self.args.iter_stop_prob_delta,
            bbox_mask_strategy = self.args.bbox_mask_strategy,
            bbox_merge_strategy = self.args.bbox_merge_strategy,
            bbox_merge_iou_threshold=self.args.bbox_merge_iou_threshold
        )
        if self.args.wsol_method == 'minmaxcam':
            params |= dict(
                minmaxcam_bss=self.args.minmaxcam_batch_set_size,
                minmaxcam_css = self.args.minmaxcam_class_set_size,
                minmaxcam_frr = self.args.minmaxcam_frr_weight,
                minmaxcam_crr = self.args.minmaxcam_crr_weight
            )
        mlflow.log_params(params)
        # tags
        tags = dict(
            architecture=self.args.architecture,
            architecture_type=self.args.architecture_type,
            dataset=self.args.dataset_name,
            dataset_spec=self.args.dataset_name_suffix,
            experiment=self.args.experiment_name,
            label=self.args.label,
            large_feature_map=self.args.large_feature_map,
            method=self.args.wsol_method,
            model=self.model.__class__.__name__,
            num_classes=self._NUM_CLASSES_MAPPING[self.args.dataset_name],
            optimizer=self.optimizer.__class__.__name__,
            pretrained=self.args.pretrained,
            train=self.args.train,
            train_augment=self.args.train_augment
        )
        mlflow.set_tags(tags)

    def _set_device(self):
        device = self.args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print('Device cuda is unavailable. Switching to cpu.')
            device = 'cpu'
        # elif device == 'mps' and not torch.backends.mps.is_available():
        #     print('Device mps is unavailable. Switching to cpu.')
        #     device = 'cpu'
        return device

    def _set_splits(self, args):
        splits = []
        if args.train:
            splits.append('train')
        splits.append('val')
        if args.dataset_name != 'ILSVRC':
            splits.append('test')
        return tuple(splits)

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
            for split in self.splits
        }
        return eval_dict

    def _set_early_stopping(self):
        early_stopping = None
        if self.args.early_stopping:
            min_delta = self.args.early_stopping_min_delta
            patience = self.args.early_stopping_patience
            min_epochs = self.args.early_stopping_minimum_epochs
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, min_epochs=min_epochs)
        return early_stopping

    def _set_model(self, **kwargs):
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](**kwargs)
        model = model.to(self.device) # model.cuda()
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

    def set_lr_scheduler(self, optimizer, last_epoch=-1):
        scheduler = None
        if self.args.lr_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                self.args.lr_scheduler_steplr_stepsize,
                gamma=0.1, last_epoch=last_epoch)
        elif self.args.lr_scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                self.args.lr_scheduler_multisteplr_milestones,
                gamma=0.1, last_epoch=last_epoch)
        self.lr_scheduler = scheduler
        return scheduler

    def train(self, epoch, split):
        self.model.train()
        loader = self.loaders[split]
        total_loss = 0.0
        num_correct = 0
        num_images = 0
        for images, targets, _ in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
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
        for split in self.splits:
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

    def _compute_loss_accuracy(self, loader):
        total_loss = 0.0
        num_correct = 0
        num_images = 0
        with torch.no_grad():
            for images, targets, image_ids in loader:
                images = images.to(self.device) #.cuda()
                targets = targets.to(self.device) #.cuda()
                output_dict = self.model(images)
                logits = output_dict['logits']
                preds = logits.argmax(dim=1)
                loss = self.cross_entropy_loss(logits, targets)
                total_loss += loss.item() * images.size(0)
                num_correct += (preds == targets).sum().item()
                num_images += images.size(0)
        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images)  # * 100
        return loss_average, classification_acc

    def evaluate_classification(self, epoch, split):
        # print("Evaluate epoch {}, split {}".format(epoch, split))
        self.model.eval()
        loss, accuracy = self._compute_loss_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['loss'].update(loss)
        self.performance_meters[split]['classification'].update(accuracy)
        eval_metrics = { 'loss': loss, 'accuracy': accuracy}
        mlflow_metrics = {f'{split}_{metric}':value for metric, value in eval_metrics.items()}
        mlflow.log_metrics(mlflow_metrics, step=epoch)
        return eval_metrics

    def evaluate_wsol(self, epoch, split, save_xai=False, save_cams=False, log=False):
        # print("Evaluate epoch {}, split {}".format(epoch, split))
        if self.args.wsol == False:
            return {}
        self.model.eval()
        cam_computer = CAMComputer(
            model=self.model,
            device=self.device,
            split=split,
            config=self.args,
            loader=self.loaders[split],
            log=log)
        # this method takes care of mlflow metrics logging
        metrics = cam_computer.compute_and_evaluate_cams(epoch=epoch, save_xai=save_xai)
        for metric, value in metrics.items():
            if metric in self.performance_meters[split]:
               self.performance_meters[split][metric].update(value)
        return metrics

    def evaluate(self, epoch, split, save_xai=False, save_cams=False, log=False):
        metrics_class = self.evaluate_classification(epoch, split)
        metrics_wsol = self.evaluate_wsol(epoch, split, save_xai, save_cams=save_cams, log=log)
        return metrics_class | metrics_wsol

    def _torch_save_model(self, filename):
        meters_state_dict = {
            split: {
                metric: self.performance_meters[split][metric].state_dict()
                for metric in self._EVAL_METRICS
            }
            for split in self.splits
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
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
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


def main(args):
    trainer = Trainer(args)
    print("===========================================================")
    print(f"Device: {trainer.device}")
    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    if trainer.args.train:
        print("===========================================================")
        print("Start training {} epochs ...".format(trainer.args.epochs))
        last_epoch = trainer.epoch - 1
        trainer.set_lr_scheduler(trainer.optimizer, last_epoch)
        epochs_range = range(trainer.epoch, trainer.args.epochs, 1)
        early_stop = False
        tq0 = tqdm.tqdm(epochs_range, total=len(epochs_range), desc='training epochs')
        for epoch in tq0:
            # trainer.adjust_learning_rate(epoch)
            train_performance = trainer.train(epoch, split='train')
            trainer.report_train(train_performance, epoch, split='train')
            val_metrics = trainer.evaluate_classification(epoch, split='val')
            if trainer.early_stopping is not None:
                val_loss = val_metrics['loss']
                trainer.early_stopping(val_loss)
                early_stop = trainer.early_stopping.stop
            last_epoch = (epoch == (trainer.args.epochs - 1)) or early_stop
            trainer.evaluate_wsol(epoch, split='val', save_xai=last_epoch, save_cams=last_epoch, log=last_epoch)
            if trainer.lr_scheduler is not None:
                trainer.lr_scheduler.step()
            # trainer.print_performances()
            trainer.report(epoch, split='val')
            trainer.epoch += 1
            trainer.save_checkpoint(epoch, split='val')
            if early_stop:
                print(f'Training stopped early after {trainer.epoch} epochs')
                break
    else:
        print("===========================================================")
        print("Final epoch evaluation on val set ...")
        trainer.evaluate(trainer.args.epochs, split='val', save_xai=True, save_cams=True, log=True)
        trainer.report(trainer.args.epochs, split='val')
    if trainer.args.dataset_name != 'ILSVRC':
        print("===========================================================")
        print("Final epoch evaluation on test set ...")
        trainer.evaluate(trainer.args.epochs, split='test', save_xai=True, save_cams=True, log=True)
        trainer.report(trainer.args.epochs, split='test')
    trainer.print_performances()
    trainer.save_performances()
    print("===========================================================")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Artifact URI: {mlflow.get_artifact_uri()}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Run Name: {run.info.run_name}")
    print("===========================================================")
    # MLFlow logging
    if trainer.args.train:
        info = {'epochs_planned': trainer.args.epochs, 'epochs_trained': trainer.epoch}
        if trainer.early_stopping is not None:
            info |= {'early_stop': str(trainer.early_stopping.early_stop).lower()}
        mlflow.log_dict(info, 'state/training.json')
    mlflow.log_metric("epochs", trainer.epoch)
    # mlflow.pytorch.log_model(trainer.model, 'model', pip_requirements='requirements.txt')

def SIGSEGV_signal_arises(signalNum, stack):
    print(f"{signalNum} : SIGSEGV arises")
    # Your code

import signal

if __name__ == '__main__':
    signal.signal(signal.SIGSEGV, SIGSEGV_signal_arises)
    args = get_configs()
    with mlflow.start_run() as run:
        with Logger(args.log_path):
                main(args)
        mlflow.log_artifact(args.log_path)