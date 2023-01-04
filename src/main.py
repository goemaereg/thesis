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
from dataloaders import get_data_loader, configure_metadata, get_image_sizes, get_bounding_boxes
from inference import CAMComputer, normalize_scoremap
from evaluation import MaskEvaluator, compute_bboxes_from_scoremaps
from util import string_contains_any, t2n  # , t2n
import wsol
# import wsol.method
import itertools
import tqdm
from os.path import join as ospj


def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
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
        self.args = get_configs()
        set_random_seed(self.args.seed)
        print(self.args)
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.model = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self._DEVICE)#.cuda()
        self.mse_loss = nn.MSELoss(reduction='mean').to(self._DEVICE)#.cuda()
        batch_set_size = None
        class_set_size = None
        if self.args.wsol_method == 'minmaxcam':
            batch_set_size = self.args.minmaxcam_batch_set_size
            class_set_size = self.args.minmaxcam_class_set_size

        self.optimizer = self._set_optimizer()
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
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold)
        model = model.to(self._DEVICE) # model.cuda()
        print(model)
        return model

    def _set_optimizer(self):
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

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer

    def _wsol_training(self, images, target):
        if (self.args.wsol_method == 'cutmix' and
                self.args.cutmix_prob > np.random.rand(1) and
                self.args.cutmix_beta > 0):
            images, target_a, target_b, lam = wsol.method.cutmix(
                images, target, self.args.cutmix_beta)
            output_dict = self.model(images)
            logits = output_dict['logits']
            loss = (self.cross_entropy_loss(logits, target_a) * lam +
                    self.cross_entropy_loss(logits, target_b) * (1. - lam))
            return logits, loss

        if self.args.wsol_method == 'has':
            images = wsol.method.has(images, self.args.has_grid_size,
                                     self.args.has_drop_rate)

        output_dict = self.model(images, target)
        logits = output_dict['logits']

        if self.args.wsol_method in ('acol', 'spg'):
            loss = wsol.method.__dict__[self.args.wsol_method].get_loss(
                output_dict, target, spg_thresholds=self.args.spg_thresholds)
        else:
            loss = self.cross_entropy_loss(logits, target)

        return logits, loss


    def vgg16_minmaxcam_regularization_loss(self, images, labels):
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.conv6.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        images = images.to(self._DEVICE)  # .cuda()
        # Compute CAMs from B(I) with I=input image
        result_orig = self.model(images, labels,
                                 return_cam=True, clone_cam=False)
        cams = result_orig['cams']
        # cams = t2n(cams)
        cams = cams.unsqueeze(1)
        # rescale cams to images input size
        resize = tuple(images.shape[2:])
        cams_resized = F.interpolate(cams, size=resize, mode='bilinear')
        # normalize cams
        cams_min = torch.amin(cams_resized, dim=(2, 3), keepdim=True)
        cams_max = torch.amax(cams_resized, dim=(2, 3), keepdim=True)
        cams_normalized = torch.zeros_like(cams_resized)
        if ((not torch.isnan(cams_resized).any()) and
            (not torch.equal(cams_min, cams_max))):
            cams_normalized = (cams_resized - cams_min)/(cams_max - cams_min)

        # Compute B(I * cams_normalized)
        result_mask = self.model(images * cams_normalized)
        x = self.model.features(images *cams_normalized)
        x = self.model.conv6(x)
        out_extra_masked = self.model.relu(x)

        # compute features_i
        features_i = result_mask['avgpool_flat']
        # compute features_o
        features_o = result_orig['avgpool_flat']

        # compute losses
        loss_crr = 0.0
        loss_frr = 0.0
        # compute ss and bs per target
        # ss and bs can be less than minmaxcam_class_set_size and minmaxcam_batch_set_size
        # for small datasets
        # ss = self.args.minmaxcam_class_set_size
        # bs = self.args.minmaxcam_batch_set_size
        bs = np.unique(labels).shape[0]
        starts = np.nonzero(np.r_[1, np.diff(labels)])[0]
        stops = np.nonzero(np.r_[0, np.diff(labels), 1])[0]
        for start, stop in zip(starts, stops):
            loss_crr_ss = 0.0
            f_i_pair_num = 0
            features_i_ss = features_i[start:stop]
            features_o_ss = features_o[start:stop]
            ss = stop - start
            for j, k in itertools.combinations(range(ss), 2):
                f_i_pair_num += 1
                feature_i_ss_j = features_i_ss[j].unsqueeze(0)
                feature_i_ss_k = features_i_ss[k].unsqueeze(0)
                loss_crr_ss += self.mse_loss(feature_i_ss_j, feature_i_ss_k)
            loss_frr += self.mse_loss(features_i_ss, features_o_ss)
            loss_crr += loss_crr_ss / f_i_pair_num
        loss_crr /= bs
        loss_frr /= bs
        return loss_crr, loss_frr


    def train(self, split):
        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        tq0 = tqdm.tqdm(loader, total=len(loader), desc='train batches')
        for batch_idx, (images, targets, _) in enumerate(tq0):
            images = images.to(self._DEVICE) # images.cuda()
            targets = targets.to(self._DEVICE) #.cuda()

            if int(batch_idx % max(len(loader) // 10, 1)) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            # minmaxcam stage I
            if self.args.wsol_method == 'minmaxcam':
                for param in self.model.features.parameters():
                    param.requires_grad = True
                for param in self.model.conv6.parameters():
                    param.requires_grad = True
                for param in self.model.fc.parameters():
                    param.requires_grad = True

            self.model.train()
            logits, loss = self._wsol_training(images, targets)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # minmaxcam stage II
            if self.args.wsol_method == 'minmaxcam':
                for param in self.model.features.parameters():
                    param.requires_grad = False
                for param in self.model.conv6.parameters():
                    param.requires_grad = False
                for param in self.model.fc.parameters():
                    param.requires_grad = True

                self.optimizer.zero_grad()
                self.model.eval()

                loss_crr, loss_frr = \
                    self.vgg16_minmaxcam_regularization_loss(images, targets)

                self.model.train()
                loss_all_p2 = self.args.minmaxcam_crr_weight * loss_crr + \
                              self.args.minmaxcam_frr_weight * loss_frr
                try:
                    loss_all_p2.backward()
                except RuntimeError as e:
                    print(e)
                    print(loss_all_p2)
                self.optimizer.step()

        loss_average = total_loss / float(num_images)
        if loss_average > 1000:
            print(loss_average)
        classification_acc = num_correct / float(num_images) # * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc,
                    loss=loss_average)


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

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        tq0 = tqdm.tqdm(loader, total=len(loader), desc='compute_accuracy')
        for _, (images, targets, image_ids) in enumerate(tq0):
            images = images.to(self._DEVICE) #.cuda()
            targets = targets.to(self._DEVICE) #.cuda()
            output_dict = self.model(images)
            pred = output_dict['logits'].argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) # * 100
        return classification_acc

    def save_cams(self, loader, split, evaluator):
        has_opt_cam_thresh = not isinstance(evaluator, MaskEvaluator)
        # dummy init to get rid of pycharm warning that this variable can be accessed before assignment
        opt_cam_thresh = 0
        if has_opt_cam_thresh:
            opt_cam_thresh, opt_cam_thresh_index = evaluator.compute_optimal_cam_threshold(50)
        metadata_root = os.path.join(self.args.metadata_root, split)
        metadata = configure_metadata(metadata_root)
        image_sizes = get_image_sizes(metadata)
        gt_bbox_dict = get_bounding_boxes(metadata)
        tq0 = tqdm.tqdm(loader, total=len(loader), desc='evaluate_save_cam_batches')
        for images, targets, image_ids in tq0:
            images = images.to(self._DEVICE)  # .cuda()
            result = self.model(images, targets, return_cam=True)
            cams = result['cams'].detach().clone()
            cams = t2n(cams)
            cams_it = zip(cams, image_ids)
            tq1 = tqdm.tqdm(cams_it, total=len(cams), desc='evaluate_save_cams')
            for cam, image_id in tq1:
                # render the CAM heatmap
                data_root = loader.dataset.data_root
                path_img = os.path.join(data_root, image_id)
                img = cv2.imread(path_img) # color channels in BGR format
                orig_img_shape = image_sizes[image_id]
                _cam = cv2.resize(cam, orig_img_shape, interpolation=cv2.INTER_CUBIC)
                _cam_norm = normalize_scoremap(_cam)
                _cam_grey = (_cam_norm * 255).astype('uint8')
                heatmap = cv2.applyColorMap(_cam_grey, cv2.COLORMAP_JET)
                cam_annotated = heatmap * 0.3 + img * 0.5
                cam_path = ospj(self.args.log_folder, 'scoremaps', split, image_id)
                if not os.path.exists(os.path.dirname(cam_path)):
                    os.makedirs(os.path.dirname(cam_path))
                cv2.imwrite(cam_path, cam_annotated)

                # render mask and bbox CAM heatmap
                image = None
                if has_opt_cam_thresh:
                    gt_bbox_list = gt_bbox_dict[image_id]
                    est_bbox_per_thresh, _ = compute_bboxes_from_scoremaps(
                        _cam_norm, [opt_cam_thresh],
                        multi_contour_eval=self.args.multi_contour_eval)
                    est_bbox_list = est_bbox_per_thresh[0]
                    if (len(gt_bbox_list) + len(est_bbox_list)) > 0:
                        image = img
                        thickness = 2  # Pixels
                        for bbox in gt_bbox_list:
                            start, end = bbox[:2], bbox[2:]
                            color = (0, 255, 0) # Green color in BGR
                            image = cv2.rectangle(image, start, end, color, thickness)
                        for bbox in est_bbox_list:
                            start, end = bbox[:2], bbox[2:]
                            color = (0, 0, 255) # Red color in BGR
                            image = cv2.rectangle(image, start, end, color, thickness)
                if image is not None:
                    img_ann_id = f'{image_id.split(".")[0]}_ann.png'
                    img_ann_path = ospj(self.args.log_folder, 'scoremaps', split, img_ann_id)
                    cv2.imwrite(img_ann_path, image)

    def evaluate(self, epoch, split, save_cams=False):
        print("Evaluate epoch {}, split {}".format(epoch, split))
        self.model.eval()
        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)

        cam_computer = CAMComputer(
            model=self.model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            multi_gt_eval=self.args.multi_gt_eval,
            log_folder=self.args.log_folder,
            device = self._DEVICE,
            bbox_metric=self.args.bbox_metric
        )
        metrics = cam_computer.compute_and_evaluate_cams()
        for metric, value in metrics.items():
            self.performance_meters[split][metric].update(value)
        if save_cams and split in ('val', 'test'):
            self.save_cams(self.loaders[split], split, cam_computer.evaluator)


    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('best'), epoch)
        if self.args.epochs == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)

    def report_train(self, train_performance, epoch, split='train'):
        reporter_log_root = ospj(self.args.reporter_log_root, split)
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
        reporter_log_root = ospj(self.args.reporter_log_root, split)
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

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


def main():
    trainer = Trainer()
    print("===========================================================")
    print(f"Accelerator: {accelerator_get()}")
    if trainer.args.train:
        print("Evaluate epoch 0 ...")
        trainer.evaluate(epoch=0, split='val')
        trainer.print_performances()
        trainer.report(epoch=0, split='val')
        trainer.save_checkpoint(epoch=0, split='val')
        print("Epoch 0 done.")
        tq0 = tqdm.tqdm(range(trainer.args.epochs), total=trainer.args.epochs, desc='training epochs')
        for epoch in tq0:
            print("===========================================================")
            print("Start training epoch {} ...".format(epoch + 1))
            trainer.adjust_learning_rate(epoch + 1)
            train_performance = trainer.train(split='train')
            trainer.report_train(train_performance, epoch + 1, split='train')
            trainer.evaluate(epoch + 1, split='val')
            trainer.print_performances()
            trainer.report(epoch + 1, split='val')
            trainer.save_checkpoint(epoch + 1, split='val')
            print("Epoch {} done.".format(epoch + 1))
    print("===========================================================")
    print("Final epoch evaluation on test set ...")
    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    trainer.evaluate(trainer.args.epochs, split='test', save_cams=True)
    trainer.print_performances()
    trainer.report(trainer.args.epochs, split='test')
    trainer.save_performances()


if __name__ == '__main__':
    main()
