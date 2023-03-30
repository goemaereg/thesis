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

import numpy as np
from data_loaders import get_eval_loader
from evaluation import BoxEvaluator, MaskEvaluator
from evaluation import configure_metadata, normalize_scoremap
import os
import tqdm
import mlflow
from wsol.cam_method.utils.model_targets import ClassifierOutputTarget
from wsol.cam_method import CAM, GradCAM, GradCAMPlusPlus, ScoreCAM
from wsol.cam_method.utils.timer import Timer
from torch.nn.functional import softmax
import cv2

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224

cam_methods = {
    'cam': CAM,
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'scorecam': ScoreCAM,
    'minmaxcam': CAM
}

def get_cam_algorithm(model, cam_method, device):
    target_layers = None
    if model.__class__.__name__ == 'VggCam':
        target_layers = [model.conv6_relu]
    elif model.__class__.__name__ == 'Vgg':
        target_layers = [model.features[-2]]  # last conv+relu layer
    elif model.__class__.__name__ == 'ResNetCam':
        target_layers = [model.layer4]
    if target_layers is None:
        raise NotImplementedError
    cam_args = dict(model=model, target_layers=target_layers, device=device)
    return cam_methods[cam_method], cam_args


class CAMComputer(object):
    @staticmethod
    def get_evaluators(**args):
        dataset_name = args.get('dataset_name', 'SYNTHETIC')
        evaluators = {
            "ILSVRC": (BoxEvaluator, None),
            "SYNTHETIC": (BoxEvaluator, MaskEvaluator)
        }[dataset_name]
        return tuple(map(lambda x: x(**args) if x is not None else None, evaluators))

    def __init__(self, model, device, split, config, loader, log=False):
        self.model = model
        self.device = device
        self.split = split
        self.config = config
        self.model.eval()
        self.cam_algorithm, self.cam_args = get_cam_algorithm(model, config.wsol_method, device)
        self.loader = loader
        self.data_root = self.config.data_paths[split]
        self.metadata_root = self.config.metadata_root
        self.scoremap_root = config.scoremap_root
        self.bbox_iter = config.bbox_iter
        self.bbox_iter_max = max(1, config.bbox_iter_max if config.bbox_iter else 1)
        self.log = log
        self.bboxes_meta_path = os.path.join(self.scoremap_root, self.split, 'bboxes_metadata.txt')
        self.optimal_thresholds_path = os.path.join(self.scoremap_root, self.split, 'optimal_thresholds.npy')

        metadata = configure_metadata(os.path.join(config.metadata_root, split))
        cam_threshold_list = list(np.arange(0, 1, config.cam_curve_interval))
        eval_args = dict(
            metadata=metadata,
            dataset_name=config.dataset_name,
            split=split,
            cam_threshold_list=cam_threshold_list,
            iou_threshold_list=config.iou_threshold_list,
            mask_root=config.mask_root,
            multi_contour_eval=config.multi_contour_eval,
            multi_gt_eval=config.multi_gt_eval,
            bbox_merge_strategy=config.bbox_merge_strategy,
            bbox_merge_iou_threshold=config.bbox_merge_iou_threshold,
            log=log)
        self.box_evaluator, self.mask_evaluator = self.get_evaluators(**eval_args)

    def compute_and_evaluate_cams(self, save_cams=False):
        # print("Computing and evaluating cams.")
        cams_metadata = {}
        contexts = {}
        metrics = {}
        optimal_threshold_list = []
        timer_cam = Timer.create_or_get_timer(self.device, 'runtime_cam', warm_up=True)
        tq0 = tqdm.tqdm(range(self.bbox_iter_max), total=self.bbox_iter_max, desc='iterative bbox extraction')
        for iter_index in tq0:
            iter_processed = 0
            if self.box_evaluator:
                self.box_evaluator.reset()
            if self.mask_evaluator:
                self.mask_evaluator.reset()
            optimal_threshold_index = 0
            bbox_mask_strategy = self.config.bbox_mask_strategy if iter_index > 0 else None
            loader = get_eval_loader(
                self.split, self.data_root, self.metadata_root, self.config.batch_size, self.config.workers,
                self.config.crop_size,
                bboxes_path=self.bboxes_meta_path, bbox_mask_strategy=bbox_mask_strategy)
            tq1 = tqdm.tqdm(loader, total=len(loader), desc='evaluate cams batches')
            for images, targets, image_ids in tq1:
                image_size = images.shape[2:]
                # Using the with statement ensures the context is freed, and you can
                # recreate different CAM objects in a loop.
                output_targets = [ClassifierOutputTarget(target.item()) for target in targets]
                with self.cam_algorithm(**self.cam_args) as cam_method:
                    timer_cam.start()
                    # cam_method returns numpy array
                    cams, outputs = cam_method(images, output_targets)
                    cams = cams.astype('float')
                    timer_cam.stop()
                cams_it = zip(cams, outputs, output_targets, image_ids, images)
                for cam, output, output_target, image_id, image in cams_it:
                    # softmax: extract prediction probability from logit scores
                    y = np.exp(output - np.max(output)) # numerically stable version of softmax
                    smax = y / np.sum(y)
                    prob = output_target(smax)
                    context = contexts[image_id] if image_id in contexts else None
                    if context is not None and 'prob' in context:
                        prev_prob = context['prob']
                        if prob - prev_prob < -0.1:
                            # prob of prediction decreased at least with 10% confidence
                            continue
                    # cam is already resized to 224x224 and normalized by cam_method call
                    cam_id = os.path.join(self.split, f'{os.path.basename(image_id)}.npy')
                    cam_path = os.path.join(self.scoremap_root, cam_id)
                    if not os.path.exists(os.path.dirname(cam_path)):
                        os.makedirs(os.path.dirname(cam_path))
                    cam_list = []
                    if os.path.exists(cam_path):
                        cam_loaded = np.load(cam_path)
                        cam_list = [cam_loaded]
                    # add new cam to cam list
                    cam_list.append(cam)
                    # stack cams
                    cam_stack = np.stack(cam_list, axis=0)
                    cam_merged = np.max(cam_stack, axis=0)
                    np.save(cam_path, cam_merged)
                    cams_metadata[image_id] = cam_id
                    bbox_context = {}
                    if self.box_evaluator:
                        bbox_context = self.box_evaluator.accumulate_bboxes(cam, image_id, context)
                        self.box_evaluator.accumulate_boxacc(bbox_context)
                    if self.mask_evaluator:
                        # merge cams of previous iterations with current cam
                        self.mask_evaluator.accumulate(cam_merged, image_id)
                    contexts[image_id] = {'image_id': image_id, 'prob': prob} | bbox_context
                    iter_processed += 1
            if self.box_evaluator:
                metrics |= self.box_evaluator.compute()
                optimal_threshold, optimal_threshold_index = self.box_evaluator.compute_optimal_cam_threshold(50)
                optimal_threshold_list.append(optimal_threshold)
            if self.mask_evaluator:
                metrics |= self.mask_evaluator.compute()
            metrics |= {'iter_processed': iter_processed}
            mlflow_metrics = {f'{self.split}_{metric}': value for metric, value in metrics.items()}
            mlflow.log_metrics(mlflow_metrics, step=iter_index)
            # write scoremap metadata
            # format: image_id,cam_id
            if len(cams_metadata) > 0 and iter_index == 0:
                metadata = dict(sorted(cams_metadata.items()))
                lines = [f'{image_id},{cam_id}' for image_id, cam_id in metadata.items()]
                scoremap_meta_path = os.path.join(self.scoremap_root, self.split, 'scoremap_metadata.txt')
                if not os.path.exists(os.path.dirname(scoremap_meta_path)):
                    os.makedirs(os.path.dirname(scoremap_meta_path))
                with open(scoremap_meta_path, 'w') as fp:
                    fp.writelines('\n'.join(lines))
            # write bboxes metadata
            if not os.path.exists(os.path.dirname(self.bboxes_meta_path)):
                os.makedirs(os.path.dirname(self.bboxes_meta_path))
            with open(self.bboxes_meta_path, 'w') as fp:
                for image_id, context in contexts.items():
                    bboxes = context['thresh_boxes'][optimal_threshold_index]
                    for bbox in bboxes:
                        x0,y0,x1,y1 = bbox
                        line = f'{image_id},{x0},{y0},{x1},{y1}\n'
                        fp.write(line)
            # write optimal threshold list
            if not os.path.exists(os.path.dirname(self.optimal_thresholds_path)):
                os.makedirs(os.path.dirname(self.optimal_thresholds_path))
            np.save(self.optimal_thresholds_path, np.asarray(optimal_threshold_list))
        metric_timers = {name: timer.get_total_elapsed_ms() for name, timer in Timer.timers.items()}
        mlflow.log_metrics(metric_timers)
        Timer.reset()
        # return most recent metrics (i.e. from last iteration)
        return metrics | metric_timers
