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
from util import t2n
from data_loaders import get_eval_loader, get_image_sizes
from evaluation import BoxEvaluator, MaskEvaluator
from evaluation import configure_metadata, xai_save_cams, xai_save_cam_inference
import os
import tqdm
import mlflow
from wsol.cam_method.utils.model_targets import ClassifierOutputTarget
from wsol.cam_method import CAM, GradCAM, GradCAMPlusPlus, ScoreCAM
from wsol.cam_method.utils.timer import Timer
import lmdb
import pickle

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
        self.iter_max = max(1, config.iter_max)
        self.log = log
        self.bboxes_meta_path = os.path.join(self.scoremap_root, self.split, 'bboxes_metadata.txt')
        self.optimal_thresholds_path = os.path.join(self.scoremap_root, self.split, 'optimal_thresholds.npy')
        self.step = 0
        self.metadata = configure_metadata(os.path.join(config.metadata_root, split))
        self.image_sizes = get_image_sizes(self.metadata)
        cam_threshold_list = list(np.arange(0, 1, config.cam_curve_interval))
        eval_args = dict(
            metadata=self.metadata,
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
        self.lmdb_scoremaps_path = os.path.join(self.scoremap_root, self.split, 'lmdb_scoremaps.lmdb')
        if not os.path.exists(os.path.dirname(self.lmdb_scoremaps_path)):
            os.makedirs(os.path.dirname(self.lmdb_scoremaps_path))
        self.db_commit_writes = 5000
        self.db_writes = 0

    def compute_and_evaluate_cams(self, epoch, save_xai=False):
        # print("Computing and evaluating cams.")
        xai_images_max = 100
        cams_saved = {}
        cams_stats = {}
        contexts = {}
        metrics = {}
        optimal_threshold_list = []
        timer_cam = Timer.create_or_get_timer(self.device, 'runtime_cam', warm_up=True)
        tq0 = tqdm.tqdm(range(self.iter_max), total=self.iter_max, desc='iterative bbox extraction')
        for iter_index in tq0:
            if self.box_evaluator:
                self.box_evaluator.reset()
            if self.mask_evaluator:
                self.mask_evaluator.reset()
            xai_images = 0
            num_skipped = 0
            optimal_threshold_index = 0
            bbox_mask_strategy = self.config.bbox_mask_strategy if iter_index > 0 else None
            loader = get_eval_loader(
                self.split, self.data_root, self.metadata_root, self.config.batch_size, self.config.workers,
                self.config.crop_size,
                bboxes_path=self.bboxes_meta_path, bbox_mask_strategy=bbox_mask_strategy)
            tq1 = tqdm.tqdm(loader, total=len(loader), desc='evaluate cams batches')
            self.db = lmdb.open(self.lmdb_scoremaps_path, subdir=False,
                                map_size=68719476736, readonly=False,  # map_size 68 GB
                                meminit=False, map_async=True)
            self.db_txn = self.db.begin(write=True)
            for images, targets, image_ids in tq1:
                # Using the with statement ensures the context is freed, and you can
                # recreate different CAM objects in a loop.
                output_targets = [ClassifierOutputTarget(target.item()) for target in targets]
                with self.cam_algorithm(**self.cam_args) as cam_method:
                    timer_cam.start()
                    # cam_method returns tuple of numpy arrays
                    # cams is already resized to 224x224 and normalized by cam_method call
                    cams, outputs = cam_method(images, output_targets)
                    cams = cams.astype('float')
                    timer_cam.stop()
                cams_it = zip(cams, outputs, output_targets, image_ids, images)
                for cam, output, output_target, image_id, image in cams_it:
                    if self.config.xai and save_xai:
                        if xai_images < xai_images_max:
                            xai_save_cam_inference(
                                xai_root=self.config.xai_root, metadata=self.metadata, split=self.split,
                                image_id=image_id, image=t2n(image), size_orig=self.image_sizes[image_id], cam=cam,
                                iter_index=iter_index, log=True)
                    xai_images += 1
                    context = contexts[image_id] if image_id in contexts else None
                    # softmax: extract prediction probability from logit scores
                    y = np.exp(output - np.max(output))  # numerically stable version of softmax
                    smax = y / np.sum(y)
                    prob = float(output_target(smax))
                    skip = False
                    if context is not None:
                        # check whether this image is tagged as to be skipped
                        skip = context.get('skip', False)
                        prev_prob = context.get('prob', prob)
                        if skip:
                            prob = prev_prob
                        elif (prev_prob - prob) >= self.config.iter_stop_prob_delta:
                            skip = True
                            prob = prev_prob
                    cam_merged = cam
                    if image_id in cams_saved:
                        raw = self.db_txn.get(u'{}'.format(image_id).encode('ascii'))
                        cam_loaded = pickle.loads(raw)
                        if skip:
                            cam_merged = cam_loaded
                        else:
                            cam_merged = np.max(np.stack([cam, cam_loaded], axis=0), axis=0)
                    if not skip:
                        # save into lmdb
                        self.db_txn.put(u'{}'.format(image_id).encode('ascii'), pickle.dumps(cam_merged))
                        self.db_writes += 1
                        if self.db_writes % self.db_commit_writes == 0:
                            self.db_txn.commit()
                            self.db_txn = self.db.begin(write=True)
                        cam_id = os.path.join(self.split, f'{os.path.basename(image_id)}.npy')
                        cams_saved[image_id] = cam_id
                    bbox_context = {}
                    if self.box_evaluator:
                        if skip:
                            bbox_context = context
                        else:
                            bbox_context = self.box_evaluator.accumulate_bboxes(cam, image_id, context)
                        self.box_evaluator.accumulate_boxacc(bbox_context)
                    if self.mask_evaluator:
                        self.mask_evaluator.accumulate(cam_merged, image_id)
                    contexts[image_id] = bbox_context | {'image_id': image_id, 'prob': prob, 'skip': skip}
                    if image_id not in cams_stats:
                        cams_stats[image_id] = dict(prob=[prob], skip=[skip])
                    else:
                        cams_stats[image_id]['prob'].append(prob)
                        cams_stats[image_id]['skip'].append(skip)
                    if skip:
                        num_skipped += 1
            if self.box_evaluator:
                metrics |= self.box_evaluator.compute()
                optimal_threshold, optimal_threshold_index = self.box_evaluator.compute_optimal_cam_threshold(50)
                optimal_threshold_list.append(optimal_threshold)
            if self.mask_evaluator:
                metrics |= self.mask_evaluator.compute()
            metrics |= {'iter_skipped': num_skipped}
            mlflow_metrics = {f'{self.split}_{metric}': value for metric, value in metrics.items()}
            mlflow.log_metrics(mlflow_metrics, step=self.step)
            self.step += 1
            # write scoremap metadata
            # format: image_id,cam_id
            if len(cams_saved) > 0 and iter_index == 0:
                metadata = dict(sorted(cams_saved.items()))
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
                        x0, y0, x1, y1 = bbox
                        line = f'{image_id},{x0},{y0},{x1},{y1}\n'
                        fp.write(line)
            # write optimal threshold list
            if not os.path.exists(os.path.dirname(self.optimal_thresholds_path)):
                os.makedirs(os.path.dirname(self.optimal_thresholds_path))
            np.save(self.optimal_thresholds_path, np.asarray(optimal_threshold_list))
            # close lmdb transactions
            self.db_txn.commit()
            self.db.sync()
            self.db.close()
            if self.config.xai and save_xai:
                metadata_root = os.path.join(self.config.metadata_root, self.split)
                metadata = configure_metadata(metadata_root)
                xai_save_cams(xai_root=self.config.xai_root,
                              split=self.split,
                              metadata=metadata,
                              data_root=self.config.data_paths[self.split],
                              scoremap_root=self.config.scoremap_root,
                              log=True,
                              images_max=xai_images_max,
                              iter_index=iter_index)
        # metrics logging
        metric_timers = {name: timer.get_total_elapsed_ms() for name, timer in Timer.timers.items()}
        mlflow.log_metrics(metric_timers, step=epoch)
        mlflow.log_dict(cams_stats, f'state/{self.split}/{epoch}/cams_stats.json')
        Timer.reset()
        # return most recent metrics (i.e. from last iteration)
        return metrics | metric_timers
