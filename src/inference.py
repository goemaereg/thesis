import cv2
import numpy as np
from evaluation import BoxEvaluator, MaskEvaluator, MultiEvaluator
from evaluation import configure_metadata, normalize_scoremap
from util import t2n
import os
import torch
from wsol.cam_method.utils.model_targets import ClassifierOutputTarget


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root, scoremap_root, cam_method,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, multi_gt_eval=False, cam_curve_interval=.001,
                 bbox_metric='MaxBoxAccV2', device='cpu', log=False):
        self.model = model
        self.model.eval()
        self.cam_method = cam_method
        self.loader = loader
        self.scoremap_root = scoremap_root
        self.split = split
        self.device = device
        self.log=log
        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator,
                          "SYNTHETIC": MultiEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval,
                                          multi_gt_eval=multi_gt_eval,
                                          metric=bbox_metric)


    def compute_and_evaluate_cams(self, save_cams=False):
        # print("Computing and evaluating cams.")
        metrics = {}
        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            # images = images.to(self.device) #.cuda()
            # result = self.model(images, targets, return_cam=True)
            # cams = result['cams'].detach().clone()
            output_targets = [ClassifierOutputTarget(target.item()) for target in targets]
            cams = self.cam_method(images, output_targets).astype('float')
            # cams = t2n(cams)
            cams_it = zip(cams, image_ids)
            for cam, image_id in cams_it:
                # cam_resized = cv2.resize(cam, image_size,
                #                          interpolation=cv2.INTER_CUBIC)
                # cam_normalized = normalize_scoremap(cam_resized)
                cam_normalized = cam
                if self.split in ('val', 'test') and save_cams:
                    cam_path = os.path.join(self.scoremap_root, image_id)
                    if not os.path.exists(os.path.dirname(cam_path)):
                        os.makedirs(os.path.dirname(cam_path))
                    np.save(cam_path, cam_normalized)
                self.evaluator.accumulate(cam_normalized, image_id)
        metrics = self.evaluator.compute()
        return metrics
