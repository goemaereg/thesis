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
from evaluation import BoxEvaluator, MaskEvaluator, MultiEvaluator
from evaluation import configure_metadata
import os
import tqdm
from wsol.cam_method.utils.model_targets import ClassifierOutputTarget
from wsol.cam_method import CAM, GradCAM, ScoreCAM
import torch
from torchvision.models import mobilenet_v2
import time, gc


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224

cam_methods = {
    'cam': CAM,
    'gradcam': GradCAM,
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
    use_cuda = ('cuda' in device)
    cam_args = dict(model=model, target_layers=target_layers, use_cuda=use_cuda)
    return cam_methods[cam_method], cam_args


class Timer:
    def __init__(self, device, gc_disable=True):
        self.device = device
        self.gc_disable = gc_disable
        self.counters_ns = []
        if 'cuda' in device:
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)
            self.warm_up()
        else:
            self.counter_start = None
            self.counter_stop = None

    def warm_up(self):
        if 'cuda' not in self.device:
            return
        model = mobilenet_v2(weights=None)
        model.to(self.device)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(self.device)
            for _ in range(10):
                _ = model(dummy_input)
        torch.cuda.synchronize() # wait for warm up to complete actions on GPU

    def start(self):
        if self.gc_disable:
            gc.disable()
        if 'cuda' in self.device:
            self.starter.record()
        else:
            self.counter_start = time.monotonic_ns()
    def stop(self):
        # wait for gpu sync
        if 'cuda' in self.device:
            torch.cuda.synchronize()
            self.ender.record()
            time_ms = self.starter.elapsed_time(self.ender) # milliseconds
            self.counters_ns.append(time_ms * 1e6)
        else:
            self.counter_stop = time.monotonic_ns()
            time_ns = self.counter_stop - self.counter_start
            self.counters_ns.append(time_ns)
        if self.gc_disable:
            gc.enable()

    def get_total_time(self):
        return np.sum(self.counters_ns)


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root, scoremap_root, cam_method,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, multi_gt_eval=False, cam_curve_interval=.001,
                 bbox_metric='MaxBoxAccV2', device='cpu', log=False):
        self.model = model
        self.model.eval()
        self.cam_algorithm, self.cam_args = get_cam_algorithm(model, cam_method, device)
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
                                          metric=bbox_metric,
                                          log=log)


    def compute_and_evaluate_cams(self, save_cams=False):
        # print("Computing and evaluating cams.")
        metrics = {}
        timer = Timer(self.device)
        tq0 = tqdm.tqdm(self.loader, total=len(self.loader), desc='evaluate cams batches')
        for images, targets, image_ids in tq0:
            image_size = images.shape[2:]
            # images =  images.to(self.device) #.cuda()
            # result = self.model(images, targets, return_cam=True)
            # cams = result['cams'].detach().clone()

            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            output_targets = [ClassifierOutputTarget(target.item()) for target in targets]
            with self.cam_algorithm(**self.cam_args) as cam_method:
                timer.start()
                cams = cam_method(images, output_targets).astype('float')
                timer.stop()
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
        metrics |= self.evaluator.compute()
        metrics |= {'cam_method_runtime': timer.get_total_time()}
        return metrics
