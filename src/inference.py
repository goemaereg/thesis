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
from wsol.cam_method.utils.timer import Timer


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
    cam_args = dict(model=model, target_layers=target_layers, device=device)
    return cam_methods[cam_method], cam_args


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root, scoremap_root, cam_method,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, multi_gt_eval=False, cam_curve_interval=.001,
                 bbox_metric='MaxBoxAccV2', device='cpu', scoremap_storage_limit=200, log=False):
        self.model = model
        self.model.eval()
        self.cam_algorithm, self.cam_args = get_cam_algorithm(model, cam_method, device)
        self.loader = loader
        self.scoremap_root = scoremap_root
        self.scoremap_storage_limit = scoremap_storage_limit
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
        cams_metadata = {}
        metrics = {}
        timer_cam = Timer.create_or_get_timer(self.device, 'runtime_cam', warm_up=True)
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
                timer_cam.start()
                cams = cam_method(images, output_targets).astype('float')
                timer_cam.stop()
            # cams = t2n(cams)
            cams_it = zip(cams, image_ids)
            for cam, image_id in cams_it:
                # cam_resized = cv2.resize(cam, image_size,
                #                          interpolation=cv2.INTER_CUBIC)
                # cam_normalized = normalize_scoremap(cam_resized)
                # already resized to 224x224 and normalized during CAM computation
                cam_normalized = cam
                if self.split in ('val', 'test') and save_cams:
                    if len(cams_metadata) < self.scoremap_storage_limit:
                        cam_id = os.path.join(self.split, f'{os.path.basename(image_id)}.npy')
                        cam_path = os.path.join(self.scoremap_root, cam_id)
                        if not os.path.exists(os.path.dirname(cam_path)):
                            os.makedirs(os.path.dirname(cam_path))
                        np.save(cam_path, cam_normalized)
                        cams_metadata[image_id] = cam_id
                self.evaluator.accumulate(cam_normalized, image_id)
        metrics |= self.evaluator.compute()
        metrics |= {name: timer.get_total_elapsed_ms() for name, timer in Timer.timers.items()}
        Timer.reset()
        # write scoremap metadata
        # format: image_id,cam_id
        if len(cams_metadata) > 0:
            metadata =  dict(sorted(cams_metadata.items()))
            lines = [f'{image_id},{cam_id}' for image_id, cam_id in metadata.items()]
            with open(os.path.join(self.scoremap_root, self.split, 'scoremap_metadata.txt'), 'w') as fp:
                fp.writelines('\n'.join(lines))
        return metrics
