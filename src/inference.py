import cv2
import numpy as np
from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import MultiEvaluator
from evaluation import configure_metadata
from dataloaders import get_image_sizes
from util import t2n
import tqdm

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, multi_gt_eval=False, cam_curve_interval=.001,
                 bbox_metric='MaxBoxAccV2', log_folder=None,
                 device='cpu'):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.device = device
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
        self.image_sizes = get_image_sizes(metadata)


    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        tq0 = tqdm.tqdm(self.loader, total=len(self.loader), desc='evaluate_cam_batches')
        for images, targets, image_ids in tq0:
            image_size = images.shape[2:]
            images = images.to(self.device) #.cuda()
            result = self.model(images, targets, return_cam=True)
            cams = result['cams'].detach().clone()
            cams = t2n(cams)
            cams_it = zip(cams, image_ids)
            tq1 = tqdm.tqdm(cams_it, total=len(cams), desc='evaluate_cams')
            for cam, image_id in tq1:
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                self.evaluator.accumulate(cam_normalized, image_id)
        return self.evaluator.compute()
