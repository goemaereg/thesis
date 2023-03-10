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
import cv2
import numpy as np
import os
import torch.utils.data as torchdata
import tqdm
import mlflow
import matplotlib.pyplot as plt
from config import str2bool, configure_bbox_metric, configure_mask_root
from data_loaders import configure_metadata
from data_loaders import get_image_ids
from data_loaders import get_bounding_boxes
from data_loaders import get_image_sizes
from data_loaders import get_mask_paths
from util import check_scoremap_validity
from util import check_box_convention
from util import t2n

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
_BBOX_METRIC_NAMES = ('MaxBoxAcc', 'MaxBoxAccV2', 'MaxBoxAccV3')
_BBOX_METRIC_DEFAULT = 'MaxBoxAccV2'
_DATASET_NAMES = ('CUB', 'ILSVRC', 'OpenImages', 'SYNTHETIC')
_DATASET_DEFAULT = 'SYNTHETIC'
_SPLIT_NAMES = ('val', 'test')
_SPLIT_DEFAULT = 'test'

def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.float, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])
        return np.asarray(estimated_boxes), len(contours)


    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


class CamDataset(torchdata.Dataset):
    def __init__(self, scoremap_path, split, image_ids):
        self.scoremap_path = scoremap_path
        self.split = split
        self.image_ids = image_ids
        self.length = len(os.listdir(os.path.join(scoremap_path, split)))

    def _load_cam(self, image_id):
        scoremap_file = os.path.join(self.scoremap_path, self.split, os.path.basename(image_id) + '.npy')
        return np.load(scoremap_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam = self._load_cam(image_id)
        return cam, image_id

    def __len__(self):
        return self.length


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metric, metadata, dataset_name, split, cam_threshold_list,
                 iou_threshold_list, mask_root, multi_contour_eval, multi_gt_eval=False, log=False):
        self.metric = metric
        self.log=log
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.mask_root = mask_root
        self.multi_gt_eval = multi_gt_eval
        self.multi_contour_eval = multi_contour_eval

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def compute_optimal_cam_threshold(self, iou_threshold):
        raise NotImplementedError


class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, metric='MaxBoxAccV2', **kwargs):
        super(BoxEvaluator, self).__init__(metric=metric, **kwargs)
        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        self.cnt = 0
        self.num_correct = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate_maxboxacc_v1_2(self, multiple_iou, number_of_box_list):
        # Computes best match (1 box) over sets of estimated and GT-boxes per threshold
        # Result is 1 IOU value per threshold
        """
        Computes best match (1 box) over sets of estimated_boxes and GT-boxes per threshold
        Result per IOU threshold is 1 IOU value per scoremap threshold

        Args:
            multiple_iou: numpy.ndarray(dtype=np.float,
                          shape=(num estimated boxes over all scoremap thresholds, num GT-boxes))
        """
        idx = 0
        sliced_multiple_iou = []
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            idx += nr_box
        # Compute true positives over different IOU thresholds
        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD / 100))[0]
            self.num_correct[_THRESHOLD][correct_threshold_indices] += 1
        self.cnt += 1

    def accumulate_maxboxacc_v3(self, multiple_iou, number_of_box_list):
        """
        Computes best match per threshold per gt-box over sets of estimated_boxes[threshold]
        Result per IOU threshold is 1 IOU value per scoremap threshold

        Args:
            multiple_iou: numpy.ndarray(dtype=np.float,
                                        shape=(num estimated boxes in all thresholded scoremaps, num GT-boxes)
                                       )
        """
        idx = 0
        sliced_multiple_iou = []
        num_thresholds = len(self.cam_threshold_list)
        num_gt_boxes = multiple_iou.shape[1]
        for nr_box in number_of_box_list:
            gt_iou_max = []
            slice_multi_iou = multiple_iou[idx : idx + nr_box].copy()
            for _ in range(num_gt_boxes):
                max_iou_index = np.unravel_index(np.argmax(slice_multi_iou), shape=slice_multi_iou.shape)
                gt_iou_max.append(slice_multi_iou[max_iou_index[0], max_iou_index[1]])
                # mark max IOU as unavailable ('0') to other (est, gt) combinations in this slice
                slice_multi_iou[max_iou_index[0], :] = 0
                slice_multi_iou[:, max_iou_index[1]] = 0
            sliced_multiple_iou.append(np.asarray(gt_iou_max))
            idx += nr_box
        multi_iou_per_threshold = np.asarray(sliced_multiple_iou)

        # Compute true positives over different IOU thresholds
        for _THRESHOLD in self.iou_threshold_list:
            num_correct_multi = np.zeros(shape=(num_thresholds, num_gt_boxes))
            correct_threshold_indices = np.where(multi_iou_per_threshold >= (_THRESHOLD / 100))
            num_correct_multi[correct_threshold_indices] = 1
            # reduce to a single score per threshold = true positives (matching GT boxes) per threshold
            self.num_correct[_THRESHOLD] += np.sum(num_correct_multi, axis=1)
        self.cnt += num_gt_boxes


    def accumulate(self, scoremap, image_id):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """

        # Computes a set of estimated boxes per scoremap threshold
        # Returns an array of threshold-related np.array objects containing estimated boxes
        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)

        # concatenates sets of boxes per threshold into a single array of boxes over all thresholds
        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        # Computes IOU of all combinations of sets of estimated and set of ground-truth boxes
        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))

        if self.multi_gt_eval:
            self.accumulate_maxboxacc_v3(multiple_iou, number_of_box_list)
        else:
            self.accumulate_maxboxacc_v1_2(multiple_iou, number_of_box_list)

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        metrics = {}
        box_acc_iou = {}
        max_box_acc_iou = []
        max_box_acc_threshold = []
        for _THRESHOLD in self.iou_threshold_list:
            box_acc_iou[_THRESHOLD] = self.num_correct[_THRESHOLD] * 1. / float(self.cnt)
            max_box_acc_iou.append(box_acc_iou[_THRESHOLD].max())
            cam_threshold_optimal = self.cam_threshold_list[box_acc_iou[_THRESHOLD].argmax()]
            max_box_acc_threshold.append(cam_threshold_optimal)
            if self.log:
                box_acc = {
                    'iou_threshold': _THRESHOLD,
                    'cam_threshold_optimal': cam_threshold_optimal,
                    'cam_threshold': self.cam_threshold_list,
                    'box_accuracy': box_acc_iou[_THRESHOLD].tolist()
                }
                log_path = f'data/{self.split}/box_acc_iou_{_THRESHOLD}.json'
                if not os.path.exists(os.path.dirname(log_path)):
                    os.makedirs(os.path.dirname(log_path))
                mlflow.log_dict(box_acc, log_path)
                fig, ax = plt.subplots()
                ax.plot(self.cam_threshold_list, box_acc_iou[_THRESHOLD])
                plt.title(f'{self.dataset_name} BoxAcc IOU={_THRESHOLD}')
                plt.xlabel('CAM threshold')
                plt.ylabel('BoxAcc')
                plt.axis('tight')
                log_path = f'plots/{self.split}/box_acc_iou_{_THRESHOLD}.png'
                if not os.path.exists(os.path.dirname(log_path)):
                    os.makedirs(os.path.dirname(log_path))
                mlflow.log_figure(fig, log_path)

        if self.metric == 'MaxBoxAcc':
            index_iou_50 = self.iou_threshold_list.index(50)
            metrics |= {self.metric: max_box_acc_iou[index_iou_50]}
            metrics |= {'optimal_threshold': max_box_acc_threshold[index_iou_50]}
        else:
            metrics |= {self.metric: np.average(max_box_acc_iou)}
        for index, _THRESHOLD in enumerate(self.iou_threshold_list):
            metrics |= {f'{self.metric}_IOU_{_THRESHOLD}': max_box_acc_iou[index]}
            metrics |= {f'optimal_threshold_IOU_{_THRESHOLD}': max_box_acc_threshold[index]}

        return metrics

    def compute_optimal_cam_threshold(self, iou_threshold):
        """
        Returns:
            optimal cam threshold t = max<t> BoxAcc(t, iou_threshold)
            index of optimal threshold in list of cam thresholds
        """
        box_acc_list = self.num_correct[iou_threshold] * 1. / float(self.cnt)
        optimal_threshold_index = np.argmax(box_acc_list)
        optimal_threshold = self.cam_threshold_list[optimal_threshold_index]
        return optimal_threshold, optimal_threshold_index


def load_mask_image(file_path, resize_size):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    return mask


def get_mask(mask_root, mask_paths, ignore_path):
    """
    Ignore mask is set as the ignore box region \setminus the ground truth
    foreground region.

    Args:
        mask_root: string.
        mask_paths: iterable of strings.
        ignore_path: string.

    Returns:
        mask: numpy.ndarray(size=(224, 224), dtype=np.uint8)
    """
    mask_all_instances = []
    for mask_path in mask_paths:
        mask_file = os.path.join(mask_root, mask_path)
        mask = load_mask_image(mask_file, (_RESIZE_LENGTH, _RESIZE_LENGTH))
        mask_all_instances.append(mask > 0.5)
    mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)
    has_ignore_mask = len(ignore_path) > 0
    if has_ignore_mask:
        ignore_file = os.path.join(mask_root, ignore_path)
        ignore_box_mask = load_mask_image(ignore_file,
                                          (_RESIZE_LENGTH, _RESIZE_LENGTH))
        ignore_box_mask = ignore_box_mask > 0.5

        ignore_mask = np.logical_and(ignore_box_mask,
                                     np.logical_not(mask_all_instances))

        if np.logical_and(ignore_mask, mask_all_instances).any():
            raise RuntimeError("Ignore and foreground masks intersect.")
        mask_all_instances = mask_all_instances.astype(np.uint8) + 255 * ignore_mask.astype(np.uint8)
    else:
        mask_all_instances = mask_all_instances.astype(np.uint8)
    return mask_all_instances


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(metric='PxAP', **kwargs)

        # if self.dataset_name != "OpenImages":
        #     raise ValueError("Mask evaluation must be performed on OpenImages.")

        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,[1.0])#, 2.0, 3.0])
        self.num_bins = len(self.threshold_list_right_edge) - 1
        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=float)

    def accumulate(self, scoremap, image_id):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        check_scoremap_validity(scoremap)
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id])

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(float)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(float)

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        precision_denominator = tp + fp
        precision_denominator[(tp + fp) == 0] = 1
        precision = tp / precision_denominator

        recall_denominator = tp + fn
        recall_denominator[(tp + fn) == 0] = 1
        recall = tp / recall_denominator

        # non_zero_indices = (tp + fp) != 0
        # auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc = (precision[1:] * np.diff(recall)).sum()
        if self.log:
            pr_curve = {
                'auc': auc,
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
            log_path = f'data/{self.split}/pr_curve.json'
            if not os.path.exists(os.path.dirname(log_path)):
                os.makedirs(os.path.dirname(log_path))
            mlflow.log_dict(pr_curve, log_path)
            fig, ax = plt.subplots()
            ax.plot(recall[1:], precision[1:])
            plt.title(f'{self.dataset_name} PR Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.axis('tight')
            log_path = f'plots/{self.split}/pr_curve.png'
            if not os.path.exists(os.path.dirname(log_path)):
                os.makedirs(os.path.dirname(log_path))
            mlflow.log_figure(fig, log_path)

        return {self.metric: auc}


class MultiEvaluator():
    def __init__(self, metric='MaxBoxAccV2', **kwargs):
        self.box_evaluator = BoxEvaluator(metric=metric, **kwargs)
        self.mask_evaluator = MaskEvaluator(**kwargs)

    def accumulate(self, scoremap, image_id):
        self.box_evaluator.accumulate(scoremap, image_id)
        self.mask_evaluator.accumulate(scoremap, image_id)

    def compute(self):
        return self.box_evaluator.compute() | self.mask_evaluator.compute()

    def compute_optimal_cam_threshold(self, iou_threshold):
        return self.box_evaluator.compute_optimal_cam_threshold(iou_threshold)


def _get_cam_loader(image_ids, scoremap_path, split):
    return torchdata.DataLoader(
        CamDataset(scoremap_path, split, image_ids),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

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

def xai_save_cams(xai_root, split, metadata, data_root, scoremap_root, evaluator, multi_contour_eval, log=False):
    has_opt_cam_thresh = not isinstance(evaluator, MaskEvaluator)
    # dummy init to get rid of pycharm warning that this variable can be accessed before assignment
    opt_cam_thresh = 0
    if has_opt_cam_thresh:
        opt_cam_thresh, opt_cam_thresh_index = evaluator.compute_optimal_cam_threshold(50)
    image_ids = get_image_ids(metadata)
    image_sizes = get_image_sizes(metadata)
    gt_bbox_dict = get_bounding_boxes(metadata)
    cam_loader = _get_cam_loader(image_ids, scoremap_root, split)
    tq0 = tqdm.tqdm(cam_loader, total=len(cam_loader), desc='xai_cam_batches')
    for cams, image_ids in tq0:
        cams = t2n(cams)
        cams_it = zip(cams, image_ids)
        for cam, image_id in cams_it:
            # render image overlayed with CAM heatmap
            path_img = os.path.join(data_root, image_id)
            img = cv2.imread(path_img) # color channels in BGR format
            orig_img_shape = image_sizes[image_id]
            # resize saved cam from 224x224 to original image size
            _cam = cv2.resize(cam, orig_img_shape, interpolation=cv2.INTER_CUBIC)
            _cam_norm = normalize_scoremap(_cam)
            _cam_mask = _cam_norm >= opt_cam_thresh
            # assign minimal value to area outside segment mask so normalization is constrained to segment values
            _cam_heatmap = _cam_norm.copy()
            _cam_heatmap[np.logical_not(_cam_mask)] = 0.0 # np.amin(_cam_norm[_cam_mask])
            # normalize
            _cam_heatmap = normalize_scoremap(_cam_heatmap)
            # mask out area outside segment mask
            _cam_heatmap[np.logical_not(_cam_mask)] = 0.0
            _cam_grey = (_cam_heatmap * 255).astype('uint8')
            heatmap = cv2.applyColorMap(_cam_grey, cv2.COLORMAP_JET)
            # mask out area outside segment mask
            heatmap[np.logical_not(_cam_mask)] = (0, 0, 0)
            cam_annotated = heatmap * 0.3 + img * 0.5
            cam_path = os.path.join(xai_root, image_id)
            if not os.path.exists(os.path.dirname(cam_path)):
                os.makedirs(os.path.dirname(cam_path))
            cv2.imwrite(cam_path, cam_annotated)
            if log:
                mlflow.log_artifact(cam_path, f'xai/{split}')

            # render image with annotations and CAM overlay
            # CAM overlay
            # _cam_mask = _cam_norm > 0
            segment = np.zeros(shape=img.shape)
            segment[_cam_mask] = (0, 0, 255)  # BGR
            img_ann = segment * 0.3 + img * 0.5
            # estimated and GT bboxes overlay
            if has_opt_cam_thresh:
                gt_bbox_list = gt_bbox_dict[image_id]
                est_bbox_per_thresh, _ = compute_bboxes_from_scoremaps(
                    _cam_norm, [opt_cam_thresh],
                    multi_contour_eval=multi_contour_eval)
                est_bbox_list = est_bbox_per_thresh[0]
                if (len(gt_bbox_list) + len(est_bbox_list)) > 0:
                    thickness = 2  # Pixels
                    for bbox in gt_bbox_list:
                        start, end = bbox[:2], bbox[2:]
                        color = (0, 255, 0) # Green color in BGR
                        img_ann = cv2.rectangle(img_ann, start, end, color, thickness)
                    for bbox in est_bbox_list:
                        start, end = bbox[:2], bbox[2:]
                        color = (0, 0, 255) # Red color in BGR
                        img_ann = cv2.rectangle(img_ann, start, end, color, thickness)

            img_ann_id = f'{image_id.split(".")[0]}_ann.png'
            img_ann_path = os.path.join(xai_root, img_ann_id)
            if not os.path.exists(os.path.dirname(img_ann_path)):
                os.makedirs(os.path.dirname(img_ann_path))
            cv2.imwrite(img_ann_path, img_ann)
            if log:
                mlflow.log_artifact(img_ann_path, f'xai/{split}')


def evaluate_wsol(xai_root, scoremap_root, data_root, metadata_root, mask_root,
                  iou_threshold_list, dataset_name, split,
                  multi_contour_eval, multi_gt_eval, cam_curve_interval=.01,
                  bbox_metric='MaxBoxAccV2', tags=None, xai=False):
    """
    Compute WSOL performances of predicted heatmaps against ground truth
    boxes (CUB, ILSVRC) or masks (OpenImages). For boxes, we compute the
    gt-known box accuracy (IoU>=0.5) at the optimal heatmap threshold.
    For masks, we compute the area-under-curve of the pixel-wise precision-
    recall curve.

    Args:
        scoremap_root: string. Score maps for each eval image are saved under
            the output_path, with the name corresponding to their image_ids.
            For example, the heatmap for the image "123/456.JPEG" is expected
            to be located at "{output_path}/123/456.npy".
            The heatmaps must be numpy arrays of type np.float, with 2
            dimensions corresponding to height and width. The height and width
            must be identical to those of the original image. The heatmap values
            must be in the [0, 1] range. The map must attain values 0.0 and 1.0.
            See check_scoremap_validity() in util.py for the exact requirements.
        metadata_root: string.
        mask_root: string.
        dataset_name: string. Supports [CUB, ILSVRC, and OpenImages].
        split: string. Supports [train, val, test].
        multi_contour_eval:  considering the best match between the set of all
            estimated boxes and the set of all ground truth boxes.
        multi_iou_eval: averaging the performance across various level of iou
            thresholds.
        iou_threshold_list: list. default: [30, 50, 70]
        cam_curve_interval: float. Default 0.01. At which threshold intervals
            will the heatmaps be evaluated?
    Returns:
        performance: dict. For CUB and ILSVRC, maxboxacc is returned.
            For OpenImages, area-under-curve of the precision-recall curve
            is returned.
    """
    if tags is None:
        tags = []
    print("Loading and evaluating cams.")
    metadata = configure_metadata(metadata_root)
    image_ids = get_image_ids(metadata)
    cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
    # The length of the output of np.arange might not be numerically stable.
    # Better to use np.linspace
    # cam_threshold_list = np.linspace(0, 1, num=int(1/cam_curve_interval),
    #                                  endpoint=False).tolist()

    evaluator = {"OpenImages": MaskEvaluator,
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

    cam_loader = _get_cam_loader(image_ids, scoremap_root)
    for cams, image_ids in cam_loader:
        for cam, image_id in zip(cams, image_ids):
            evaluator.accumulate(t2n(cam), image_id)
    metrics = evaluator.compute()
    for metric, value in metrics.items():
        print(f'{metric}: {value}')
    if xai is False:
        return
    # XAI
    xai_save_cams(xai_root=xai_root, split=split, metadata=metadata, data_root=data_root,
                  scoremap_root=scoremap_root, evaluator=evaluator, multi_contour_eval=multi_contour_eval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test_case')
    parser.add_argument('--log_folder', type=str, default='train_log', help='log folder')
    parser.add_argument('--scoremap_folder', type=str,
                        default='scoremaps',
                        help="The root folder for score maps to be evaluated.")
    parser.add_argument('--xai_folder', type=str, default='xai', help='xai folder')
    parser.add_argument('--data_root', type=str, default='dataset/',
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default='metadata/',
                        help="Root folder of metadata.")
    parser.add_argument('--mask_root', type=str, default='maskdata/',
                        help="Root folder of masks.")
    parser.add_argument('--dataset_name', type=str, default=_DATASET_DEFAULT,
                        choices=_DATASET_NAMES)
    parser.add_argument('--split', type=str, default=_SPLIT_DEFAULT,
                        choices=_SPLIT_NAMES)
    parser.add_argument('--split', type=str,
                        help="One of [val, test]. They correspond to "
                             "train-fullsup and test, respectively.")
    parser.add_argument('--cam_curve_interval', type=float, default=0.01,
                        help="At which threshold intervals will the score maps "
                             "be evaluated?.")
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])
    parser.add_argument('--bbox_metric', type=str, default=_BBOX_METRIC_DEFAULT,
                        choices=_BBOX_METRIC_NAMES)
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False, help=None)
    parser.add_argument('--xai', action=argparse.BooleanOptionalAction, default=False, help=None)
    # tags
    parser.add_argument('--dataset_name_suffix', type=str, default='',
                        help='Suffix = <tag1><tag2><tag3> used to partition SYNTHETHIC dataset. '
                             'tag1 = <choice o (overlapping) | d (disjunct)'
                             'tag2 = <n_instances: 0..4>'
                             'tag3 = <choice b (background) | t (transparent'),

    args = parser.parse_args()
    tags = []
    if args.dataset_name_suffix:
        tags.append('_'.join(list(args.dataset_name_suffix)))
    data_root = os.path.join(args.data_root, args.dataset_name, *tags)
    metadata_root = os.path.join(args.metadata_root, args.dataset_name, *tags)
    metadata_root = os.path.join(metadata_root, args.split)
    mask_root = configure_mask_root(args, tags=tags)
    log_folder = os.path.join(args.log_folder, args.experiment_name)
    scoremap_root = os.path.join(log_folder, args.scoremap_folder)
    xai_root = os.path.join(log_folder, args.xai_folder)
    configure_bbox_metric(args)

    evaluate_wsol(xai_root=xai_root,
                  scoremap_root=scoremap_root,
                  data_root=data_root,
                  metadata_root=metadata_root,
                  mask_root=mask_root,
                  dataset_name=args.dataset_name,
                  split=args.split,
                  cam_curve_interval=args.cam_curve_interval,
                  multi_contour_eval=args.multi_contour_eval,
                  multi_gt_eval=args.multi_gt_eval,
                  iou_threshold_list=args.iou_threshold_list,
                  tags=tags,
                  xai=args.xai)


if __name__ == "__main__":
    main()
