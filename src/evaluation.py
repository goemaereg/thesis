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
import tqdm
import mlflow
import matplotlib.pyplot as plt
from config import str2bool, configure_bbox_metric, configure_mask_root
from data_loaders import configure_metadata
from data_loaders import get_image_ids
from data_loaders import get_bounding_boxes, get_bounding_boxes_from_file
from data_loaders import get_image_sizes
from data_loaders import get_mask_paths
from data_loaders import get_cam_loader, get_cam_lmdb_loader
from data_loaders import _CAT_IMAGE_MEAN_STD
from util import check_scoremap_validity
from util import check_box_convention
from util import t2n
from sklearn.metrics import ConfusionMatrixDisplay

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

def is_point_in_box(point, box):
    """
    Arguments:
        point {list} -- list of float values (x,y)
        bbox {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
    Returns:
        {boolean} -- true if the point is inside the bbox
    """
    return point[0] >= box[0] and point[0] <= box[2] and point[1] >= box[1] and point[1] <= box[3]

def is_bbox_in_bbox(boxa, boxb):
    return (is_point_in_box(boxa[:2], boxb) and is_point_in_box(boxa[2:], boxb)) or \
           (is_point_in_box(boxb[:2], boxa) and is_point_in_box(boxb[2:], boxa))

def intersecting(boxa, boxb):
    # num_a x num_b
    min_x = np.maximum(boxa[0], boxb[0])
    min_y = np.maximum(boxa[1], boxb[1])
    max_x = np.minimum(boxa[2], boxb[2])
    max_y = np.minimum(boxa[3], boxb[3])
    area_intersect = (np.maximum(0, max_x - min_x + 1) * np.maximum(0, max_y - min_y + 1))
    return area_intersect > 0

def intersecting_bboxes_indices(bboxes):
    num_boxes = len(bboxes)
    intersecting_indices = []
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            if intersecting(bboxes[i], bboxes[j]):
                intersecting_indices.append((i, j))
    return intersecting_indices

def scoremap2bbox(threshold, scoremap_image, width, height):
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
        return np.asarray([[0, 0, 0, 0]]), 1, np.asarray([0])

    areas = [cv2.contourArea(c) for c in contours]
    max_contour_index = np.argmax(areas)
    # if not multi_contour_eval:
    #     # We only evaluate the tightest box around the largest-area connected component of the thresholded cam
    #     contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours), np.asarray(areas)

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

    thresh_boxes = []
    thresh_boxes_num = []
    thresh_boxes_areas = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box, areas = scoremap2bbox(threshold, scoremap_image, width, height)
        thresh_boxes.append(boxes)
        thresh_boxes_num.append(number_of_box)
        thresh_boxes_areas.append(areas)
    return thresh_boxes, thresh_boxes_num, thresh_boxes_areas


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metadata, dataset_name, split, cam_threshold_list,
                 iou_threshold_list, mask_root, multi_contour_eval, multi_gt_eval=False,
                 bbox_merge_strategy='add', bbox_merge_iou_threshold=0.2, log=False):
        self.log=log
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.mask_root = mask_root
        self.multi_gt_eval = multi_gt_eval
        self.multi_contour_eval = multi_contour_eval
        self.bbox_merge_strategy = bbox_merge_strategy
        self.bbox_merge_iou_threshold = bbox_merge_iou_threshold

    def reset(self):
        raise NotImplementedError

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def compute_optimal_cam_threshold(self, iou_threshold):
        raise NotImplementedError

class BoxCounters():
    def __init__(self, iou_threshold_list, cam_threshold_list, num_gt_bboxes):
        self.iou_threshold_list = iou_threshold_list
        self.cam_threshold_list = cam_threshold_list
        self.num_gt_bboxes = num_gt_bboxes
        self.reset()

    def reset(self):
        # num_targets to predict: depends on used metric
        # MaxBoxAcc and MaxBoxAccV2: total number of images in dataset (sufficient to match 1 GT bounding box)
        # MaxBoxAccV3: total number of GT bounding boxes in dataset
        self.num_targets = 0
        self.num_correct = {iou_threshold: np.zeros(shape=len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.num_estimated = np.zeros(shape=len(self.cam_threshold_list))

    def add_correct(self, iou_threshold, count_per_threshold):
        self.num_correct[iou_threshold] += count_per_threshold

    def add_estimated(self, count_per_threshold):
        self.num_estimated += count_per_threshold

    def add_target(self, count):
        self.num_targets += count

    def get_box_accuracy(self, iou_threshold):
        return self.num_correct[iou_threshold] * 1. / float(self.num_targets)

def f1_score(precision, recall):
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def precision_recall_f1(confusion_matrix):
    _, fp, fn, tp = confusion_matrix.ravel()
    precision = float(tp) / float(tp + fp) if (tp + fp > 0) else 0.0
    recall = float(tp) / float(tp + fn)
    return precision, recall, f1_score(precision, recall)

class BoxEvaluator(LocalizationEvaluator):
    """
    A class used to evaluate location accuracy using bounding boxes.

    The different metrics supported are:

    MaxBoxxAcc:

        Counts number of images in a dataset where tightest box around largest-area connected component of thresholded
        cam mask, matches with the GT bounding box. When more than 1 GT bounding box is provided, we count
        the number of images where the extimated box matches with at least one of the GT bounding boxes.
        Boxes match when IOU(est bb, GT bb) > IOU_Threshold.
        MaxBoxAcc is called GT-known localization accuracy for IOU_Threshold = 0.5

    MaxBoxAccV2:

        Counts number of images in a dataset where there is a best match between the set of estimated bounding boxes in
        the thresholded cam mask and the set of GT bounding boxes.
        Boxes match when IOU(est bb, GT bb) > IOU_Threshold.
        MaxBoxAccV2 averages the performance over IOU thresholds {0.3, 0.5, 0.7} to address different
        demands for localizaton fineness.

    MaxBoxAccV3:

        Counts number of GT-boxes in a dataset where there is a best match between a GT-box and the set of
        estimated bounding boxes in a thresholded cam mask. An estimated bounding box can at most have a
        match with a single GT-box.
        Boxes match when IOU(est bb, GT bb) > IOU_Threshold.
        MaxBoxAccV3 averages the performance over IOU thresholds {0.3, 0.5, 0.7} to address different
        demands for localizaton fineness.
    """

    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)
        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)
        self.num_gt_bboxes = 0
        for image_id, bboxes in self.original_bboxes.items():
            self.num_gt_bboxes += len(bboxes)
        self.counters = { metric: BoxCounters(iou_threshold_list=self.iou_threshold_list,
                                              cam_threshold_list=self.cam_threshold_list,
                                              num_gt_bboxes=self.num_gt_bboxes)
            for metric in ['MaxBoxAcc', 'MaxBoxAccV2', 'MaxBoxAccV3'] }
        self.reset()

    def reset(self):
        for counter in self.counters.values():
            counter.reset()

    def confusion_matrix(self, metric, iou_threshold, cam_threshold):
        counter = self.counters[metric]
        cam_threshold_index = self.cam_threshold_list.index(cam_threshold)
        tp = counter.num_correct[iou_threshold][cam_threshold_index]
        fp = counter.num_estimated[cam_threshold_index] - tp
        fn = counter.num_targets - tp
        tn = 0
        return np.reshape(np.asarray((tn, fp, fn, tp)), newshape=(2,2))

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    @staticmethod
    def unify(boxa, boxb):
        a_x0, a_y0, a_x1, a_y1 = boxa
        b_x0, b_y0, b_x1, b_y1 = boxb
        x0 = min(a_x0, b_x0)
        y0 = min(a_y0, b_y0)
        x1 = max(a_x1, b_x1)
        y1 = max(a_y1, b_y1)
        return np.asarray([x0, y0, x1, y1])

    def accumulate_bboxes(self, scoremap, image_id, context=None):
        # Computes a set of estimated boxes per scoremap threshold
        # Returns an array of threshold-related np.array objects containing estimated boxes
        # boxes_at_thresholds: list of 100 arrays, one array for each threshold. array.shape = (nr of estimated boxes)
        # thresh_boxes: List[np.ndarray(est_boxes_thresh0, 4), np.ndarray(est_boxes_thresh1, 4), ..., np.ndarray(est_boxes_thresh99, 4)]
        # thresh_boxes_num: List[int, int, ..., int]
        # thresh_boxes_areas: List[float, float, ..., float]
        thresh_boxes, thresh_boxes_num, thresh_boxes_areas = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)

        # for i in range(40, len(thresh_boxes)): # start checking from threshold=0.4
        #     # check for overlapping boxes
        #     intersect_list = intersecting_bboxes_indices(thresh_boxes[i])
        #     if len(intersect_list) > 0:
        #         print(f'{image_id}: Intersecting bboxes at threshold {i/100}: {intersect_list}')

        context_delta = dict(image_id=image_id,
                             thresh_boxes=thresh_boxes,
                             thresh_boxes_num=thresh_boxes_num,
                             thresh_boxes_areas=thresh_boxes_areas)

        if context is None or context['image_id'] != image_id:
            context = context_delta
        else:
            for i in range(len(thresh_boxes)):
                if self.bbox_merge_strategy == 'add':
                    context['thresh_boxes'][i] = np.concatenate([context['thresh_boxes'][i],
                                                                 thresh_boxes[i]], axis=0)
                    context['thresh_boxes_areas'][i] = np.concatenate([context['thresh_boxes_areas'][i],
                                                                       thresh_boxes_areas[i]], axis=0)
                    context['thresh_boxes_num'][i] += thresh_boxes_num[i]
                else:
                    # compute IOU
                    # multiple_iou shape = (est_boxes_iter_accumulated, est_boxes_iter_now)
                    multiple_iou = calculate_multiple_iou(
                        context['thresh_boxes'][i], thresh_boxes[i])
                    # loop over newly computed boxes
                    bboxes_index_add_list = []
                    bboxes_index_unify_list = []
                    bboxes_index_drop_list = []
                    for _ in range(thresh_boxes_num[i]):
                        # find largest IOU
                        max_iou_index = np.unravel_index(np.argmax(multiple_iou), shape=multiple_iou.shape)
                        max_iou = multiple_iou[max_iou_index]
                        boxa = context['thresh_boxes'][i][max_iou_index[0]]
                        boxb = thresh_boxes[i][max_iou_index[1]]
                        if self.bbox_merge_strategy == 'add':
                            bboxes_index_add_list.append(max_iou_index[1])
                        elif self.bbox_merge_strategy in ('drop', 'unify'):
                            if max_iou >= self.bbox_merge_iou_threshold or is_bbox_in_bbox(boxa, boxb):
                                if self.bbox_merge_strategy == 'drop':
                                    bboxes_index_drop_list.append(max_iou_index[1])
                                elif self.bbox_merge_strategy == 'unify':
                                    bboxes_index_unify_list.append(max_iou_index)
                            else:
                                bboxes_index_add_list.append(max_iou_index[1])
                        # mark newly estimated bbox with max IOU as unavailable to other combinations
                        multiple_iou[max_iou_index[0], :] = -1
                        multiple_iou[:, max_iou_index[1]] = -1
                    # unify
                    for index_to, index_from in bboxes_index_unify_list:
                        context['thresh_boxes'][i][index_to] = self.unify(
                            context['thresh_boxes'][i][index_to],
                            thresh_boxes[i][index_from]
                        )
                        # adding overlapping areas number comes with error (intersection counted twice)
                        context['thresh_boxes_areas'][i][index_to] += thresh_boxes_areas[i][index_from]
                    # add
                    context['thresh_boxes'][i] = np.concatenate(
                        [context['thresh_boxes'][i], thresh_boxes[i][bboxes_index_add_list]], axis=0)
                    context['thresh_boxes_areas'][i] = np.concatenate(
                        [context['thresh_boxes_areas'][i], thresh_boxes_areas[i][bboxes_index_add_list]], axis=0)
                    context['thresh_boxes_num'][i] += len(bboxes_index_add_list)
                    # drop: just ignore boxes in bboxes_index_drop_list
        return context, context_delta

    def _accumulate_maxboxacc_v1_2(self, multiple_iou, number_of_box_list, metric):
        # Computes single best match (1 box) over sets of estimated and GT-boxes per cam threshold
        # Result is 1 match value (0 or 1) per cam threshold
        """
        Computes best match = max(iou(est bounding boxes, GT bounding boxes)) per cam threshold
        Result per IOU threshold is 1 IOU value per scoremap threshold

        Args:
            multiple_iou: numpy.ndarray(dtype=np.float,
                          shape=(num estimated boxes over all scoremap thresholds, num GT-boxes))
        """
        idx = 0
        sliced_multiple_iou = [] # best matching estimated bounding box with single GT box per cam threshold
        num_thresholds = len(self.cam_threshold_list)
        for nr_box in number_of_box_list:
            # nr_box: number of estimated bounding boxes per cam threshold
            # (*) multiple_iou.max(1): best match between an estimated bounding box and set of GT bounding boxes
            # max(*)[idx: idx+nr_box]: best match over all estimated bounding boxes for a specific cam threshold
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            idx += nr_box
        # Compute true positives over different IOU thresholds
        for IOU_THRESHOLD in self.iou_threshold_list:
            num_correct = np.zeros(shape=num_thresholds)
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (IOU_THRESHOLD / 100))[0]
            num_correct[correct_threshold_indices] = 1
            self.counters[metric].add_correct(IOU_THRESHOLD, num_correct)
        self.counters[metric].add_estimated(np.asarray(number_of_box_list))
        self.counters[metric].add_target(1)

    def _accumulate_maxboxacc_v3(self, multiple_iou, number_of_box_list):
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
        # iterate over CAM thresholds
        for nr_box in number_of_box_list:
            gt_iou_max = []
            # sliced_multi_iou: IOU(estimated bounding boxes for current threshold, GT bounding boxes)
            slice_multi_iou = multiple_iou[idx : idx + nr_box].copy()
            # iterate over GT bounding boxes
            for _ in range(num_gt_boxes):
                max_iou_index = np.unravel_index(np.argmax(slice_multi_iou), shape=slice_multi_iou.shape)
                gt_iou_max.append(slice_multi_iou[max_iou_index[0], max_iou_index[1]])
                # mark max IOU as unavailable ('0') to other (est, gt) combinations in this slice
                slice_multi_iou[max_iou_index[0], :] = -1
                slice_multi_iou[:, max_iou_index[1]] = -1
            sliced_multiple_iou.append(np.asarray(gt_iou_max))
            idx += nr_box
        multi_iou_per_threshold = np.asarray(sliced_multiple_iou)

        # Compute true positives over different IOU thresholds
        for IOU_THRESHOLD in self.iou_threshold_list:
            num_correct_multi = np.zeros(shape=(num_thresholds, num_gt_boxes))
            correct_threshold_indices = np.where(multi_iou_per_threshold >= (IOU_THRESHOLD / 100))
            num_correct_multi[correct_threshold_indices] = 1
            # reduce to a single score per threshold = true positives (matching GT boxes) per threshold
            num_correct = np.sum(num_correct_multi, axis=1)
            self.counters['MaxBoxAccV3'].add_correct(IOU_THRESHOLD, num_correct)
        self.counters['MaxBoxAccV3'].add_estimated(np.asarray(number_of_box_list))
        self.counters['MaxBoxAccV3'].add_target(num_gt_boxes)

    def accumulate_boxacc(self, context):
        image_id = context['image_id']
        boxes_at_thresholds = context['thresh_boxes']
        number_of_box_list = context['thresh_boxes_num']
        max_contour_index_list = [np.argmax(context['thresh_boxes_areas'][i]) for i in range(len(boxes_at_thresholds))]

        # extract single box at thresholds for maxboxacc metric
        boxes_at_thresholds_v1 = []
        for boxes, max_contour_index in zip(boxes_at_thresholds, max_contour_index_list):
            boxes_at_thresholds_v1.append(boxes[max_contour_index].reshape((1, -1)))
        boxes_at_thresholds_v1 = np.concatenate(boxes_at_thresholds_v1, axis=0)
        number_of_box_list_v1 = [1] * len(max_contour_index_list)

        # concatenates sets of boxes per threshold into a single array of boxes over all thresholds
        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        # Computes IOU of all combinations of sets of estimated and set of ground-truth boxes
        # result: array.shape(len([est boxes for threshold 0..99]), nr of GT bounding boxes)
        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))

        multiple_iou_v1 = calculate_multiple_iou(
            np.array(boxes_at_thresholds_v1),
            np.array(self.gt_bboxes[image_id]))

        self._accumulate_maxboxacc_v1_2(multiple_iou_v1, number_of_box_list_v1, metric='MaxBoxAcc')
        self._accumulate_maxboxacc_v1_2(multiple_iou, number_of_box_list, metric='MaxBoxAccV2')
        self._accumulate_maxboxacc_v3(multiple_iou, number_of_box_list)
        return context

    def accumulate(self, scoremap, image_id, context=None):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        context = self.accumulate_bboxes(scoremap, image_id, context=None)
        return self.accumulate_boxacc(context)

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        metrics = {}
        for metric, counters in self.counters.items():
            box_acc_iou = {}
            precision_iou = []
            recall_iou = []
            f1_iou = []
            max_box_acc_iou = []
            max_box_acc_threshold = []
            for IOU_THRESHOLD in self.iou_threshold_list:
                box_acc_iou[IOU_THRESHOLD] = counters.get_box_accuracy(IOU_THRESHOLD)
                max_box_acc_iou.append(box_acc_iou[IOU_THRESHOLD].max())
                cam_threshold_optimal_index = box_acc_iou[IOU_THRESHOLD].argmax()
                cam_threshold_optimal = self.cam_threshold_list[cam_threshold_optimal_index]
                max_box_acc_threshold.append(cam_threshold_optimal)
                cm = self.confusion_matrix(metric, iou_threshold=IOU_THRESHOLD, cam_threshold=cam_threshold_optimal)
                precision, recall, f1 = precision_recall_f1(cm)
                precision_iou.append(precision)
                recall_iou.append(recall)
                f1_iou.append(f1)
                metrics |= {
                    f'{metric}_precision_IOU_{IOU_THRESHOLD}': precision,
                    f'{metric}_recall_IOU_{IOU_THRESHOLD}': recall,
                    f'{metric}_f1_IOU_{IOU_THRESHOLD}': f1
                }
                if self.log:
                    # plot confusion matrix
                    disp = ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues)
                    log_path = f'plots/{self.split}/{metric}_confusion_matrix_{IOU_THRESHOLD}.png'
                    if not os.path.exists(os.path.dirname(log_path)):
                        os.makedirs(os.path.dirname(log_path))
                    mlflow.log_figure(disp.figure_, log_path)
                    plt.close('all')
                    # log confusion matrix
                    tn, fp, fn, tp = cm.ravel()
                    cm_dict = dict(tn=tn, fp=fp, fn=fn, tp=tp)
                    log_path = f'data/{self.split}/{metric}_confusion_matrix_{IOU_THRESHOLD}.json'
                    if not os.path.exists(os.path.dirname(log_path)):
                        os.makedirs(os.path.dirname(log_path))
                    mlflow.log_dict(cm_dict, log_path)
                    # log box accuracy data
                    box_acc = {
                        'iou_threshold': IOU_THRESHOLD,
                        'cam_threshold_optimal': cam_threshold_optimal,
                        'cam_threshold': self.cam_threshold_list,
                        'box_accuracy': box_acc_iou[IOU_THRESHOLD].tolist()
                    }
                    log_path = f'data/{self.split}/{metric}_box_acc_iou_{IOU_THRESHOLD}.json'
                    if not os.path.exists(os.path.dirname(log_path)):
                        os.makedirs(os.path.dirname(log_path))
                    mlflow.log_dict(box_acc, log_path)
                    # plot BoxAcc(iou_threshold)
                    fig, ax = plt.subplots()
                    ax.plot(self.cam_threshold_list, box_acc_iou[IOU_THRESHOLD])
                    plt.title(f'{self.dataset_name} BoxAcc IOU={IOU_THRESHOLD}')
                    plt.xlabel('CAM threshold')
                    plt.ylabel('BoxAcc')
                    plt.axis('tight')
                    log_path = f'plots/{self.split}/{metric}_box_acc_iou_{IOU_THRESHOLD}.png'
                    if not os.path.exists(os.path.dirname(log_path)):
                        os.makedirs(os.path.dirname(log_path))
                    mlflow.log_figure(fig, log_path)
                    plt.close('all')

            if metric == 'MaxBoxAcc':
                index_iou_50 = self.iou_threshold_list.index(50)
                metrics |= {
                    metric: max_box_acc_iou[index_iou_50],
                    f'{metric}_precision': precision_iou[index_iou_50],
                    f'{metric}_recall': recall_iou[index_iou_50],
                    f'{metric}_f1': f1_iou[index_iou_50]
                }
                metrics |= {f'{metric}_optimal_threshold': max_box_acc_threshold[index_iou_50]}
            else:
                precision = np.average(precision_iou)
                recall = np.average(recall_iou)
                f1 = f1_score(precision, recall)
                metrics |= {
                    metric: np.average(max_box_acc_iou),
                    f'{metric}_precision': precision,
                    f'{metric}_recall': recall,
                    f'{metric}_f1': f1
                }
            for index, IOU_THRESHOLD in enumerate(self.iou_threshold_list):
                metrics |= {f'{metric}_IOU_{IOU_THRESHOLD}': max_box_acc_iou[index]}
                metrics |= {f'{metric}_optimal_threshold_IOU_{IOU_THRESHOLD}': max_box_acc_threshold[index]}

        return metrics

    def compute_optimal_cam_threshold(self, iou_threshold):
        """
        Returns:
            optimal cam threshold t = max<t> BoxAcc(t, iou_threshold)
            index of optimal threshold in list of cam thresholds
        """
        counters = self.counters['MaxBoxAccV3']
        box_acc_iou = counters.get_box_accuracy(iou_threshold)
        optimal_threshold_index = box_acc_iou.argmax()
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
        super(MaskEvaluator, self).__init__(**kwargs)

        # if self.dataset_name != "OpenImages":
        #     raise ValueError("Mask evaluation must be performed on OpenImages.")
        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,[1.0])#, 2.0, 3.0])
        self.num_bins = len(self.threshold_list_right_edge) - 1
        self.reset()

    def reset(self):
        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=float)

    def accumulate(self, scoremap, image_id, context=None):
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
            plt.close('all')

        return {'PxAP': auc}


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

def scale_bounding_boxes(bboxes_dict, image_sizes, orig_shape):
    resized_bboxes = { image_id: [
                resize_bbox(bbox, orig_shape, image_sizes[image_id])
            for bbox in bboxes ]
        for image_id, bboxes in bboxes_dict.items() }
    return resized_bboxes

def xai_save_cam_inference(xai_root, metadata, data_root, split, image_id, image, size_orig, cam, iter_index=None, log=False):
    # get dataset mean and std
    mean_std = _CAT_IMAGE_MEAN_STD[metadata.root]
    mean = mean_std['mean']
    std = mean_std['std']
    # image size: 224x224, cam size: 224x224
    # scale image and cam to original size
    # convert cam to grey image
    cam = (cam * 255).astype('uint8')
    # convert to heatmap in BGR format
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    # resize to original image size
    cam = cv2.resize(cam, size_orig, interpolation=cv2.INTER_CUBIC)
    # load original image
    path_img = os.path.join(data_root, image_id)
    image_orig = cv2.imread(path_img)  # color channels in BGR format
    # de-normalize image
    for i, image_color in enumerate(image):
        image[i, :, :] = std[i] * image_color + mean[i]
    # scale to 255 range
    image = (image * 255).astype('uint8')
    # transform image into cv shape: (C, H, W) -> (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # transform RGB to BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # resize to original image size
    image = cv2.resize(image, size_orig, interpolation=cv2.INTER_CUBIC)
    # image path naming
    img_base_id = os.path.basename(image_id).split(".")[0]
    suffix = f'_{str(iter_index)}' if iter_index is not None else ''
    img_id = f'{img_base_id}_img{suffix}.png'
    img_cam_id = f'{img_base_id}_cam{suffix}.png'
    img_path = os.path.join(xai_root, split, os.path.basename(img_id))
    img_cam_path = os.path.join(xai_root, split, os.path.basename(img_cam_id))
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))
    if not os.path.exists(os.path.dirname(img_cam_path)):
        os.makedirs(os.path.dirname(img_cam_path))
    # merge cam heatmap and image
    image_cam = cam * 0.5 + image_orig * 0.5
    # save image
    cv2.imwrite(img_path, image)
    cv2.imwrite(img_cam_path, image_cam)
    if log:
        mlflow.log_artifact(img_path, f'xai/{split}')
        mlflow.log_artifact(img_cam_path, f'xai/{split}')

def xai_save_cams(xai_root, split, image_ids, metadata, data_root, scoremap_root, log=False, images_max=100, iter_index=None):
    optimal_thresholds_path = os.path.join(scoremap_root, split, 'optimal_thresholds.npy')
    thresholds = np.load(optimal_thresholds_path)
    optimal_threshold = thresholds[-1]
    image_sizes = get_image_sizes(metadata)
    gt_bbox_dict = get_bounding_boxes(metadata)
    est_bbox_dict = get_bounding_boxes_from_file(os.path.join(scoremap_root, split, 'bboxes_metadata.txt'))
    est_bbox_dict = scale_bounding_boxes(est_bbox_dict, image_sizes, (224,224))
    est_bbox_delta_dict = get_bounding_boxes_from_file(os.path.join(scoremap_root, split, 'bboxes_delta_metadata.txt'))
    est_bbox_delta_dict = scale_bounding_boxes(est_bbox_delta_dict, image_sizes, (224,224))
    cam_loader = get_cam_lmdb_loader(scoremap_root, image_ids, split)
    images_num = 0
    color_red = (0, 0, 255)  # BGR
    color_green = (0, 255, 0)  # BGR
    tq0 = tqdm.tqdm(cam_loader, total=len(cam_loader), desc='xai_cam_batches')
    for cams, cams_delta, img_ids in tq0:
        cams = t2n(cams)
        cams_delta = t2n(cams_delta)
        cams_it = zip(cams, cams_delta, img_ids)
        for cam_merged, cam_delta, image_id in cams_it:
            images_num += 1
            if images_num > images_max:
                return

            # render image overlayed with CAM heatmap
            path_img = os.path.join(data_root, image_id)
            # load image
            img = cv2.imread(path_img)  # color channels in BGR format
            orig_img_shape = image_sizes[image_id]
            # estimated and GT bboxes overlay
            gt_bbox_list = gt_bbox_dict[image_id]

            est_bbox_merged_list = est_bbox_dict[image_id]
            est_bbox_delta_list = est_bbox_delta_dict[image_id]

            items = [(cam_merged, est_bbox_merged_list, ''), (cam_delta, est_bbox_delta_list, '_delta')]
            for cam, est_bbox_list, marker in items:
                # resize cam from 224x224 to original image size
                cam = cv2.resize(cam, orig_img_shape, interpolation=cv2.INTER_CUBIC)
                # normalize
                cam = normalize_scoremap(cam)
                # assign minimal value to area outside segment mask so normalization is constrained to segment values
                cam_heatmap = cam.copy()
                # transform to grey image
                cam_grey = (cam_heatmap * 255).astype('uint8')
                heatmap = cv2.applyColorMap(cam_grey, cv2.COLORMAP_JET)
                img_ann = heatmap * 0.5 + img * 0.5

                # CAM segment mask
                segment = np.zeros(shape=img.shape, dtype=np.uint8)
                # mask cam above optimal threshold
                cam_mask = cam >= optimal_threshold
                segment[cam_mask] = color_red
                img_seg = segment * 0.5 + img * 0.5

                if (len(gt_bbox_list) + len(est_bbox_list)) > 0:
                    thickness = 2  # Pixels
                    for bbox in gt_bbox_list:
                        start, end = bbox[:2], bbox[2:]
                        img_ann = cv2.rectangle(img_ann, start, end, color_green, thickness)
                        img_seg = cv2.rectangle(img_seg, start, end, color_green, thickness)
                    for bbox in est_bbox_list:
                        start, end = bbox[:2], bbox[2:]
                        img_ann = cv2.rectangle(img_ann, start, end, color_red, thickness)
                        img_seg = cv2.rectangle(img_seg, start, end, color_red, thickness)

                # image path naming
                suffix = f'{marker}_{str(iter_index)}' if iter_index is not None else marker
                img_base_id = os.path.basename(image_id).split(".")[0]
                img_ann_id = f'{img_base_id}_ann{suffix}.png'
                img_seg_id = f'{img_base_id}_seg{suffix}.png'
                img_ann_path = os.path.join(xai_root, split, os.path.basename(img_ann_id))
                img_seg_path = os.path.join(xai_root, split, os.path.basename(img_seg_id))
                if not os.path.exists(os.path.dirname(img_ann_path)):
                    os.makedirs(os.path.dirname(img_ann_path))
                if not os.path.exists(os.path.dirname(img_seg_path)):
                    os.makedirs(os.path.dirname(img_seg_path))
                cv2.imwrite(img_ann_path, img_ann)
                cv2.imwrite(img_seg_path, img_seg)

                if log:
                    mlflow.log_artifact(img_ann_path, f'xai/{split}')
                    mlflow.log_artifact(img_seg_path, f'xai/{split}')

def get_evaluators(**args):
    dataset_name = args.get('dataset_name', 'SYNTHETIC')
    return {
        "ILSVRC": (BoxEvaluator(**args), None),
        "SYNTHETIC": (BoxEvaluator(**args), MaskEvaluator(**args))
    }[dataset_name]

def evaluate_wsol(xai_root, scoremap_root, data_root, metadata_root, mask_root,
                  iou_threshold_list, dataset_name, split,
                  multi_contour_eval, multi_gt_eval, cam_curve_interval=.01,
                  xai=False):
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
    print("Loading and evaluating cams.")
    metadata = configure_metadata(metadata_root)
    est_bboxes = get_bounding_boxes_from_file(os.path.join(scoremap_root, split, 'bboxes_metadata.txt'))
    optimal_thresholds_path = os.path.join(scoremap_root, split, 'optimal_thresholds.npy')
    thresholds = np.load(optimal_thresholds_path)
    optimal_threshold = thresholds[-1]
    # The length of the output of np.arange might not be numerically stable.
    # Better to use np.linspace
    # cam_threshold_list = np.linspace(0, 1, num=int(1/cam_curve_interval),
    #                                  endpoint=False).tolist()

    eval_args = dict(
        metadata=metadata,
        dataset_name=dataset_name,
        split=split,
        cam_threshold_list=[optimal_threshold],
        iou_threshold_list=iou_threshold_list,
        mask_root=mask_root,
        multi_contour_eval=multi_contour_eval,
        multi_gt_eval=multi_gt_eval,
        log=False)

    box_evaluator, mask_evaluator = get_evaluators(**eval_args)
    cam_loader = get_cam_loader(scoremap_root, split)
    for cams, image_ids in cam_loader:
        for cam, image_id in zip(cams, image_ids):
            cam = t2n(cam)
            if box_evaluator:
                bboxes = np.asarray(est_bboxes[image_id])
                num_bboxes = len(bboxes)
                # fake estimate: assume bbox area represents coutour area
                areas = np.asarray([(b[2]-b[0])*[b[3]-b[1]] for b in bboxes])
                context = dict(image_id=image_id,
                               thresh_boxes=[bboxes],
                               thresh_boxes_num=[num_bboxes],
                               thresh_boxes_areas=[areas])
                box_evaluator.accumulate_boxacc(context)
            if mask_evaluator:
                # merge scoremaps of different iterations
                mask_evaluator.accumulate(cam, image_id)
    metrics = {}
    if box_evaluator:
        metrics |= box_evaluator.compute()
    if mask_evaluator:
        metrics |= mask_evaluator.compute()
    for metric, value in metrics.items():
        print(f'{metric}: {value}')
    if xai is False:
        return
    # XAI
    xai_save_cams(xai_root=xai_root, split=split, metadata=metadata, data_root=data_root,
                  scoremap_root=scoremap_root, log=False)


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
                  xai=args.xai)


if __name__ == "__main__":
    main()
