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

import munch
import numpy as np
import tarfile
import lmdb
import pickle
import os
import io
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Sampler
from typing import Iterator, List

_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]

# _LMDB_MAPSIZE = 34359738368 # 32 GB
_LMDB_MAPSIZE = 68719476736 # 64 GB

_CAT_IMAGE_MEAN_STD = {
'data/metadata/ILSVRC/train': {'mean': _IMAGE_MEAN_VALUE, 'std': _IMAGE_STD_VALUE},
'data/metadata/ILSVRC/val': {'mean': _IMAGE_MEAN_VALUE, 'std': _IMAGE_STD_VALUE},
'data/metadata/ILSVRC/test': {'mean': _IMAGE_MEAN_VALUE, 'std': _IMAGE_STD_VALUE},
}
_CAT_IMAGE_MEAN_STD |= {'data/metadata/SYNTHETIC/d_1_t/train': {'mean': [0.04273996502161026, 0.04311680421233177, 0.05591028183698654], 'std': [0.14303097128868103, 0.14330065250396729, 0.16584224998950958]}, 'data/metadata/SYNTHETIC/d_1_t/val': {'mean': [0.04410117119550705, 0.043009135872125626, 0.05730963125824928], 'std': [0.14504630863666534, 0.1432878077030182, 0.16645550727844238]}, 'data/metadata/SYNTHETIC/d_1_t/test': {'mean': [0.04006754979491234, 0.040725477039813995, 0.05305006355047226], 'std': [0.13640396296977997, 0.1406128704547882, 0.16055920720100403]}, 'data/metadata/SYNTHETIC/d_1_b/train': {'mean': [0.24238689243793488, 0.24167926609516144, 0.2536686956882477], 'std': [0.22504417598247528, 0.22468362748622894, 0.22636130452156067]}, 'data/metadata/SYNTHETIC/d_1_b/val': {'mean': [0.2390381395816803, 0.24038280546665192, 0.2509518563747406], 'std': [0.2209886610507965, 0.22130391001701355, 0.22285866737365723]}, 'data/metadata/SYNTHETIC/d_1_b/test': {'mean': [0.23923085629940033, 0.23854359984397888, 0.24699807167053223], 'std': [0.22572922706604004, 0.22539030015468597, 0.2264942079782486]}, 'data/metadata/SYNTHETIC/d_2_t/train': {'mean': [0.0215207040309906, 0.021198157221078873, 0.02777492254972458], 'std': [0.10190754383802414, 0.10040006786584854, 0.1182713732123375]}, 'data/metadata/SYNTHETIC/d_2_t/val': {'mean': [0.021659355610609055, 0.021440880373120308, 0.029011119157075882], 'std': [0.10243792831897736, 0.10145081579685211, 0.12226879596710205]}, 'data/metadata/SYNTHETIC/d_2_t/test': {'mean': [0.020800502970814705, 0.02161002717912197, 0.027953317388892174], 'std': [0.09884179383516312, 0.10199198126792908, 0.11878983676433563]}, 'data/metadata/SYNTHETIC/d_2_b/train': {'mean': [0.2285696119070053, 0.22865375876426697, 0.2344420999288559], 'std': [0.21984994411468506, 0.21931613981723785, 0.22186172008514404]}, 'data/metadata/SYNTHETIC/d_2_b/val': {'mean': [0.2182750403881073, 0.21741211414337158, 0.22465910017490387], 'std': [0.21942591667175293, 0.21870020031929016, 0.2219187319278717]}, 'data/metadata/SYNTHETIC/d_2_b/test': {'mean': [0.2445860058069229, 0.2432548701763153, 0.25054073333740234], 'std': [0.21880653500556946, 0.218492791056633, 0.22151072323322296]}, 'data/metadata/SYNTHETIC/d_3_t/train': {'mean': [0.03193404898047447, 0.03240562975406647, 0.041789256036281586], 'std': [0.1217082142829895, 0.12346523255109787, 0.14246729016304016]}, 'data/metadata/SYNTHETIC/d_3_t/val': {'mean': [0.03284938633441925, 0.032680120319128036, 0.04229772090911865], 'std': [0.1241430938243866, 0.12412027269601822, 0.14369632303714752]}, 'data/metadata/SYNTHETIC/d_3_t/test': {'mean': [0.03049885295331478, 0.03126080334186554, 0.039372991770505905], 'std': [0.119247667491436, 0.12156332284212112, 0.13802514970302582]}, 'data/metadata/SYNTHETIC/d_3_b/train': {'mean': [0.25665509700775146, 0.2563091516494751, 0.2662285268306732], 'std': [0.21810951828956604, 0.2176094800233841, 0.22042134404182434]}, 'data/metadata/SYNTHETIC/d_3_b/val': {'mean': [0.22744280099868774, 0.22860224545001984, 0.23734967410564423], 'std': [0.22192175686359406, 0.2221284955739975, 0.2242507040500641]}, 'data/metadata/SYNTHETIC/d_3_b/test': {'mean': [0.24295121431350708, 0.243118554353714, 0.25332996249198914], 'std': [0.22262203693389893, 0.22303451597690582, 0.2254263013601303]}, 'data/metadata/SYNTHETIC/d_4_t/train': {'mean': [0.042220667004585266, 0.04245072230696678, 0.055196069180965424], 'std': [0.13874958455562592, 0.139445498585701, 0.16160985827445984]}, 'data/metadata/SYNTHETIC/d_4_t/val': {'mean': [0.04591738432645798, 0.045563943684101105, 0.060396455228328705], 'std': [0.14541621506214142, 0.14355485141277313, 0.16986531019210815]}, 'data/metadata/SYNTHETIC/d_4_t/test': {'mean': [0.04439207911491394, 0.042193274945020676, 0.05604483559727669], 'std': [0.14307141304016113, 0.137443408370018, 0.1624082773923874]}, 'data/metadata/SYNTHETIC/d_4_b/train': {'mean': [0.24409303069114685, 0.2448749840259552, 0.25704532861709595], 'std': [0.22179637849330902, 0.2217387706041336, 0.22530175745487213]}, 'data/metadata/SYNTHETIC/d_4_b/val': {'mean': [0.24998736381530762, 0.25162938237190247, 0.2648410201072693], 'std': [0.21950455009937286, 0.22046080231666565, 0.22500573098659515]}, 'data/metadata/SYNTHETIC/d_4_b/test': {'mean': [0.25904580950737, 0.25815334916114807, 0.2719438970088959], 'std': [0.22231793403625488, 0.22278547286987305, 0.2270577996969223]}}

def mch(**kwargs):
    return munch.Munch(dict(**kwargs))

def configure_metadata(metadata_root):
    metadata = mch()
    metadata.root = metadata_root
    metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = os.path.join(metadata_root,
                                            'image_ids_proxy.txt')
    metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
    metadata.localization = os.path.join(metadata_root, 'localization.txt')
    metadata.segments = os.path.join(metadata_root, 'segments.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels

def get_bounding_boxes_from_file(path):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(path) as f:
        for line in f.readlines():
            image_id, x0s, y0s, x1s, y1s = line.strip('\n').split(',')
            x0, y0, x1, y1 = int(x0s), int(y0s), int(x1s), int(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, y0, x1, y1))
            else:
                boxes[image_id] = [(x0, y0, x1, y1)]
    return boxes


def get_bounding_boxes(metadata):
    return get_bounding_boxes_from_file(metadata.localization)


def get_mask_paths(metadata):
    """
    segments.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(metadata.segments) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                # ignore mask is optional (empty string)
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


class WSOLImageLabelDataset(Dataset):
    def __init__(self, data_root, metadata_root, transform, normalize, proxy, num_sample_per_class=0,
                 bboxes_path=None, bbox_mask_strategy=None, mask_method=None,
                 scoremap_lmdb_path=None, filter_instances=0):
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.normalize = normalize
        self.image_ids = get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class
        self._adjust_samples_per_class()
        self.bbox_mask_strategy = bbox_mask_strategy
        self.mask_method = mask_method
        self.computed_bboxes = {}
        mean_std = _CAT_IMAGE_MEAN_STD[metadata_root]
        self.dataset_mean = mean_std['mean']
        self.dataset_std = mean_std['std']
        self.num_channels = len(self.dataset_mean)
        if bboxes_path is not None and os.path.exists(bboxes_path):
            self.computed_bboxes = get_bounding_boxes_from_file(bboxes_path)
        self.filter_instance = filter_instances
        if filter_instances > 0:
            image_ids = []
            image_labels = {}
            bboxes_gt_dict = get_bounding_boxes(self.metadata)
            for image_id, bboxes in bboxes_gt_dict.items():
                instances = len(bboxes)
                if instances != filter_instances:
                    continue
                image_ids.append(image_id)
                image_labels[image_id] = self.image_labels[image_id]
            self.image_ids = image_ids
            self.image_labels = image_labels
        self.scoremap_lmdb_path = scoremap_lmdb_path
        if scoremap_lmdb_path is not None:
            self.scoremap_db = lmdb.open(self.scoremap_lmdb_path, subdir=os.path.isdir(self.scoremap_lmdb_path),
                                         map_size=_LMDB_MAPSIZE,
                                         readonly=True, lock=False,
                                         readahead=False, meminit=False)
        else:
            self.scoremap_db = None

    # Deleting (Calling destructor)
    def __del__(self):
        if self.scoremap_db is not None:
            self.scoremap_db.close()

    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    def mask_image_with_bboxes(self, image, image_id):
        # image masking strategy for iterative bounding box extraction
        if self.bbox_mask_strategy is not None and image_id in self.computed_bboxes:
            bboxes = self.computed_bboxes[image_id]
            for bbox in bboxes:
                # convert box(x0,y0,x1,y1) coordinates to array(i,j) semantics
                x0, y0, x1, y1 = bbox
                w = x1 - x0
                h = y1 - y0
                i, j = y0, x0
                v = image[..., i:i + h, j:j + w]
                if self.bbox_mask_strategy == 'zero':
                    # Erasing by filling black values. Has to be done before normalization.
                    v = torch.zeros(*v.shape)
                elif self.bbox_mask_strategy == 'mean':
                    # fill with image mean value
                    image_mean = torch.mean(image, dim=(1, 2), keepdim=True)
                    v = torch.tile(image_mean, dims=(1, h, w))
                elif self.bbox_mask_strategy == 'random':
                    # fill with random value from dataset (mean, std) before normalization
                    # first sample from standard normal distribution (mean=0, std=1)
                    v = torch.rand(*v.shape)
                    # Then scale to dataset distribution with (mean=<dataset.mean>, std=<dataset.std>)
                    for c in range(self.num_channels):
                        v[c, :, :] = self.dataset_std[c] * v[c, :, :] + self.dataset_mean[c]
                    # after this step, normalization will transform to standard normal distribution
                image[..., i:i + h, j:j + w] = v
        return image

    def mask_image_with_scoremap(self, image, image_id):
        if self.bbox_mask_strategy is None:
            return image
        # get score map
        with self.scoremap_db.begin(write=False) as txn:
            key = u'{}'.format(image_id).encode('ascii')
            raw = txn.get(key)
            unpacked = pickle.loads(raw)
            cam = unpacked[0]
        # mask image with score map: image x (1 - scoremap) (element-wise product)
        cam_threshold = np.mean(cam)
        mask = torch.tile(torch.tensor(cam, dtype=image.dtype) > cam_threshold, dims=(image.shape[0], 1, 1))
        v = image
        if self.bbox_mask_strategy == 'zero':
            v = torch.zeros(*v.shape)
        elif self.bbox_mask_strategy == 'mean':
            # fill with image mean value
            image_mean = torch.mean(image, dim=(1, 2), keepdim=True)
            v = torch.tile(image_mean, dims=(1, *v.shape[1:]))
        else:
            # fill with random value from dataset (mean, std) before normalization
            # first sample from standard normal distribution (mean=0, std=1)
            v = torch.rand(*v.shape)
            # Then scale to dataset distribution with (mean=<dataset.mean>, std=<dataset.std>)
            for c in range(self.num_channels):
                v[c, :, :] = self.dataset_std[c] * v[c, :, :] + self.dataset_mean[c]
            # after this step, normalization will transform to standard normal distribution
        image[mask] = v[mask]
        # image = image * (1.0 - torch.tensor(cam, dtype=image.dtype))
        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')
        image = self.transform(image)
        # image masking strategy for iterative bounding box extraction
        if self.mask_method is not None:
            if self.mask_method == 'bbox':
                image = self.mask_image_with_bboxes(image, image_id)
            elif self.mask_method == 'cam':
                image = self.mask_image_with_scoremap(image, image_id)
        # normalize image
        image = self.normalize(image)
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


class WSOLImageLabelTarDataset(Dataset):
    def __init__(self, tar_path, metadata_root, transform, normalize):
        self.tar_path = tar_path
        self.tar_file = tarfile.open(name=tar_path, mode='r')
        self.metadata = configure_metadata(metadata_root)
        self.image_ids = get_image_ids(self.metadata, proxy=False)
        self.image_labels = get_class_labels(self.metadata)
        self.transform = transform
        self.normalize = normalize

    def __getitem__(self, idx):
        # example image_id: train/n01440764/n01440764_10026.JPEG
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        id_parts = image_id.split('/')
        class_id = id_parts[1]
        image_filename = id_parts[2]
        # extract <class_id>.tar file
        class_tarfname = f'{class_id}.tar'
        tar = tarfile.open(fileobj=self.tar_file.extractfile(class_tarfname))
        buf = tar.extractfile(image_filename).read()
        tar.close()
        with io.BytesIO(buf) as bio:
            image = Image.open(bio).convert('RGB')
        image = self.transform(image)
        image = self.normalize(image)
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


class WSOLImageLabelLmdbDataset(Dataset):
    environments = dict()

    @classmethod
    def _init_db(cls, lmdb_path):
        if lmdb_path not in cls.environments:
            cls.environments[lmdb_path] = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                                                    readonly=True, lock=False,
                                                    readahead=False, meminit=False)
        env = cls.environments[lmdb_path]
        with env.begin(write=False) as txn:
            length = pickle.loads(txn.get(b'__len__'))
        return env, length

    def __init__(self, lmdb_path, metadata_root, transform, normalize):
        self.lmdb_path = lmdb_path
        self.env = None
        self.length = 1281168
        self.metadata = configure_metadata(metadata_root)
        self.image_labels = get_class_labels(self.metadata)
        self.image_ids = list(self.image_labels)
        self.transform = transform
        self.normalize = normalize

    def __getitem__(self, idx):
        if self.env is None:
            self.env, self.length = self._init_db(self.lmdb_path)
        image, target = None, None
        # example image_id: train/n01440764/n01440764_10026.JPEG
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        with self.env.begin(write=False) as txn:
            key = u'{}'.format(image_id).encode('ascii')
            byteflow = txn.get(key)
            image_buffer = pickle.loads(byteflow)
            with io.BytesIO(image_buffer) as bio:
                image = Image.open(bio).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            if self.normalize is not None:
                image = self.normalize(image)
            return image, image_label, image_id

    def __len__(self):
        return self.length


class MiniBatchSampler(Sampler[int]):
    def __init__(self, image_label_dataset,
                 class_set_size=5,
                 batch_set_size=12) -> None:
        super().__init__(image_label_dataset)
        self.dataset = image_label_dataset
        self.labels = np.asarray(list(self.dataset.image_labels.values()))
        self.unique_labels = np.sort(np.unique(self.labels))
        self.class_set_size = class_set_size
        self.batch_set_size = batch_set_size
        batch_size = self.class_set_size * self.batch_set_size
        self.batch_num = int((len(self.labels) + batch_size - 1) // batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.batch_num):
            label_set = np.random.permutation(
                self.unique_labels)[0:self.batch_set_size]
            index_list = []
            for c in label_set:
                index = np.flatnonzero(self.labels == c)
                index = index[np.random.permutation(len(index))]
                index_list += index[0:self.class_set_size].tolist()
            yield index_list

    def __len__(self) -> int:
        return self.batch_num

class CamDataset(Dataset):
    def __init__(self, scoremap_root, split):
        self.scoremap_root = scoremap_root
        self.split = split
        metadata = {}
        with open(os.path.join(scoremap_root, split, 'scoremap_metadata.txt'), 'r') as fp:
            for line in fp.readlines():
                image_id, cam_id = line.strip('\n').split(',')
                metadata[image_id] = cam_id
        metadata = dict(sorted(metadata.items()))
        self.image_ids = list(metadata.keys())
        self.cam_ids = list(metadata.values())
        self.length = len(self.image_ids)

    def _load_cam(self, cam_id):
        scoremap_path = os.path.join(self.scoremap_root, cam_id)
        return np.load(scoremap_path)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam_id = self.cam_ids[index]
        cam = self._load_cam(cam_id)
        return cam, image_id

    def __len__(self):
        return self.length


class CamLmdbDataset(Dataset):
    def __init__(self, lmdb_path, image_ids):
        super(CamLmdbDataset, self).__init__()
        self.lmdb_path = lmdb_path
        self.image_ids = image_ids
        self.length = len(image_ids)
        self.db = None

    # Deleting (Calling destructor)
    def __del__(self):
        if self.db is not None:
            self.db.close()

    def __getitem__(self, index):
        if self.db is None:
            self.db = lmdb.open(self.lmdb_path, subdir=os.path.isdir(self.lmdb_path),
                                map_size=_LMDB_MAPSIZE,
                                readonly=True, lock=False,
                                readahead=False, meminit=False)
        with self.db.begin(write=False) as txn:
            image_id = self.image_ids[index]
            key = u'{}'.format(image_id).encode('ascii')
            raw = txn.get(key)
            unpacked = pickle.loads(raw)
            cam = unpacked[0]
            cam_delta = unpacked[1]
            return cam, cam_delta, image_id

    def __len__(self):
        return self.length

def get_cam_loader(scoremap_path, split):
    return DataLoader(
        CamDataset(scoremap_path, split),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

def get_cam_lmdb_loader(scoremap_root, image_ids, split):
    lmdb_path = os.path.join(scoremap_root, split, 'lmdb_scoremaps.lmdb')
    return DataLoader(
        CamLmdbDataset(lmdb_path, image_ids),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True)


def get_eval_loader(split, data_root, metadata_root, batch_size, workers,
                    resize_size, bboxes_path=None, bbox_mask_strategy=None, mask_method=None,
                    scoremap_lmdb_path=None, filter_instances=0):
    metadata_root_split = os.path.join(metadata_root, split)
    mean_std = _CAT_IMAGE_MEAN_STD[metadata_root_split]
    mean = mean_std['mean']
    std = mean_std['std']
    dataset_transforms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor()])
    dataset_normalize = transforms.Normalize(mean, std)
    dataset = WSOLImageLabelDataset(
        data_root=data_root,
        metadata_root=metadata_root_split,
        transform=dataset_transforms,
        normalize=dataset_normalize,
        proxy=False,
        bboxes_path=bboxes_path,
        bbox_mask_strategy=bbox_mask_strategy,
        mask_method=mask_method,
        scoremap_lmdb_path=scoremap_lmdb_path,
        filter_instances=filter_instances)
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers)

def get_data_loader(splits, data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, proxy_training_set,
                    num_val_sample_per_class=0, batch_set_size=None,
                    class_set_size=None, train_augment=True, dataset_name='SYNTHETIC'):
    loaders = {}
    for split in splits:
        metadata_root_split = os.path.join(metadata_root, split)
        mean_std = _CAT_IMAGE_MEAN_STD[metadata_root_split]
        mean, std = mean_std['mean'], mean_std['std']
        dataset_normalize = transforms.Normalize(mean, std)
        if split == 'train' and train_augment is True:
            dataset_transforms = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            dataset_transforms = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor()])
        if split == 'train' and dataset_name == 'ILSVRC':
            dataset = WSOLImageLabelLmdbDataset(
                lmdb_path = os.path.join(data_roots[split], 'lmdb_train.lmdb'),
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms,
                normalize=dataset_normalize)
        else:
            dataset = WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms,
                normalize=dataset_normalize,
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0))
        # default case: if batch_size > 1 then automatic batch loading
        shuffle = split == 'train'
        batch_sampler = None
        _batch_size = batch_size
        if split == 'train':
            # MinMaxCAM case: customized MiniBatchSampler
            if batch_set_size is not None and class_set_size is not None:
                _batch_size = 1
                shuffle = None
                batch_sampler = MiniBatchSampler(
                    dataset,
                    batch_set_size=batch_set_size,
                    class_set_size=class_set_size)
        loaders[split] = DataLoader(
            dataset,
            batch_size=_batch_size,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            num_workers=workers)
    return loaders
