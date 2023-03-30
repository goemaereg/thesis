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
import torchvision.transforms.functional as TF
from torch.utils.data import Sampler
from typing import Iterator, List

_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_SPLITS = ('train', 'val', 'test')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))

def configure_metadata(metadata_root):
    metadata = mch()
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
    def __init__(self, data_root, metadata_root, transform, proxy, num_sample_per_class=0, 
                 bboxes_path=None, bbox_mask_strategy=None):
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class
        self._adjust_samples_per_class()
        self.bbox_mask_strategy = bbox_mask_strategy
        self.computed_bboxes = {}
        if bboxes_path is not None and os.path.exists(bboxes_path):
            self.computed_bboxes = get_bounding_boxes_from_file(bboxes_path)

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

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')
        image = self.transform(image)
        if self.bbox_mask_strategy is not None and image_id in self.computed_bboxes:
            bboxes = self.computed_bboxes[image_id]
            for bbox in bboxes:
                # convert box(x0,y0,x1,y1) coordinates to array(i,j) semantics
                x0, y0, x1, y1 = bbox
                w = x1 - x0
                h = y1 - y0
                i, j = y0, x0
                if self.bbox_mask_strategy == 'zero':
                    v = 0 # erasing value
                    image = TF.erase(image, i, j, h, w, v, inplace=True)
                elif self.bbox_mask_strategy == 'mean':
                    v = torch.mean(image)
                    image = TF.erase(image, i, j, h, w, v, inplace=True)
                elif self.bbox_mask_strategy == 'random':
                    v = np.random.randn(3, h, w)
                    for i in range(len(_IMAGE_STD_VALUE)):
                        v[i,:,:] = _IMAGE_STD_VALUE[i] * v[i,:,:] + _IMAGE_MEAN_VALUE[i]
                    v = torch.tensor(v)
                    image[..., i:i+h, j:j+w] = v
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


class WSOLImageLabelTarDataset(Dataset):
    def __init__(self, tar_path, metadata_root, transform):
        self.tar_path = tar_path
        self.tar_file = tarfile.open(name=tar_path, mode='r')
        self.metadata = configure_metadata(metadata_root)
        self.image_ids = get_image_ids(self.metadata, proxy=False)
        self.image_labels = get_class_labels(self.metadata)
        self.transform = transform

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
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


class WSOLImageLabelLmdbDataset(Dataset):
    environments = dict()
    def __init__(self, lmdb_path, metadata_root, transform):
        self.lmdb_path = lmdb_path
        if lmdb_path not in self.environments:
            self.environments[lmdb_path] = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                                                     readonly=True, lock=False,
                                                     readahead=False, meminit=False)
        with self.environments[lmdb_path].begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
        self.metadata = configure_metadata(metadata_root)
        self.image_ids = get_image_ids(self.metadata, proxy=False)
        self.image_labels = get_class_labels(self.metadata)
        self.transform = transform

    def __getitem__(self, idx):
        image, target = None, None
        # example image_id: train/n01440764/n01440764_10026.JPEG
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        env = self.environments[self.lmdb_path]
        with env.begin(write=False) as txn:
            key = u'{}'.format(image_id).encode('ascii')
            byteflow = txn.get(key)
        image_buffer = pickle.loads(byteflow)
        with io.BytesIO(image_buffer) as bio:
            image = Image.open(bio).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_label, image_id

    def __len__(self):
        return self.length


class MiniBatchSampler(Sampler[int]):
    def __init__(self, image_label_dataset,
                 class_set_size=5,
                 batch_set_size=12) -> None:
        super().__init__(image_label_dataset)
        self.dataset = image_label_dataset
        self.labels = list(self.dataset.image_labels.values())
        self.unique_labels = np.unique(self.labels)
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

def get_eval_loader(split, data_root, metadata_root, batch_size, workers,
                    resize_size, bboxes_path=None, bbox_mask_strategy=None):
    dataset_transforms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    ])
    dataset = WSOLImageLabelDataset(
        data_root=data_root,
        metadata_root=os.path.join(metadata_root, split),
        transform=dataset_transforms,
        proxy=False,
        bboxes_path=bboxes_path,
        bbox_mask_strategy=bbox_mask_strategy)
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers)

def get_data_loader(data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, proxy_training_set,
                    num_val_sample_per_class=0, batch_set_size=None,
                    class_set_size=None, train_augment=True, dataset_name='SYNTHETIC'):
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]))
    if not train_augment:
        dataset_transforms['train'] = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ])
    loaders = {}
    for split in _SPLITS:
        if dataset_name == 'ILSVRC' and split == 'train':
            dataset = WSOLImageLabelLmdbDataset(
                lmdb_path = os.path.join(data_roots[split], 'lmdb_train.lmdb'),
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split]
            )
        else:
            dataset = WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0)
            )
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
        loaders |= {
         split: DataLoader(
             dataset,
             batch_size=_batch_size,
             shuffle=shuffle,
             batch_sampler=batch_sampler,
             num_workers=workers)
        }

    return loaders
