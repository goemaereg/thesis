from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.data_loaders import configure_metadata, get_image_ids, get_class_labels
from PIL import Image
import tarfile
import lmdb
import pickle
import os
import io
import cv2
import torch
import argparse


class ImagenetTrainTarDataset(Dataset):
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
            print('hello')
        image_buffer = pickle.loads(byteflow)
        with io.BytesIO(image_buffer) as bio:
            image = Image.open(bio).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_label, image_id

    def __len__(self):
        return self.length


_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
resize_size = 256
crop_size = 224

transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
            # transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)])

tar_path = 'data/dataset/ILSVRC/ILSVRC2012_img_train.tar'
lmdb_path = 'lmdb_train.lmdb'
metadata_root = 'data/metadata/ILSVRC/train'


def get_image_by_index(index):
    dataset = WSOLImageLabelLmdbDataset(lmdb_path, metadata_root, transform)
    return dataset[index]

def save_image(image, image_id):
    img = torch.permute(image, (1, 2, 0)) # CHW -> HWC
    img_np = img.detach().cpu().numpy()
    img_np = (img_np * 255).astype('uint8')
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
    fname = image_id.split('/')[2]
    cv2.imwrite(fname, img_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=1, help='image count')
    parser.add_argument('--offset', type=int, default=0, help='image start index')
    args = parser.parse_args()
    dataset = WSOLImageLabelLmdbDataset(lmdb_path, metadata_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        batch_sampler=None,
        num_workers=0)
    i = 0
    for image, label, image_id in loader:
        print(f'image.shape: {image.shape}, label: {label}, image_id: {image_id}')
        i += 1
        if i >= args.images:
            break
