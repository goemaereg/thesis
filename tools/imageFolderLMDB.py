import os.path as osp
from PIL import Image
import numpy as np

import lmdb
import tarfile

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pickle
import cv2
import time
import torch
from torchvision import transforms
import io

labels = [1, 25, 30, 32, 50, 61, 69, 71, 75, 76, 79, 99, 105, 107, 109, 113, 114, 115, 122, 123, 128, 145, 146, 149, 151, 187, 207, 208, 235, 267, 281, 283, 285, 286, 291, 294, 301, 308, 309, 311, 313, 314, 315, 319, 323, 325, 329, 338, 341, 345, 347, 349, 353, 354, 365, 367, 372, 386, 387, 398, 400, 406, 411, 414, 421, 424, 425, 427, 430, 435, 436, 437, 438, 440, 445, 447, 448, 457, 458, 462, 463, 466, 467, 470, 471, 474, 480, 485, 488, 492, 496, 500, 508, 509, 511, 517, 525, 526, 532, 542, 543, 557, 562, 565, 567, 568, 570, 573, 576, 604, 605, 612, 614, 619, 621, 625, 627, 635, 645, 652, 655, 675, 677, 678, 682, 683, 687, 704, 707, 716, 720, 731, 733, 734, 735, 737, 739, 744, 747, 758, 760, 761, 765, 768, 774, 779, 781, 786, 801, 806, 808, 811, 815, 817, 821, 826, 837, 839, 842, 845, 849, 850, 853, 862, 866, 873, 874, 877, 879, 887, 888, 890, 899, 900, 909, 910, 917, 923, 924, 928, 929, 932, 935, 938, 945, 947, 950, 951, 954, 957, 962, 963, 964, 967, 970, 972, 973, 975, 978, 988]
# 258, 758  -> nr images for limited Training

class ImageFolderLMDB(data.Dataset):
    environments = dict()
    # TODO: keep counters how many instances of the class are opened for each LMDB file. Delete the corresponding environment if the counter reaches zero.

    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        if db_path not in self.environments:
            self.environments[db_path] = lmdb.open(db_path, subdir=osp.isdir(db_path),
                                readonly=True, lock=False,
                                readahead=False, meminit=False)
        with self.environments[db_path].begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.environments[self.db_path]
        with env.begin(write=False) as txn:
            byteflow = txn.get(u'{}'.format(index).encode('ascii'))
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        with io.BytesIO(imgbuf) as bio:
            img = Image.open(bio).convert('RGB')


        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data

def image_reader(path):
    return Image.open(path).convert('RGB')

def folder2lmdb(dpath, name="train", write_frequency=5000, num_workers=16):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=image_reader)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = osp.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        # print(type(data), data)
        image, label = data[0]
        if label not in labels: continue
        size = image.size
        raw = image.tobytes()
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((size, raw, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(idx+1))

    print("Flushing database ...")
    db.sync()
    db.close()


def tar2lmdb(tarpath, name='train', write_frequency=5000):
    print("Loading dataset from %s" % tarpath)

    lmdb_path = osp.join("./%s.lmdb" % name)
    isdir = osp.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)

    tar = tarfile.TarFile(tarpath, 'r')
    idx = 0
    for label, tarfname in enumerate(sorted(tar.getnames())):
        if label not in labels:
            continue
        tar2 = tarfile.TarFile(fileobj=tar.extractfile(tarfname))
        for tarfname2 in tar2.getnames():
            image = tar2.extractfile(tarfname2).read()
            txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, labels.index(label))))  # TODO: this labels.index() is only used when making the limited dataset!
            if idx % write_frequency == 0:
                print("[%d/1'281'167]" % (idx))
                txn.commit()
                txn = db.begin(write=True)
            idx += 1

    # finish iterating through dataset
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(idx))
        print(idx)

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    tar2lmdb("../imagenet-2012/train.tar", "train_limited")
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    # dataset = ImageFolder("./val",transform=transform)
    dataset = ImageFolderLMDB("./val_limited.lmdb", transform=transform)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=512, collate_fn=lambda d : tuple(d))
    for i in range(10):
        start = time.time()
        for j, batch in enumerate(dataloader):
            if j%2 == 0:
                print(f"{j} / {len(dataloader)}")
            if j == 100:
                break
        end = time.time()
        print("TIME:", end-start)
