import os
import munch
from torch.utils.data import Dataset, DataLoader
import lmdb
import pickle
from PIL import Image
import numpy as np
import io
from torch.utils.data import Sampler
from typing import Iterator, List
from src.data_loaders import _CAT_IMAGE_MEAN_STD
from torchvision import transforms
from tqdm import tqdm
import argparse
from src.data_loaders import configure_metadata, get_image_ids, get_class_labels
import torch
import cProfile, pstats
from src.main import Trainer
from src.config import configure_load, configure_log_folder, configure_log, configure_data_paths, \
    configure_mask_root, configure_reporter, configure_pretrained_path, get_architecture_type


data_root = 'data/dataset'
metadata_root = 'data/metadata'
split = 'train'
batch_size=256
resize_size=256
crop_size = 224
proxy_training_set = False
num_val_sample_per_class = 0
batch_set_size = 12
class_set_size = 5
train_augment = True
dataset_name = 'ILSVRC'
metadata_root = os.path.join(metadata_root, dataset_name)
train = val = test = os.path.join(data_root, dataset_name)
data_paths = munch.Munch(dict(train=train, val=val, test=test))
lmdb_path = 'data/dataset/ILSVRC/lmdb_train.lmdb'
#lmdb_path = '/project_scratch/thesis/data/dataset/ILSVRC/lmdb_train.lmdb'
lmdb_path_multi = '/project/imagenet/train.lmdb'


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

    def __init__(self, lmdb_path, metadata_root, transform, normalize, lmdb_multi=False):
        self.lmdb_path = lmdb_path
        self.env = None
        self.length = 1281168
        self.metadata = configure_metadata(metadata_root)
        self.image_labels = get_class_labels(self.metadata)
        self.image_ids = list(self.image_labels)
        self.transform = transform
        self.normalize = normalize
        self.lmdb_multi = lmdb_multi
        # self.env, self.length = self._init_db(lmdb_path)

    def __getitem__(self, idx):
        if self.env is None:
            self.env, self.length = self._init_db(self.lmdb_path)
        image, target = None, None
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        key_id = idx if self.lmdb_multi else image_id
        with self.env.begin(write=False) as txn:
            key = u'{}'.format(key_id).encode('ascii')
            byteflow = txn.get(key)

        try:
            unpacked = pickle.loads(byteflow)
        except TypeError as e:
            print(f'TypeError with key = {image_id}')
            raise e
        imgbuf = unpacked[0] if self.lmdb_multi else unpacked
        with io.BytesIO(imgbuf) as bio:
            image = Image.open(bio)
            image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            if self.normalize is not None:
                image = self.normalize(image)
            return image, image_label, image_id
            # return torch.tile(image, (5,1,1,1)), torch.tile(torch.tensor(image_label), (5,1)), [image_id] * 5

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
        for i in range(self.batch_num):
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


transform = transforms.Compose([
    transforms.Resize((resize_size, resize_size)),
    transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
metadata_root_split = os.path.join(metadata_root, split)
mean_std = _CAT_IMAGE_MEAN_STD[metadata_root_split]
mean, std = mean_std['mean'], mean_std['std']
normalize = transforms.Normalize(mean, std)

def custom_collate(batch):
    image_list = []
    label_list = []
    id_list = []
    for images, labels, image_ids in batch:
        image_list.extend(images)
        label_list.extend(labels)
        id_list.extend(image_ids)
    images = torch.stack(image_list, dim=0)
    labels = torch.tensor(label_list)
    return images, labels, id_list

def train_init(args):
    targs = argparse.Namespace(**configure_load(args.config))
    targs.log_folder = 'perf_log'
    targs.experiment_name = 'minmaxcamperf'
    targs.workers = 0
    targs.log_folder = configure_log_folder(targs)
    targs.log_path = configure_log(targs)
    targs.architecture_type = get_architecture_type(targs.architecture_type, targs.wsol_method)
    targs.data_paths = configure_data_paths(targs)
    targs.metadata_root = os.path.join(targs.metadata_root, targs.dataset_name)
    targs.mask_root = configure_mask_root(targs)
    targs.reporter, targs.reporter_log_root = configure_reporter(targs)
    targs.pretrained_path = configure_pretrained_path(targs)
    trainer = Trainer(targs, log=False)
    last_epoch = trainer.epoch - 1
    trainer.set_lr_scheduler(trainer.optimizer, last_epoch)
    return trainer

def perf(args):
    lmdb_multi = args.path == lmdb_path_multi
    dataset = WSOLImageLabelLmdbDataset(
        lmdb_path=args.path,
        metadata_root=os.path.join(metadata_root, split),
        transform=transform, normalize=normalize,
        lmdb_multi=lmdb_multi
    )

    if args.batch_sampler is True:
        _batch_size = 1
        shuffle = None

        batch_sampler = MiniBatchSampler(
            dataset,
            batch_set_size=args.minmaxcam_batch_set_size,
            class_set_size=args.minmaxcam_class_set_size)
    else:
        _batch_size = args.batch_size
        shuffle = args.shuffle
        batch_sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=_batch_size,
        shuffle=shuffle,
        batch_sampler=batch_sampler,
        num_workers=args.workers)
    num_images = 0
    trainer = None
    if args.train and args.config is not None:
        trainer = train_init(args)

    try:
        if trainer is not None:
            epochs_range = range(1) # range(trainer.epoch, trainer.args.epochs, 1)
            tq0 = tqdm(epochs_range, total=len(epochs_range), desc='training epochs')
            for epoch in tq0:
                trainer.model.train()
                loader = trainer.loaders[split]
                total_loss = 0.0
                num_correct = 0
                # tq1 = tqdm(loader, total=len(loader), desc='dataset loading')
                tq1 = tqdm(dataloader, total=len(dataloader), desc='imagenet')
                for images, targets, _ in tq1:
                    images = images.to(trainer.device)
                    targets = targets.to(trainer.device)
                    logits, loss = trainer.wsol_method.train(images, targets)
                    pred = logits.argmax(dim=1)
                    total_loss += loss.item() * images.size(0)
                    num_correct += (pred == targets).sum().item()
                    num_images += images.size(0)
                    if num_images >= args.num_images:
                        break
                if num_images >= args.num_images:
                    break
                loss = total_loss / float(num_images)
                accuracy = num_correct / float(num_images)  # * 100
                print(f'train: epoch = {epoch} loss = {loss}, accuracy = {accuracy}')

        else:
            for index, (images, targets, image_ids) in enumerate(tqdm(dataloader, total=len(dataloader), desc='imagenet')):
                num_images += images.shape[0]
                if num_images >= args.num_images:
                    break
    except KeyboardInterrupt as e:
        print('Stopped after keyboard interrupt.')
    print(f'Iterated {num_images} images')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default=lmdb_path, help='lmdb path')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='number of images to process')
    parser.add_argument('--batch_sampler', '-m', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--minmaxcam_class_set_size', '-l', type=int, default=5, help='MinMaxCam class set size'),
    parser.add_argument('--minmaxcam_batch_set_size', '-a', type=int, default=12, help='MinMaxCam batch set size'),
    parser.add_argument('--num_images', '-i', type=int, default=np.inf, help='number of images to process')
    parser.add_argument('--shuffle', '-s', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--train', '-t', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--config', '-c', type=str, help='Configuration JSON file path with saved arguments')
    parser.add_argument('--workers', '-w', type=int, default=0)
    parser.add_argument('--num_stats', '-r', type=int, help='top number of statistics in pstats')
    args = parser.parse_args()
    restrictions = []
    if args.num_stats is not None:
        restrictions.append(args.num_stats)
    restrictions = tuple(restrictions)
    pr = cProfile.Profile()
    pr.enable()

    # ... do something ...
    perf(args)

    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(*restrictions)
    pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME).print_stats(*restrictions)
    print(s.getvalue())
