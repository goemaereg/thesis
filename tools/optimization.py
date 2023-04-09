import os
import io
from tqdm import tqdm
import argparse
import cProfile, pstats
from src.main import Trainer
from src.config import configure_load, configure_log_folder, configure_log, configure_data_paths, \
    configure_mask_root, configure_reporter, configure_pretrained_path, get_architecture_type, get_configs
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from abc import ABC, abstractmethod

lmdb_path = 'data/dataset/ILSVRC/lmdb_train.lmdb'

class ModelFreezer(ABC):
    def __init__(self, model):
        self.model = model
        self.frozen = None

    def is_frozen(self):
        return self.frozen is True
    @abstractmethod
    def _freeze_features(self):
        pass
    @abstractmethod
    def _unfreeze_features(self):
        pass
    def freeze_features(self):
        self._freeze_features()
        self.frozen = True
    def unfreeze_features(self):
        self._unfreeze_features()
        self.frozen = False

class VggCamFreezer(ModelFreezer):
    def __init__(self, model):
        super(VggCamFreezer, self).__init__(model)
    def _freeze_features(self):
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.conv6.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
    def _unfreeze_features(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
        for param in self.model.conv6.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True

class ResNetCamFreezer(ModelFreezer):
    def __init__(self, model):
        super(ResNetCamFreezer, self).__init__(model)
        self.feature_layers = [self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
                               self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4,
                               self.model.avgpool]
    def _freeze_features(self):
        for layer in self.feature_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
    def _unfreeze_features(self):
        for layer in self.feature_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True

class BaseMethod:
    def __init__(self,
                 model,
                 optimizer,
                 device,
                 **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self, images, labels):
        output_dict = self.model(images)
        logits = output_dict['logits']
        loss = self.loss_fn(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return logits, loss


class MinMaxCAMMethod(BaseMethod):
    def __init__(self, **kwargs):
        super(MinMaxCAMMethod, self).__init__(**kwargs)
        self.mse_loss = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.minmaxcam_class_set_size = kwargs.get('minmaxcam_class_set_size', 5)
        self.minmaxcam_batch_set_size = kwargs.get('minmaxcam_batch_set_size', 12)
        self.minmaxcam_frr_weight = kwargs.get('minmaxcam_frr_weight', 10)
        self.minmaxcam_crr_weight = kwargs.get('minmaxcam_crr_weight', 1)
        self.freezer = None
        if self.model.__class__.__name__ == 'VggCam':
            self.freezer = VggCamFreezer(self.model)
        elif self.model.__class__.__name__ == 'ResNetCam':
            self.freezer = ResNetCamFreezer(self.model)
        else:
            raise NotImplementedError

    def regularization_loss(self, images, labels):
        # images shape: mini batch size, RGB channels, H, W
        # Compute CAMs from B(I) with I=input image
        result_orig = self.model(images, labels, return_cam=True)
        cams = result_orig['cams']
        # cams shape: mini batch size, 1, H, W
        cams = cams.unsqueeze(1)
        # rescale cams to images input size
        resize = tuple(images.shape[2:])
        cams_resized = F.interpolate(cams, size=resize, mode='bilinear')
        # normalize cams
        cams_min = torch.amin(cams_resized, dim=(2, 3), keepdim=True)
        cams_max = torch.amax(cams_resized, dim=(2, 3), keepdim=True)
        cams_normalized = torch.zeros_like(cams_resized)
        if ((not torch.isnan(cams_resized).any()) and
                (not torch.equal(cams_min, cams_max))):
            cams_normalized = (cams_resized - cams_min) / (cams_max - cams_min)

        # Compute B(I * cams_normalized)
        result_mask = self.model(images * cams_normalized)
        # x = self.model.features(images * cams_normalized)
        # x = self.model.conv6(x)
        # out_extra_masked = self.model.relu(x)

        # compute features_i
        features_i = result_mask['avgpool_flat']
        # compute features_o
        features_o = result_orig['avgpool_flat']

        # compute losses
        loss_crr = 0.0
        loss_frr = 0.0
        # compute ss and bs per target
        # ss and bs can be less than minmaxcam_class_set_size and minmaxcam_batch_set_size for small datasets
        # ss = self.args.minmaxcam_class_set_size
        # bs = self.args.minmaxcam_batch_set_size
        # adapt bs to actual number of unique labels
        # labels_np = labels.numpy(force=True)
        # bs = np.unique(labels_np).shape[0]
        # starts, stops mark ranges of sequentially identical classes
        lb_diff = torch.diff(labels)
        one = torch.tensor([1]).to(self.device)
        zero = torch.tensor([0]).to(self.device)
        starts = torch.nonzero(torch.concatenate([one, lb_diff]), as_tuple=True)[0]
        stops = torch.nonzero(torch.concatenate([zero, lb_diff, one]), as_tuple=True)[0]
        # starts = np.nonzero(np.r_[1, np.diff(labels_np)])[0]
        # stops = np.nonzero(np.r_[0, np.diff(labels_np), 1])[0]
        bs = 0
        for start, stop in zip(starts, stops):
            loss_crr_ss = 0.0
            f_i_pair_num = 0
            features_i_ss = features_i[start:stop]
            features_o_ss = features_o[start:stop]
            ss = stop - start
            for j, k in itertools.combinations(range(ss), 2):
                f_i_pair_num += 1
                feature_i_ss_j = features_i_ss[j].unsqueeze(0)
                feature_i_ss_k = features_i_ss[k].unsqueeze(0)
                loss_crr_ss += self.mse_loss(feature_i_ss_j, feature_i_ss_k)
            loss_frr += self.mse_loss(features_i_ss, features_o_ss)
            loss_crr += loss_crr_ss / f_i_pair_num
            bs += 1
        loss_crr /= bs
        loss_frr /= bs
        return loss_crr, loss_frr

    def train(self, images, labels):
        # minmaxcam stage I
        # is_frozen call makes sure freezer is unfrozen at first call of train()
        if self.freezer.is_frozen():
            self.freezer.unfreeze_features()
        self.model.train()
        output_dict = self.model(images, labels, return_cam=True)
        logits = output_dict['logits']
        loss = self.loss_fn(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # minmaxcam state II
        self.freezer.freeze_features()
        self.optimizer.zero_grad()
        self.model.eval()
        loss_crr, loss_frr = self.regularization_loss(images, labels)
        self.model.train()
        loss_all_p2 = self.minmaxcam_crr_weight * loss_crr + \
                      self.minmaxcam_frr_weight * loss_frr
        try:
            loss_all_p2.backward()
        except RuntimeError as e:
            print(e)
            print(loss_all_p2)
        self.optimizer.step()

        self.freezer.unfreeze_features()

        return logits, loss


def train_init(targs):
    # targs = argparse.Namespace(**configure_load(args.config))
    targs.experiment_name = 'minmaxcamperf'
    targs.log_folder = 'perf_log'
    targs.log_folder = configure_log_folder(targs)
    targs.log_path = configure_log(targs)
    print(f'workers = {targs.workers}')
    trainer = Trainer(targs, log=False)
    last_epoch = trainer.epoch - 1
    trainer.set_lr_scheduler(trainer.optimizer, last_epoch)
    method_args = vars(trainer.args) | {'model': trainer.model, 'device': trainer.device, 'optimizer': trainer.optimizer}
    trainer.wsol_method = MinMaxCAMMethod(**method_args)
    return trainer

def perf(args):
    num_images = 0
    trainer = train_init(args)
    epochs_range = range(1) # range(trainer.epoch, trainer.args.epochs, 1)
    tq0 = tqdm(epochs_range, total=len(epochs_range), desc='training epochs')
    try:
        for epoch in tq0:
            trainer.model.train()
            loader = trainer.loaders['train']
            total_loss = 0.0
            num_correct = 0
            tq1 = tqdm(loader, total=len(loader), desc='dataset loading')
            for images, targets, _ in tq1:
                images = images.to(trainer.device)
                targets = targets.to(trainer.device)
                logits, loss = trainer.wsol_method.train(images, targets)
                pred = logits.argmax(dim=1)
                total_loss += loss.item() * images.size(0)
                num_correct += (pred == targets).sum().item()
                num_images += images.size(0)
            loss = total_loss / float(num_images)
            accuracy = num_correct / float(num_images)  # * 100
            print(f'train: epoch = {epoch} loss = {loss}, accuracy = {accuracy}')
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
    # parser.add_argument('--path', '-p', type=str, default=lmdb_path, help='lmdb path')
    # parser.add_argument('--num_images', '-i', type=int, default=np.inf, help='number of images to process')
    # parser.add_argument('--config', '-c', type=str, help='Configuration JSON file path with saved arguments')
    # parser.add_argument('--workers', '-w', type=int, default=0)
    # parser.add_argument('--num_stats', '-r', type=int, help='top number of statistics in pstats')
    # args = parser.parse_args()
    args = get_configs()
    restrictions = [20]
    restrictions = tuple(restrictions)
    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...
    perf(args)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
    pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME).print_stats(20)
    print(s.getvalue())