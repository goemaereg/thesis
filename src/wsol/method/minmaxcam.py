from .base_method import BaseMethod
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from abc import ABC, abstractmethod

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
            for param in layer.params():
                param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
    def _unfreeze_features(self):
        for layer in self.feature_layers:
            for param in layer.params():
                param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True


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
        # Compute CAMs from B(I) with I=input image
        result_orig = self.model(images, labels,
                                 return_cam=True, clone_cam=False)
        cams = result_orig['cams']
        # cams = t2n(cams)
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
        labels_np = labels.numpy(force=True)
        bs = np.unique(labels_np).shape[0]
        # starts, stops mark ranges of sequentially identical classes
        starts = np.nonzero(np.r_[1, np.diff(labels_np)])[0]
        stops = np.nonzero(np.r_[0, np.diff(labels_np), 1])[0]
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
        loss_crr /= bs
        loss_frr /= bs
        return loss_crr, loss_frr

    def train(self, images, labels):
        # minmaxcam stage I
        # is_frozen call makes sure freezer is unfrozen at first call of train()
        if self.freezer.is_frozen():
            self.freezer.unfreeze_features()
        self.model.train()
        output_dict = self.model(images)
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
