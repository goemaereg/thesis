"""
Original repository: https://github.com/clovaai/CutMix-PyTorch
"""

import numpy as np
import torch
from .base_method import BaseMethod

__all__ = ['cutmix']


def cutmix(x, target, beta):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size()[0]).cuda()

    target_a = target.clone().detach()
    target_b = target[rand_index].clone().detach()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam


def rand_bbox(size, lam):
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


class CutMixMethod(BaseMethod):
    def __init__(self, **kwargs):
        super(CutMixMethod, self).__init__(**kwargs)
        self.cutmix_prob = kwargs.get('cutmix_prob', 1.0)
        self.cutmix_beta = kwargs.get('cutmix_beta', 1.0)

    def train(self, images, labels):
        if self.cutmix_prob > np.random.rand(1) and self.cutmix_beta > 0:
            images, target_a, target_b, lam = cutmix(images, labels, self.cutmix_beta)
            output_dict = self.model(images)
            logits = output_dict['logits']
            loss = (self.loss_fn(logits, target_a) * lam +
                    self.loss_fn(logits, target_b) * (1. - lam))
        else:
            output_dict = self.model(images)
            logits = output_dict['logits']
            loss = self.loss_fn(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return logits, loss
