import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class ShuffleMix(nn.Module):
    def __init__(self, alpha=1.0, num_seg=3, shuff="True", MixOrCut="CutMix"):
        super(ShuffleMix, self).__init__()
        self.alpha = alpha
        self.num_seg = num_seg
        self.shuff = shuff
        self.MixOrCut = MixOrCut

    def forward(self, x):
        if self.shuff == "True":
            x = self.Shuffle(x, self.num_seg)

        if self.MixOrCut == "Mixup":
            x, lam, index = self.Mixup(x, self.alpha)
        elif self.MixOrCut == "CutMix":
            x, lam, index = self.CutMix(x, self.alpha)

        return x, lam, index

    def Shuffle(self, x, num_seg):
        # Shuffle inputs
        num_token = int(num_seg)
        if num_token == 1:
            return x
        
        x_len = x.shape[1]
        token_len = math.ceil(x_len / (num_token-1))
        sx = np.random.randint(int(token_len/4), int(token_len*3/4))

        shuffle_x = torch.zeros(0).cuda()
        for ii in random.sample(range(num_token), num_token):
            bbx1 = np.clip(sx + token_len * (ii - 1), 0, x_len)
            bbx2 = np.clip(sx + token_len * ii, 0, x_len)
            x_cut = x[:, bbx1:bbx2]
            shuffle_x = torch.cat((shuffle_x, x_cut), 1)
        
        return shuffle_x

    def Mixup(self, x, alpha):
        # Mixup inputs.
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0)).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, lam, index

    def CutMix(self, x, alpha):
        # Cutmix inputs.
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0)).cuda()
        bbx1, bbx2 = self.rand_bbox(x.size(1), lam)
        mixed_x = x
        mixed_x[:, bbx1:bbx2] = x[index, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) / x.size(1))
        return mixed_x, lam, index

    def Cutout(self, x, alpha):
        # Cutout inputs
        ratio = 1. - alpha
        bbx1, bbx2 = self.rand_bbox(x.size(1), ratio)
        mask = np.ones(x.size(1), np.float32)
        mask[bbx1: bbx2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(x).cuda()
        masked_x = x * mask
        return masked_x

    def rand_bbox(self, length, lam):
        cut_rat = 1. - lam
        cut_len = int(length * cut_rat)
        # uniform
        cx = np.random.randint(length)
        bbx1 = np.clip(cx - cut_len // 2, 0, length)
        bbx2 = np.clip(cx + cut_len // 2, 0, length)
        return bbx1, bbx2




