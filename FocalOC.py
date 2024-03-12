import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalOC_loss(nn.Module):
    def __init__(self, feat_dim=212, alpha=4):
        super(FocalOC_loss, self).__init__()
        self.feat_dim = feat_dim
        self.alpha = alpha
        init = torch.randn(1, self.feat_dim)
        self.centers = nn.Parameter(init)
        nn.init.kaiming_uniform_(self.centers, 0.25)
        self.gamm = 1.0
        self.scale_pos = 40
        self.scale_neg = 40
        self.r_real = 0.05
        self.r_fake = 0.4

    def forward(self, x, labels, lam, index):
        """
        Args:
            x: featrue matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """  
        w = F.normalize(self.centers, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        x = x @ w.transpose(0,1)
        scores = x.clone()
        distance = (1.0 - x) / 2.

        if index != None:
            mask1 = torch.ones(labels.shape).long().cuda()
            mask1 = labels.eq(mask1)
            loss_1, true_loss_1, false_loss_1 = self.calc(distance, mask1)

            labels2 = labels[index]
            mask2 = torch.ones(labels2.shape).long().cuda()
            mask2 = labels2.eq(mask2)
            loss_2, true_loss_2, false_loss_2 = self.calc(distance, mask2)
            
            loss = lam * loss_1 + (1-lam) * loss_2
            true_loss = lam * true_loss_1 + (1-lam) * true_loss_2
            false_loss = lam * false_loss_1 + (1-lam) * false_loss_2
        else:
            mask = torch.ones(labels.shape).long().cuda()
            mask = labels.eq(mask)
            loss, true_loss, false_loss = self.calc(distance, mask)
            
        return loss, true_loss, false_loss, scores.squeeze(1)
    
    def calc(self, x, mask):
        x_true = x[mask]
        x_fake = x[~mask]
        
        sigma_true = torch.clamp(x_true, min=0.) ** self.alpha
        sigma_fake = torch.clamp(0.5-x_fake, min=0.) ** self.alpha

        true_loss = 1.0 / 2 * torch.log(
            1 + torch.sum(sigma_true * torch.exp(self.scale_pos * (x_true - self.r_real))))
        false_loss = 1.0 / 2 * torch.log(
            1 + torch.sum(sigma_fake * torch.exp(self.scale_neg * (self.r_fake - x_fake))))
        
        loss = true_loss + self.gamm * false_loss
        
        return loss, true_loss, false_loss



