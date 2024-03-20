import torch
import torch.nn as nn
from copy import deepcopy
import math
from ShuffleMix import ShuffleMix


class MTML(nn.Module):
    def __init__(self, args, device):
        super(MTML, self).__init__()
        self.args = args
        self.device = device
        self.DataAug = ShuffleMix(args.alpha, args.num_seg, args.Shuffle, args.MixOrCut)

    def forward(self, meta_data_loader, model, optim, FocalOC, FocalOC_optim, cur_epoch):
        model.train()
        FocalOC.train()
        
        self.adjust_learning_rate_loss(FocalOC_optim, cur_epoch)

        # There are 250 iterations (batches) per epoch.
        for iter, batch in enumerate(meta_data_loader}:
            meta_lr = self.get_meta_lr(cur_epoch, iter, 70, 0.05):
            
            original_weights = [deepcopy(model.state_dict()), deepcopy(FocalOC.state_dict())]
            new_weights = [[],[]]

            for ii in range(self.args.multi_tasks):
                mtrn_x, mtrn_y, mtes_x, mtes_y = batch[ii][0], batch[ii][1], batch[ii][2], batch[ii][3]
                mtrn_x = mtrn_x.reshape(-1, mtrn_x.shape[-1])
                mtes_x = mtes_x.reshape(-1, mtes_x.shape[-1])

                #meta-train step
                mtrn_data = [mtrn_x, mtrn_y]
                _ = self.inner_train_step(mtrn_data, model, optim, FocalOC, FocalOC_optim) 

                #meta-test step
                mtes_data = [mtes_x, mtes_y]
                _ = self.inner_train_step(mtes_data, model, optim, FocalOC, FocalOC_optim)

            new_weights[0].append(deepcopy(model.state_dict()))
            new_weights[1].append(deepcopy(FocalOC.state_dict()))

            #model update step
            self.update_model(model, original_weights[0], new_weights[0], meta_lr)
            self.update_model(FocalOC, original_weights[1], new_weights[1], 1.0)

        return model, FocalOC


    def inner_train_step(self, data, model, optim, FocalOC, FocalOC_optim):
        batch_x, batch_y = data[0], data[1]
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.view(-1).type(torch.int64).to(self.device)

        batch_x, lam, index = self.DataAug(batch_x)
        feat = model(batch_x)
        loss, _, _, _ = FocalOC(feat, batch_y, lam, index)
        
        optim.zero_grad()
        FocalOC_optim.zero_grad()
        loss.backward()
        optim.step()
        FocalOC_optim.step()
        return loss


    def update_model(self, model, original_weights, new_weights, meta_lr):
        ws = len(new_weights)
        fweights = { name : new_weights[0][name]/float(ws) for name in new_weights[0] }
        for i in range(1, ws):
            for name in new_weights[i]:
                fweights[name] += new_weights[i][name]/float(ws)

        model.load_state_dict({name : 
            original_weights[name] + ((fweights[name] - original_weights[name]) * meta_lr) for name in original_weights})


    def adjust_learning_rate_loss(self, optimizer, epoch_num):
        lr = 0.0003 * (0.5 ** (epoch_num // 8))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def get_meta_lr(self, epoch_num, iter, max_eps, min_lr):
        if epoch_num < max_eps:
            meta_lr = 1.0 * (1. + math.cos(math.pi*(epoch_num + iter / 250) / max_eps)) / 2.
            meta_lr = max(meta_lr, min_lr)
            return meta_lr
        else:
            return min_lr


