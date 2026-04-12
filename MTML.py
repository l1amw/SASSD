import torch
import math
from importlib import import_module
from ShuffleMix import ShuffleMix
from FocalOC import FocalOC

def get_model(device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format("Wav2vec2-Aasist"))
    _model = getattr(module, "Model")
    model = _model().to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    return model


def inner_train_step(data, model, optim, FocalOC, FocalOC_optim, device):
    batch_x, batch_y = data[0], data[1]
    batch_x = batch_x.to(device)
    batch_y = batch_y.view(-1).type(torch.int64).to(device)

    batch_x, lam, index = DataAug(batch_x)
    feat = model(batch_x, batch_y)
    loss, *_ = FocalOC(feat, batch_y, lam, index)
    
    optim.zero_grad()
    FocalOC_optim.zero_grad()
    loss.backward()
    optim.step()
    FocalOC_optim.step()
    return loss


def update_meta_model(meta_model, model, optimizers):
    meta_params, meta_states = get_parameters(meta_model)
    params, states = get_parameters(model)
    
    optimizers.zero_grad()

    # update grad
    for k in meta_params.keys():
        new_v, old_v = params[k], meta_params[k]
        meta_params[k].grad = (old_v - new_v).data

    # update with original optimizers
    optimizers.step()

    for k in meta_states.keys():
        meta_states[k].data = states[k].data


def get_parameters(model):
    parameters = dict(model.named_parameters())
    states = dict(model.named_buffers())
    return parameters, states


def adjust_learning_rate_loss(optimizer, epoch_num):
    lr = 0.0003 * (0.5 ** (epoch_num // 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_meta_lr(epoch_num, iter, ep_iters, max_eps, max_lr, min_lr):
    if epoch_num < max_eps:
        meta_lr = max_lr * (1. + math.cos(math.pi*(epoch_num + iter / ep_iters) / max_eps)) / 2.
        meta_lr = max(meta_lr, min_lr)
        return meta_lr
    else:
        return min_lr


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


device = "cuda"
model = get_model(device)
meta_model = get_model(device)
model.load_state_dict(meta_model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.0001)
meta_optimizer = torch.optim.SGD(meta_model.parameters(), lr=1.0, momentum=0.2)

DataAug = ShuffleMix(MixOrCut="CutMix", alpha=1.0, shuff="True", num_seg=3)
FocalOC = FocalOC(feat_dim=212, alpha=4).to(device)
FocalOC_optim = torch.optim.SGD(FocalOC.parameters(), lr=3e-4)


def MTML(meta_data_loader, model, optim, meta_model, meta_optim, FocalOC, FocalOC_optim, device, cur_epoch, multi_tasks=3):
    model.train()
    meta_model.train()
    FocalOC.train()

    adjust_learning_rate_loss(FocalOC_optim, cur_epoch)

    # 250 iterations per epoch.
    for iter, batch in enumerate(meta_data_loader):
        meta_lr = get_meta_lr(cur_epoch, iter, len(meta_data_loader), 100, 1.0, 0.05)
        set_lr(meta_optim, meta_lr)

        for ii in range(multi_tasks):
            mtrn_x, mtrn_y, mtes_x, mtes_y = batch[ii][0], batch[ii][1], batch[ii][2], batch[ii][3]
            mtrn_x = mtrn_x.reshape(-1, mtrn_x.shape[-1])
            mtes_x = mtes_x.reshape(-1, mtes_x.shape[-1])

            #meta-train step
            mtrn_data = [mtrn_x, mtrn_y]
            loss_mtrn = inner_train_step(mtrn_data, model, optim, FocalOC, FocalOC_optim, device) 

            #meta-test step
            mtes_data = [mtes_x, mtes_y]
            loss_mtes = inner_train_step(mtes_data, model, optim, FocalOC, FocalOC_optim, device)

        update_meta_model(meta_model, model, meta_optim)

        model.load_state_dict(meta_model.state_dict())

    return loss_mtrn, loss_mtes
