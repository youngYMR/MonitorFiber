#encoding=utf8
import pandas as pd
import numpy as np
import os
from model import CLNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def prune(model,cfg,percent):
    """
    :param model: 模型配置文件
    :param cfg: 模型配置参数
    :param percent: 删减比例
    :return:
    """
    model = model.cuda()
    model.train(False)

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]  # 一共有total个卷积channels

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()  # 输出gamma，伸缩因子
            index += size

    y, i = torch.sort(bn)  # BN层gamma值排序
    thre_index = int(total * percent)
    thre = y[thre_index].cuda()  # 删减阈值

    pruned = 0  # 剪枝个数
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(thre).float().cuda()  # 伸缩因子大于阈值的为1.0,否则为0.0
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))  # 记录剪枝后cfg，[19,19,36,31,70,84]
            cfg_mask.append(mask.clone())  # 记录剪枝前cfg,是个BN通道张量，[0,0,0,1,1,,0...,1,0]
        elif isinstance(m, nn.AvgPool2d):
            cfg.append('A')

    model_iter = nn.Sequential(*list(model.children())[:1])[0]  # 获取CNN迭代部分
    newmodel_iter = nn.Sequential(*list(model.children())[:1])[0]

    layer_id_in_cfg = 0
    start_mask = torch.ones(1)
    end_mask = cfg_mask[layer_id_in_cfg]

    for [m0, m1] in zip(model_iter, newmodel_iter):

        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg].cpu().numpy())))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1

        elif isinstance(m0, nn.Conv1d):
            if layer_id_in_cfg == 0:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg].cpu().numpy())))
                w = m0.weight.data[:,:, :, :]
                w = w[:,idx1, :, :]
                m1.weight.data = w.clone()
                print('In shape: ,Out shape:', 1, idx1.shape)
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg - 1].cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg].cpu().numpy())))

                w = m0.weight.data[:, idx0, :,:]
                if len(w.size()) == 2:
                    w = w.unsqueeze(1)

                w = w[idx1, :, :, :].clone()
                m1.weight.data = w.clone()
                print('In shape: ,Out shape:', idx0.shape, idx1.shape)
                # print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))

    def SVD(mat, k=1):
        u, s, v = np.linalg.svd(mat)
        u_ = u[:, :k]
        s_ = s[:k]
        v_ = v[:k, :]
        L = np.dot(np.diag(s_), v_)  # 3*2
        M = np.dot(u_, L)  # 2*5632
        print(L.shape, M.shape)
        return M

    for m0 in model.modules():
        if isinstance(m0, nn.Linear):
            linear_w = m0.weight.data.clone()
            for m1 in model.modules():
                if isinstance(m1, nn.Linear):
                    M = SVD(linear_w.clone().cpu().numpy())  # 分解后矩阵
                    m1.weight.data = torch.from_numpy(M).float().cuda()

    print(cfg)
    prune_save_path = "./prune_model/"
    if not os.path.exists(prune_save_path):
        os.mkdir(prune_save_path)
    torch.save({"cfg": cfg, "state_dict": model.state_dict()}, prune_save_path + "pruned.pth.tar")


