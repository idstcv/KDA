# Copyright (c) Alibaba Group

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['KDA']


class KDA(nn.Module):
    def forward(self, s_feat, t_feat, s_center, t_center):
        s_sim = s_feat.matmul(s_center.t())
        with torch.no_grad():
            t_sim = t_feat.matmul(t_center.t())
        loss = F.smooth_l1_loss(s_sim, t_sim, reduction='sum') / torch.norm(t_sim)
        return loss
