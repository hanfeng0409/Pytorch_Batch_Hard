 #!/usr/bin/env python
# coding=utf-8
import torch
import torchvision
from torch import nn
import numpy as np
import torch.nn.functional as F


class BatchHard(nn.Module):
    def __init__(self, margin=1,squared=False):
        super().__init__()
        self.margin = margin
        self.squared = squared

    def _pairwise_distances(self,inputs):
        dot_product = torch.matmul(inputs,torch.transpose(inputs,0,1))
        square_norm = torch.diag(dot_product)
        distances = torch.unsqueeze(square_norm,0)-2*dot_product + torch.unsqueeze(square_norm,1)

        distances = F.relu(distances)

        if not self.squared:
            distances = distances.clamp(min=1e-16)
            distances = torch.sqrt(distances)
        return distances

    def _get_anchor_positive_triplet_mask(self, labels):
        indices_equal = torch.eye(labels.shape[0])
        indices_not_equal = -1*indices_equal+1
        
        labels_equal = torch.eq(torch.unsqueeze(labels,0),torch.unsqueeze(labels,1))
        
        mask = indices_not_equal.byte().cuda() & labels_equal.byte().cuda()
        mask = mask.float().cuda()
        return mask

    def _get_anchor_negative_triplet_mask(self,labels):

        labels_equal = torch.eq(torch.unsqueeze(labels,0),torch.unsqueeze(labels,1))
        labels_equal = labels_equal.float().cuda()
        
        mask = -1*labels_equal + 1
        return mask

    def forward(self, input,targets):
        pairwise_dist = self._pairwise_distances(input)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(targets)
        mask_anchor_positive = mask_anchor_positive.float()

        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        hardest_positive_dist = anchor_positive_dist.max(1,True)[0]
        
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(targets)
        mask_anchor_negative = mask_anchor_negative.float()

        max_anchor_negative_dist = pairwise_dist.max(1,True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0-mask_anchor_negative)

        hardest_negative_dist = anchor_negative_dist.min(1,True)[0]

        dist = 1*hardest_positive_dist - 1*hardest_negative_dist + self.margin
        triplet_loss = F.relu(dist)

        return triplet_loss.float().sum()
