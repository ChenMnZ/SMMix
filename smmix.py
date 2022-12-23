import torch
import torch.nn as nn
import math
from timm.data.mixup import Mixup, cutmix_bbox_and_lam, one_hot
import numpy as np
import torch
import random

def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    mixed_target = y1 * lam + y2 * (1. - lam)
    return mixed_target, y1, y2


def batch_index_generate(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        return idx.reshape(-1)
    elif len(x.size()) == 2:
        B, N = x.size()
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        return idx
    else:
        raise NotImplementedError


class SMMix(Mixup):
    """ act like Mixup()
        Mixup/SMMix that applies different params to each element or whole batch, where per-batch is set as default

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
        min_side_ratio (int): lower bound on uniform sampling
        max_side_ratio (int): upper bound on uniform sampling
        side: side length of attention map in image shape
    """
    def __init__(self, mixup_alpha=1., prob=1.0, switch_prob=0.5,
                 mode='batch',  label_smoothing=0.1, num_classes=1000, min_side_ratio=0.25, max_side_ratio=0.75, side=14):
        self.mixup_alpha = mixup_alpha
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
        
        self.side = side
        self.min_side = int(side*min_side_ratio)
        self.max_side = int(side*max_side_ratio)
        self.rectangle_size_list = []
        for i in range(self.min_side, self.max_side+1):
            self.rectangle_size_list.append((i,i))


    def smmix(self, inputs,attn,rectangle_size):
        inputs_side = inputs.shape[2]
        patch_size = inputs_side//self.side
        inputs = torch.nn.functional.unfold(inputs,patch_size,stride=patch_size).transpose(1,2)
        source_image = inputs.flip(0)
        
        attn = attn.reshape(-1,self.side,self.side).unsqueeze(1)
        # aggregating the image attention score of each candidate region
        rectangle_attn = torch.nn.functional.unfold(attn,rectangle_size,stride=1)
        rectangle_attn = rectangle_attn.sum(dim=1)
        
        # generating path index of mixed regions
        min_region_center_index = torch.argmin(rectangle_attn,dim=1)
        max_region_center_index = torch.argmax(rectangle_attn,dim=1)
        min_region_index = self.index_translate(min_region_center_index,rectangle_size, token_size=(self.side,self.side))
        max_region_index = self.index_translate(max_region_center_index,rectangle_size, token_size=(self.side,self.side))
        min_region_index =  batch_index_generate(inputs,min_region_index)
        max_region_index =  batch_index_generate(inputs,max_region_index.flip(0))
        
        # image mixing 
        B,N,C = inputs.shape
        inputs = inputs.reshape(B*N, C)
        source_image = source_image.reshape(B*N, C)
        inputs[min_region_index] = source_image[max_region_index]
        inputs = inputs.reshape(B,N,C)
        inputs = torch.nn.functional.fold(inputs.transpose(1,2),inputs_side,patch_size,stride=patch_size)

        # source_mask: indicate the source region in mixed image
        # target_mask: indicate the target region in mixed image
        source_mask = torch.zeros_like(attn).bool()
        source_mask = source_mask.reshape(-1)
        source_mask[min_region_index] = True
        source_mask = source_mask.reshape(B,N)
        target_mask = ~source_mask

        return inputs, target_mask, source_mask

    def index_translate(self,rectangle_index, rectangle_size=(3,3), token_size=(7,7)):
        total_index = torch.arange(token_size[0]*token_size[1]).reshape(1,1,token_size[0],token_size[1]).cuda()
        total_index_list = torch.nn.functional.unfold(total_index.float(),rectangle_size,stride=1).transpose(1,2).long()
        sequence_index=total_index_list.index_select(dim=1,index=rectangle_index).squeeze(0)
        return sequence_index


    def __call__(self, x, target, motivat_model=None):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        assert self.mode == 'batch', 'Mixup mode is batch by default'

        use_smmix = np.random.rand() < self.switch_prob
        if use_smmix:
            with torch.no_grad():
                motivat_model.eval()
                un_mixed_prediction_distribution,attn = motivat_model(x)
                motivat_model.train()
            rectangle_size = random.choice(self.rectangle_size_list)
            # Following the original Mixup code of Timm codebase, lam indicates the area ratio of target image, which is equal to the (1-\lambda) in the paper.
            lam = (self.side**2-rectangle_size[0]*rectangle_size[1])/self.side**2
            x,target_mask, source_mask = self.smmix(x, attn, rectangle_size)
            mixed_target, target_target, source_target= mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device) # tuple or tensor

            target_prediction_distribution = lam*un_mixed_prediction_distribution
            source_prediction_distributiont = (1-lam)*un_mixed_prediction_distribution.flip(0)
            mixed_prediction_distribution = target_prediction_distribution + source_prediction_distributiont
            
            return x, mixed_target, (target_target, source_target, target_mask, source_mask, mixed_prediction_distribution)

        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            if not lam == 1:
                lam = float(lam)
                x_flipped = x.flip(0).mul_(1. - lam)
                x.mul_(lam).add_(x_flipped)
            mixed_target, _, _= mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device) # tuple or tensor

            return x, mixed_target, None
        
        


