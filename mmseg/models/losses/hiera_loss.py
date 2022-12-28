import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def prepare_targets(targets, upper_ids, lower_ids):
    h, w = targets.shape[-2:]
    targets_middle = []
    for targets_per_image in targets:
        mask = torch.zeros((h, w), dtype=targets_per_image.dtype, device=targets_per_image.device)
        for ii in upper_ids:
            mask[targets_per_image==ii]=1
        for ii in lower_ids:
            mask[targets_per_image==ii]=2
        targets_middle.append(mask) 

    fore_ids = upper_ids+lower_ids
    targets_top = []
    for targets_per_image in targets:
        mask = torch.zeros((h, w), dtype=targets_per_image.dtype, device=targets_per_image.device)
        indices_fore = torch.logical_and(targets_per_image>=fore_ids[0], targets_per_image<=fore_ids[-1])
        mask[indices_fore]=1
        targets_top.append(mask) 
        
    return targets, torch.stack(targets_middle, dim=0), torch.stack(targets_top, dim=0)

def losses_hiera(predictions, targets, targets_middle, targets_top, num_classes, upper_ids, lower_ids, eps=1e-8, gamma=2):
    predictions = torch.sigmoid(predictions.float())
    void_indices = (targets==255)
    targets[void_indices]=0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2)
    targets_middle = F.one_hot(targets_middle, num_classes = 3).permute(0,3,1,2)
    targets_top = F.one_hot(targets_top, num_classes = 2).permute(0,3,1,2)

    MCMA = predictions[:,:num_classes,:,:]
    MCMB_back = torch.max(torch.cat([predictions[:,0:1,:,:],predictions[:,num_classes:num_classes+1,:,:]], dim=1), 1, True)[0]
    MCMB1 = torch.max(torch.cat([predictions[:,ii:ii+1,:,:] for ii in upper_ids]+[predictions[:,num_classes+1:num_classes+2,:,:]], dim=1), 1, True)[0]
    MCMB2 = torch.max(torch.cat([predictions[:,ii:ii+1,:,:] for ii in lower_ids]+[predictions[:,num_classes+2:num_classes+3,:,:]], dim=1), 1, True)[0]
    MCMB = torch.cat([MCMB_back, MCMB1, MCMB2], dim=1)
    MCMC_back = torch.max(torch.cat([MCMB_back ,predictions[:,num_classes+3:num_classes+4,:,:]], dim=1), 1, True)[0]
    MCMC1 = torch.max(torch.cat([MCMB1, MCMB2, predictions[:,num_classes+4:num_classes+5,:,:]], dim=1), 1, True)[0]
    MCMC = torch.cat([MCMC_back, MCMC1], dim=1)
    

    MCLC = predictions[:,num_classes+3:num_classes+5,:,:]
    MCLB_back = torch.min(torch.cat([MCLC[:,0:1,:,:], predictions[:,num_classes:num_classes+1,:,:]], dim=1), 1, True)[0]
    MCLB1 = torch.min(torch.cat([MCLC[:,1:2,:,:], predictions[:,num_classes+1:num_classes+2,:,:]], dim=1), 1, True)[0]
    MCLB2 = torch.min(torch.cat([MCLC[:,1:2,:,:], predictions[:,num_classes+2:num_classes+3,:,:]], dim=1), 1, True)[0]
    MCLB = torch.cat((MCLB_back, MCLB1, MCLB2), dim=1)
    MCLA_back = torch.min(torch.cat([predictions[:,0:1,:,:],MCLB[:,0:1,:,:]], dim=1), 1, True)[0]
    MCLA1 = torch.cat([torch.min(torch.cat([predictions[:,ii:ii+1,:,:],MCLB[:,1:2,:,:]], dim=1), 1, True)[0] for ii in upper_ids], dim=1)
    MCLA2 = torch.cat([torch.min(torch.cat([predictions[:,ii:ii+1,:,:],MCLB[:,2:3,:,:]], dim=1), 1, True)[0] for ii in lower_ids], dim=1)
    if len(upper_ids)>5:
        MCLA = torch.cat([MCLA_back, MCLA1[:,0:7,:,:], MCLA2[:,0:2,:,:], MCLA1[:,7:9,:,:], MCLA2[:,2:3,:,:], MCLA1[:,9:12,:,:], MCLA2[:,3:7,:,:]], dim=1)
    else:
        MCLA = torch.cat([MCLA_back, MCLA1, MCLA2], dim=1)

    valid_indices = (~void_indices).unsqueeze(1)
    num_valid = valid_indices.sum()
    loss = ((-targets[:,:num_classes,:,:]*torch.log(MCLA+eps)
            -(1-targets[:,:num_classes,:,:])*torch.log(1-MCMA+eps))
            *valid_indices).sum()/num_valid/num_classes
    loss+= ((-targets_middle[:,:3,:,:]*torch.log(MCLB+eps)
            -(1-targets_middle[:,:3,:,:])*torch.log(1-MCMB+eps))
            *valid_indices).sum()/num_valid/3
    loss+= ((-targets_top[:,:2,:,:]*torch.log(MCLC+eps)
            -(1-targets_top[:,:2,:,:])*torch.log(1-MCMC+eps))
            *valid_indices).sum()/num_valid/2

    return 5*loss

def losses_hiera_focal(predictions, targets, targets_middle, targets_top, num_classes, upper_ids, lower_ids, eps=1e-8, gamma=2):
    predictions = torch.sigmoid(predictions.float())
    void_indices = (targets==255)
    targets[void_indices]=0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2)
    targets_middle = F.one_hot(targets_middle, num_classes = 3).permute(0,3,1,2)
    targets_top = F.one_hot(targets_top, num_classes = 2).permute(0,3,1,2)

    MCMA = predictions[:,:num_classes,:,:]
    MCMB_back = torch.max(torch.cat([predictions[:,0:1,:,:],predictions[:,num_classes:num_classes+1,:,:]], dim=1), 1, True)[0]
    MCMB1 = torch.max(torch.cat([predictions[:,ii:ii+1,:,:] for ii in upper_ids]+[predictions[:,num_classes+1:num_classes+2,:,:]], dim=1), 1, True)[0]
    MCMB2 = torch.max(torch.cat([predictions[:,ii:ii+1,:,:] for ii in lower_ids]+[predictions[:,num_classes+2:num_classes+3,:,:]], dim=1), 1, True)[0]
    MCMB = torch.cat([MCMB_back, MCMB1, MCMB2], dim=1)
    MCMC_back = torch.max(torch.cat([MCMB_back ,predictions[:,num_classes+3:num_classes+4,:,:]], dim=1), 1, True)[0]
    MCMC1 = torch.max(torch.cat([MCMB1, MCMB2, predictions[:,num_classes+4:num_classes+5,:,:]], dim=1), 1, True)[0]
    MCMC = torch.cat([MCMC_back, MCMC1], dim=1)
    

    MCLC = predictions[:,num_classes+3:num_classes+5,:,:]
    MCLB_back = torch.min(torch.cat([MCLC[:,0:1,:,:], predictions[:,num_classes:num_classes+1,:,:]], dim=1), 1, True)[0]
    MCLB1 = torch.min(torch.cat([MCLC[:,1:2,:,:], predictions[:,num_classes+1:num_classes+2,:,:]], dim=1), 1, True)[0]
    MCLB2 = torch.min(torch.cat([MCLC[:,1:2,:,:], predictions[:,num_classes+2:num_classes+3,:,:]], dim=1), 1, True)[0]
    MCLB = torch.cat((MCLB_back, MCLB1, MCLB2), dim=1)
    MCLA_back = torch.min(torch.cat([predictions[:,0:1,:,:],MCLB[:,0:1,:,:]], dim=1), 1, True)[0]
    MCLA1 = torch.cat([torch.min(torch.cat([predictions[:,ii:ii+1,:,:],MCLB[:,1:2,:,:]], dim=1), 1, True)[0] for ii in upper_ids], dim=1)
    MCLA2 = torch.cat([torch.min(torch.cat([predictions[:,ii:ii+1,:,:],MCLB[:,2:3,:,:]], dim=1), 1, True)[0] for ii in lower_ids], dim=1)
    if len(upper_ids)>5:
        MCLA = torch.cat([MCLA_back, MCLA1[:,0:7,:,:], MCLA2[:,0:2,:,:], MCLA1[:,7:9,:,:], MCLA2[:,2:3,:,:], MCLA1[:,9:12,:,:], MCLA2[:,3:7,:,:]], dim=1)
    else:
        MCLA = torch.cat([MCLA_back, MCLA1, MCLA2], dim=1)

    valid_indices = (~void_indices).unsqueeze(1)
    num_valid = valid_indices.sum()
    loss = ((-targets[:,:num_classes,:,:]*torch.pow((1.0-MCLA),gamma)*torch.log(MCLA+eps)
             -(1.0-targets[:,:num_classes,:,:])*torch.pow(MCMA, gamma)*torch.log(1.0-MCMA+eps))
             *valid_indices).sum()/num_valid/num_classes
    loss+= ((-targets_middle*torch.pow((1.0-MCLB), gamma)*torch.log(MCLB+eps)
             -(1.0-targets_middle)*torch.pow(MCMB, gamma)*torch.log(1-MCMB+eps))
             *valid_indices).sum()/num_valid/3
    loss+= ((-targets_top*torch.pow((1-MCLC), gamma)*torch.log(MCLC+eps)
             -(1.0-targets_top)*torch.pow(MCMC, gamma)*torch.log(1-MCMC+eps))
             *valid_indices).sum()/num_valid/2

    return loss


@LOSSES.register_module()
class HieraLoss2(nn.Module):

    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 loss_weight=1.0):
        super(HieraLoss2, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                **kwargs):
        """Forward function."""
        upper_ids = [1,2,3,4]
        lower_ids = [5,6]
        targets, targets_middle, targets_top = prepare_targets(label, upper_ids, lower_ids)
        loss = losses_hiera_focal(cls_score, targets, targets_middle, targets_top, self.num_classes, upper_ids, lower_ids)
        return loss*self.loss_weight
