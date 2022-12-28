import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TreeTripletLoss(nn.Module):
    def __init__(self, num_classes, hiera_map, hiera_index, ignore_index=255):
        super(TreeTripletLoss, self).__init__()

        self.ignore_label = ignore_index
        self.num_classes = num_classes
        self.hiera_map = hiera_map
        self.hiera_index = hiera_index

    def forward(self, feats, labels=None, max_triplet=200):
        batch_size = feats.shape[0]
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        labels = labels.view(-1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(-1, feats.shape[-1])
        
        triplet_loss=0
        exist_classes = torch.unique(labels)
        exist_classes = [x for x in exist_classes if x != 255]
        class_count=0
        
        for ii in exist_classes:
            index_range = self.hiera_index[self.hiera_map[ii]]
            index_anchor = labels==ii
            index_pos = (labels>=index_range[0]) & (labels<index_range[-1]) & (~index_anchor)
            index_neg = (labels<index_range[0]) | (labels>=index_range[-1])
            
            min_size = min(torch.sum(index_anchor), torch.sum(index_pos), torch.sum(index_neg), max_triplet)
            
#             label_anchor = labels[index_anchor][:min_size]
#             label_pos = labels[index_pos][:min_size]
#             label_neg = labels[index_neg][:min_size]
            
            feats_anchor = feats[index_anchor][:min_size]
            feats_pos = feats[index_pos][:min_size]
            feats_neg = feats[index_neg][:min_size]
            
            distance = torch.zeros(min_size,2).cuda()
            distance[:,0:1] = 1-(feats_anchor*feats_pos).sum(1, True) 
            distance[:,1:2] = 1-(feats_anchor*feats_neg).sum(1, True) 
            
            # margin always 0.1 + (4-2)/4 since the hierarchy is three level
            # TODO: should include label of pos is the same as anchor
            margin = 0.6*torch.ones(min_size).cuda()
            
            tl = distance[:,0] - distance[:,1] + margin
            tl = F.relu(tl)

            if tl.size(0)>0:
                triplet_loss += tl.mean()
                class_count+=1
        if class_count==0:
            return None, torch.tensor([0]).cuda()
        triplet_loss /=class_count
        return triplet_loss, torch.tensor([class_count]).cuda()
    