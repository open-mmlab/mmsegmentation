import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .hiera_loss import prepare_targets, losses_hiera_focal, losses_hiera
from .cross_entropy_loss import CrossEntropyLoss

TORCH_VERSION = torch.__version__[:3]

_euler_num = 2.718281828  # euler number
_pi = 3.14159265  # pi
_ln_2_pi = 1.837877  # ln(2 * pi)
_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
_POS_ALPHA = 1e-3  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel

def map_get_pairs(labels_4D, probs_4D, radius=3, is_combine=True):
    label_shape = labels_4D.size()
    h, w = label_shape[2], label_shape[3]
    new_h, new_w = h - (radius - 1), w - (radius - 1)
    la_ns = []
    pr_ns = []
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
            pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    if is_combine:
        pair_ns = la_ns + pr_ns
        p_vectors = torch.stack(pair_ns, dim=2)
        return p_vectors
    else:
        la_vectors = torch.stack(la_ns, dim=2)
        pr_vectors = torch.stack(pr_ns, dim=2)
        return la_vectors, pr_vectors

def log_det_by_cholesky(matrix):
    chol = torch.cholesky(matrix)
    return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)


class TreeTripletLoss(nn.Module):
    def __init__(self, num_classes, upper_ids, lower_ids, ignore_index=255):
        super(TreeTripletLoss, self).__init__()

        self.ignore_label = ignore_index
        self.num_classes = num_classes
        self.upper_ids = upper_ids
        self.lower_ids = lower_ids

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
        exist_classes = [x for x in exist_classes if x != 255 and x!=0]
        class_count=0
        
        for ii in exist_classes:
            index_anchor = labels==ii
            if ii in self.upper_ids:
                label_pos = self.upper_ids.copy()
                label_neg = self.lower_ids.copy()
            else:
                label_pos = self.lower_ids.copy()
                label_neg = self.upper_ids.copy()
            label_pos.remove(ii)
            index_pos = torch.zeros_like(index_anchor)
            index_neg = torch.zeros_like(index_anchor)
            for pos_l in label_pos:
                index_pos += labels==pos_l
            for neg_l in label_neg:
                index_neg += labels==neg_l
            
            min_size = min(torch.sum(index_anchor), torch.sum(index_pos), torch.sum(index_neg), max_triplet)
            
            feats_anchor = feats[index_anchor][:min_size]
            feats_pos = feats[index_pos][:min_size]
            feats_neg = feats[index_neg][:min_size]
            
            distance = torch.zeros(min_size,2).cuda()
            distance[:,0:1] = 1-(feats_anchor*feats_pos).sum(1, True) 
            distance[:,1:2] = 1-(feats_anchor*feats_neg).sum(1, True) 
            
            # margin always 0.1 + (4-2)/4 since the hierarchy is three level
            # TODO: should include label of pos is the same as anchor, i.e. margin=0.1
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

    
@LOSSES.register_module()
class RMIHieraTripletLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 num_classes=7,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 loss_weight=1.0):
        super(RMIHieraTripletLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.num_classes = num_classes
        self.rmi_radius = rmi_radius
        assert self.rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_pool_way = rmi_pool_way
        assert self.rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        assert self.rmi_pool_size == self.rmi_pool_stride
        
        if self.num_classes>15:
            self.upper_ids=[1,2,3,4,5,6,7,10,11,13,14,15]
            self.lower_ids=[8,9,12,16,17,18,19]
        else:
            self.upper_ids = [1,2,3,4]
            self.lower_ids = [5,6]

        self.weight_lambda = loss_weight_lambda
        self.loss_weight = loss_weight
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        self.ce = CrossEntropyLoss()
        self.treetripletloss = TreeTripletLoss(self.num_classes, self.upper_ids, self.lower_ids)

    def forward(self,
                step,
                embedding,
                cls_score_before,
                cls_score,
                label,
                weight=None,
                **kwargs):
        
        targets, targets_middle, targets_top = prepare_targets(label, self.upper_ids, self.lower_ids)
        hiera_loss = losses_hiera_focal(cls_score, targets, targets_middle, targets_top, self.num_classes, self.upper_ids, self.lower_ids)
        
        void_indices = (targets==255)
        targets[void_indices]=0
        valid_indices = (targets!=255)
        targets_ = F.one_hot(targets, num_classes=self.num_classes).permute(0,3,1,2)
        targets_middle_ = F.one_hot(targets_middle, num_classes = 3).permute(0,3,1,2)
        targets_top_ = F.one_hot(targets_top, num_classes = 2).permute(0,3,1,2)
        new_targets = torch.cat([targets_, targets_middle_, targets_top_], dim=1)   
        
        rmi_loss = self.forward_sigmoid(cls_score, new_targets, valid_indices, self.num_classes)
        ce_loss = self.ce(cls_score[:,:-5],label)
        ce_loss2 = self.ce(cls_score[:,-5:-2],targets_middle)
        ce_loss3 = self.ce(cls_score[:,-2:],targets_top)
        
        loss = 0.5*rmi_loss + 0.5*hiera_loss + ce_loss + ce_loss2 + ce_loss3
        
        loss_triplet, class_count = self.treetripletloss(embedding, label)
        class_counts = [torch.ones_like(class_count) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(class_counts, class_count, async_op=False)
        class_counts = torch.cat(class_counts, dim=0)

        if torch.distributed.get_world_size()==torch.nonzero(class_counts, as_tuple=False).size(0):
            all_step = 160000 if len(self.upper_ids)>5 else 60000
            factor = 1/4*(1+torch.cos(torch.tensor((step.item()-all_step)/all_step*math.pi))) if step.item()<all_step else 0.5
            loss+=factor*loss_triplet
            
        return loss

    def forward_sigmoid(self, logits_4D, labels_4D, label_mask_3D, num_classes):
        # valid label
        valid_onehot_labels_4D = labels_4D.float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=1)
        valid_onehot_labels_4D.requires_grad_(False)
        
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D, num_classes)
        final_loss = rmi_loss * self.weight_lambda
        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D, num_classes):
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]
        la_vectors, pr_vectors = map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)
        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        rmi_now = 0.5 * log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        rmi_per_class = rmi_now.view([-1, num_classes+5]).mean(dim=0).float()
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))
        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss

