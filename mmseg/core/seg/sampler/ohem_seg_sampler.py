import torch
import torch.nn.functional as F

from ..registry import SEG_SAMPLERS
from .base_seg_sampler import BasSegSampler


@SEG_SAMPLERS.register_module
class OHEMSegSampler(BasSegSampler):

    def __init__(self, thresh=0.7, min_kept=100000, ignore_index=255):
        super(OHEMSegSampler, self).__init__()
        assert min_kept > 1
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index

    def sample(self, seg_logit, seg_label):
        """

        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)

        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)

        """
        with torch.no_grad():
            assert seg_logit.shape[2:] == seg_label.shape[2:]
            assert seg_label.shape[1] == 1
            seg_label = seg_label.squeeze(1).long()
            batch_kept = self.min_kept * seg_label.size(0)
            seg_prob = F.softmax(seg_logit, dim=1)
            mask = seg_label.contiguous().view(-1, ) != self.ignore_index

            tmp_seg_label = seg_label.clone()
            tmp_seg_label[tmp_seg_label == self.ignore_index] = 0
            seg_prob = seg_prob.gather(1, tmp_seg_label.unsqueeze(1))
            sort_prob, sort_indices = seg_prob.contiguous().view(
                -1, )[mask].contiguous().sort()

            if sort_prob.numel() > 0:
                min_threshold = sort_prob[min(batch_kept,
                                              sort_prob.numel() - 1)]
            else:
                min_threshold = 0.0
            threshold = max(min_threshold, self.thresh)

            seg_weight = seg_logit.new_ones(size=seg_label.size())
            seg_weight = seg_weight.view(-1)
            seg_weight[mask][sort_prob < threshold] = 0.
            seg_weight = seg_weight.view_as(seg_label)

            return seg_weight
