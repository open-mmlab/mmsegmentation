"""
MoE-related loss functions - MMSeg 1.x Version
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBalanceLoss(nn.Module):
    """MoE load balancing loss wrapper"""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, balance_loss):
        if balance_loss is None:
            return torch.tensor(0.0)
        return self.loss_weight * balance_loss


class ExpertDiversityLoss(nn.Module):
    """Expert diversity loss - encourages different expert representations"""

    def __init__(self,
                 loss_weight=0.01,
                 similarity_type='cosine',
                 normalize=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.similarity_type = similarity_type
        self.normalize = normalize

    def forward(self, expert_features_list):
        if expert_features_list is None or len(expert_features_list) < 2:
            return torch.tensor(
                0.0,
                device=(expert_features_list[0].device
                        if expert_features_list else None))

        expert_vectors = []
        for expert_feat in expert_features_list:
            pooled = F.adaptive_avg_pool2d(
                expert_feat, 1).squeeze(-1).squeeze(-1)

            if self.normalize:
                pooled = F.normalize(pooled, dim=1)

            n_samples = pooled.shape[0]
            expert_vector = pooled.sum(dim=0) / max(n_samples, 1)
            expert_vectors.append(expert_vector)

        num_experts = len(expert_vectors)
        total_sim = 0.0
        count = 0

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                if self.similarity_type == 'cosine':
                    sim = F.cosine_similarity(
                        expert_vectors[i].unsqueeze(0),
                        expert_vectors[j].unsqueeze(0),
                        dim=1
                    )
                elif self.similarity_type == 'l2':
                    dist = torch.norm(
                        expert_vectors[i] - expert_vectors[j], p=2)
                    sim = torch.clamp(1.0 - dist / 2.0, min=0.0, max=1.0)
                else:
                    raise ValueError(
                        f"Unknown similarity type: {self.similarity_type}")

                total_sim += sim
                count += 1

        if count > 0:
            avg_sim = total_sim / count
        else:
            avg_sim = torch.tensor(0.0, device=expert_vectors[0].device)

        diversity_loss = F.relu(avg_sim)
        return self.loss_weight * diversity_loss


class CombinedMoELoss(nn.Module):
    """Combined MoE loss: balance + diversity"""

    def __init__(self,
                 balance_weight=1.0,
                 diversity_weight=0.01,
                 diversity_similarity='cosine'):
        super().__init__()
        self.balance_loss = MoEBalanceLoss(loss_weight=balance_weight)
        self.diversity_loss = ExpertDiversityLoss(
            loss_weight=diversity_weight,
            similarity_type=diversity_similarity
        )

    def forward(self, balance_loss, expert_features_list):
        loss_balance = self.balance_loss(balance_loss)
        loss_diversity = self.diversity_loss(expert_features_list)

        total_loss = loss_balance + loss_diversity

        loss_dict = {
            'loss_moe_balance': loss_balance,
            'loss_moe_diversity': loss_diversity
        }

        return total_loss, loss_dict
