import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# source: https://colab.research.google.com/drive/1UK8BD3xvTpuSj75blz8hThfXksh19YbA?usp=sharing#scrollTo=GBNm6bbDT9J3
# source: https://www.youtube.com/watch?v=_1eKr4rbgRI
class NTXentLoss(nn.Module):
    def __init__(self, temperature: float):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self,
                out_1: Tensor,
                out_2: Tensor,
                use_unnormalized_feats: bool = False):
        """
        Input tensor size: [B, 128 or 256]
        Args:
            out_1 (): normalized tensor
            out_2 (): normalized tensor
            use_unnormalized_feats (): when enabled out's can be unnormalized...

        Returns: NT-XEnt Loss

        """
        # Concatenated tensor size: [2B, D]
        out = torch.cat([out_1, out_2], dim=0)
        n_samples = len(out)

        # Full similarity matrix
        # Covariance tensor size: [2B, 2B]
        if use_unnormalized_feats:
            cov = F.cosine_similarity(out.unsqueeze(1),
                                      out.unsqueeze(0), dim=-1)
        else:
            cov = torch.mm(out, out.t().contiguous())

        # Similarity tensor size: [2B, 2B]
        sim = torch.exp(cov / self.temperature)

        pos_mask = torch.eye(n_samples, device=sim.device).bool()
        mask = ~pos_mask
        pos_mask = pos_mask.roll(shifts=sim.shape[0] // 2, dims=0)
        # [2B, 2B - 1]
        neg_sim = sim.masked_select(mask).view(n_samples, -1)
        # Negative similarity summed tensor size: [2B]
        neg = neg_sim.sum(dim=-1)

        # Positive similarity tensor size: [2B]
        pos = sim[pos_mask]

        # Loss scalar value
        loss = -torch.log(pos / neg).mean()

        comb_sim = torch.cat(
            [pos[:, None], neg_sim],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        logging_metrics = {
            "acc_top1": (sim_argsort == 0).float().mean(),
            "acc_top5": (sim_argsort < 5).float().mean(),
            "acc_mean_pos": 1 + sim_argsort.float().mean()
        }

        return loss, logging_metrics


if __name__ == '__main__':
    loss_fn = NTXentLoss(temperature=0.07)
    z1 = F.normalize(torch.randn(5, 128))
    z2 = F.normalize(torch.randn(5, 128))
    loss = loss_fn(z1, z2)
    print(loss)
