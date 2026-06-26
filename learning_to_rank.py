"""
True Learning-to-Rank Loss — LambdaRank & ListNet

Replaces pointwise binary cross-entropy with ranking losses that directly
optimize NDCG / MAP rather than classification accuracy.

LambdaRank:  Lambda-gradient for pairwise ranking
ListNet:     Top-1 probability cross-entropy (Lu & Li, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dcg(scores, k=None):
    if k is None:
        k = scores.size(-1)
    scores = scores[:, :k]
    denom = torch.log2(torch.arange(2, k + 2, device=scores.device).float())
    return (scores / denom).sum(dim=-1)


def ndcg(scores, targets, k=None):
    ideal, _ = targets.sort(descending=True)
    return dcg(scores, k) / (dcg(ideal, k).clamp(min=1e-10))


def lambdarank_loss(scores, targets, sigma=1.0):
    """
    LambdaRank loss (Burges et al. 2006)
    Approximates NDCG optimization via pairwise gradients.
    """
    n = scores.size(-1)
    device = scores.device

    pairs = torch.combinations(torch.arange(n), r=2)
    i, j = pairs[:, 0], pairs[:, 1]

    si = scores[:, i]
    sj = scores[:, j]
    ti = targets[:, i]
    tj = targets[:, j]

    delta_ndcg = torch.abs(
        (2 ** ti - 2 ** tj) * (1.0 / torch.log2((i + 2).float()) - 1.0 / torch.log2((j + 2).float()))
    )

    sij = si - sj
    lambda_weight = delta_ndcg * (1.0 / (1.0 + torch.exp(-sigma * sij)))
    loss = (lambda_weight * torch.log(1 + torch.exp(-sigma * sij))).sum(dim=-1)

    return loss.mean()


def listnet_loss(scores, targets):
    """
    ListNet loss (Lu & Li, 2023) — Top-1 probability cross-entropy
    """
    prob_scores = F.softmax(scores, dim=-1)
    prob_targets = F.softmax(targets.float(), dim=-1)
    loss = -(prob_targets * torch.log(prob_scores.clamp(min=1e-15))).sum(dim=-1)
    return loss.mean()


def ranknet_loss(scores, targets):
    n = scores.size(-1)
    device = scores.device
    pairs = torch.combinations(torch.arange(n), r=2)
    i, j = pairs[:, 0], pairs[:, 1]
    sij = scores[:, i] - scores[:, j]
    pij = torch.sigmoid(sij)
    tij = (targets[:, i] > targets[:, j]).float()
    return F.binary_cross_entropy(pij, tij)


class LambdaRankModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_rank_model(model, loader, loss_fn="lambdarank", epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_map = {"lambdarank": lambdarank_loss, "listnet": listnet_loss, "ranknet": ranknet_loss}
    criterion = loss_map[loss_fn]
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch["features"].float()
            y = batch["targets"].float()
            scores = model(x)
            scores = scores.view(batch.get("group_size", scores.size(0)), -1)
            y = y.view(batch.get("group_size", y.size(0)), -1)
            loss = criterion(scores, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")


if __name__ == "__main__":
    print("Learning-to-Rank Loss Functions — Trendyol Search Ranking")
    print()
    print("Available losses:")
    print(f"  1. LambdaRank  — Pairwise NDCG approximation  (lambdarank_loss)")
    print(f"  2. ListNet     — Top-1 probability CE         (listnet_loss)")
    print(f"  3. RankNet     — Binary pairwise cross-entropy (ranknet_loss)")
    print()
    print("Usage:")
    print("  model = LambdaRankModel(input_dim)")
    print("  train_rank_model(model, dataloader, loss_fn='lambdarank')")
