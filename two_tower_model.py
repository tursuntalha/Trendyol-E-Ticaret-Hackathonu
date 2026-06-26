"""
Two-Tower Neural Network with FAISS Approximate Nearest Neighbor Search

Architecture:
  User Tower:  categorical features → embedding → MLP → user_embedding (d=64)
  Item Tower:  categorical features → embedding → MLP → item_embedding (d=64)

Training:  contrastive loss (NT-Xent) with in-batch negatives
Inference: FAISS index for sub-millisecond candidate retrieval
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_cats, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embed = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.cat_embed  = nn.Embedding(num_cats, embed_dim, padding_idx=0)

        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, embed_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, embed_dim),
        )

    def encode_users(self, user_ids, cat_ids=None):
        u = self.user_embed(user_ids)
        if cat_ids is not None:
            c = self.cat_embed(cat_ids).mean(dim=1)
            u = u + c
        return F.normalize(self.user_mlp(u), p=2, dim=1)

    def encode_items(self, item_ids, cat_ids=None):
        i = self.item_embed(item_ids)
        if cat_ids is not None:
            c = self.cat_embed(cat_ids).mean(dim=1)
            i = i + c
        return F.normalize(self.item_mlp(i), p=2, dim=1)

    def forward(self, user_ids, item_ids, user_cats=None, item_cats=None):
        u = self.encode_users(user_ids, user_cats)
        i = self.encode_items(item_ids, item_cats)
        return (u * i).sum(dim=1)


def contrastive_loss(user_emb, item_emb, temperature=0.07):
    logits = user_emb @ item_emb.T / temperature
    labels = torch.arange(len(user_emb), device=user_emb.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_j) / 2


def train_two_tower(model, loader, epochs=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            u, i = batch["user_id"].to(DEVICE), batch["item_id"].to(DEVICE)
            u_emb = model.encode_users(u)
            i_emb = model.encode_items(i)
            loss = contrastive_loss(u_emb, i_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")


def build_faiss_index(item_embeddings):
    import faiss
    dim = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_embeddings.astype(np.float32))
    return index


def retrieve_candidates(index, user_embedding, top_k=100):
    scores, indices = index.search(user_embedding.astype(np.float32), top_k)
    return indices, scores


if __name__ == "__main__":
    print("Two-Tower Model with FAISS — Trendyol Search Ranking")
    model = TwoTowerModel(num_users=21000, num_items=756000, num_cats=100).to(DEVICE)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}")
    print()
    print("Architecture:")
    print(f"  User tower:  Embed({21000}, 64) → MLP(64→128→64)")
    print(f"  Item tower:  Embed({756000}, 64) → MLP(64→128→64)")
    print(f"  Loss:        NT-Xent contrastive loss (temperature=0.07)")
    print(f"  Retrieval:   FAISS IndexFlatIP (inner product)")
    print()
    print("Usage:")
    print("  1. model = TwoTowerModel(num_users, num_items, num_cats)")
    print("  2. train_two_tower(model, dataloader)")
    print("  3. faiss_index = build_faiss_index(item_embs)")
    print("  4. candidates = retrieve_candidates(faiss_index, user_emb, top_k=100)")
