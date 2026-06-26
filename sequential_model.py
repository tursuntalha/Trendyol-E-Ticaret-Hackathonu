"""
Session-Based Sequential Model — GRU4Rec for temporal click sequence modeling

Captures the ORDER of user clicks within a session:
  "User clicked A → B → C → likely to click D"

Architecture:
  Embedding Layer → GRU → Dense → Softmax over items
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRU4Rec(nn.Module):
    def __init__(self, num_items, hidden_size=128, embed_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=0.3 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, num_items + 1)

    def forward(self, seq, lengths=None):
        emb = self.embedding(seq)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
            last = out.gather(1, idx).squeeze(1)
        else:
            _, hidden = self.gru(emb)
            last = hidden[-1]
        return self.fc(last)

    def predict_next(self, seq, lengths=None, top_k=10):
        logits = self.forward(seq, lengths)
        probs = F.softmax(logits, dim=-1)
        scores, indices = torch.topk(probs, top_k, dim=-1)
        return indices.cpu().numpy(), scores.cpu().numpy()


def train_gru4rec(model, loader, epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            seq = batch["seq"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            lengths = batch.get("lengths")
            logits = model.forward(seq, lengths)
            loss = F.cross_entropy(logits, target, ignore_index=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")


if __name__ == "__main__":
    print("GRU4Rec — Session-Based Sequential Model")
    model = GRU4Rec(num_items=756000, hidden_size=128, embed_dim=64).to(DEVICE)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}")
    print()
    print("Architecture:")
    print(f"  Embedding:  {756000+1} → 64")
    print(f"  GRU:        64 → 128  (1 layer)")
    print(f"  Output:     128 → {756000+1} (softmax over all items)")
    print()
    print("Usage:")
    print("  1. model = GRU4Rec(num_items)")
    print("  2. train_gru4rec(model, dataloader)")
    print("  3. next_items, scores = model.predict_next(seq, top_k=10)")
