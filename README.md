# Trendyol E-Commerce Hackathon 2025 — Search Ranking Pipeline

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Polars-CD792C?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
</p>

> **Private Leaderboard Score: `0.64677`**

End-to-end search ranking pipeline for the **Trendyol E-Commerce Hackathon 2025** on Kaggle. Predicts click and order probabilities for `(user, search_term, session)` triplets to optimize fashion product ranking in search results.

**Competition:** [Trendyol E-Ticaret Hackathonu 2025](https://www.kaggle.com/competitions/trendyol-e-ticaret-hackathonu-2025-kaggle)

**Project Vision:** Search ranking at e-commerce scale. The click/order AUC optimization approach implemented here is the exact foundation of production recommendation systems at Trendyol, Amazon, Zalando, and similar platforms. Learning-to-rank is one of the highest-value ML applications in industry — a 1% improvement in ranking quality at scale translates to millions in additional revenue. This hackathon entry is the starting point for understanding how those systems are built.

---

## Competition Overview

| Field | Detail |
|---|---|
| Organizer | Trendyol |
| Platform | Kaggle |
| Domain | Fashion e-commerce search ranking |
| Session definition | (user, search_term, date) triplet |
| Task | Predict `clicked` and `ordered` probabilities per product |
| Metric | Weighted AUC of click AUC + order AUC (order weighted higher) |
| Private Score | **0.64677** |

---

## Problem Formulation

This is a **learning-to-rank** problem framed as binary classification. For each session, products are ranked by their predicted click/order probability. The model must surface the most relevant items at the top of search results.

```
Score = w₁ · AUC(click) + w₂ · AUC(order)    where w₂ > w₁
```

A score of 1.0 means the model perfectly ranks clicked and ordered items above all others within every session.

---

## Pipeline Architecture

```
Raw Parquet Files (products, sessions, interactions)
         │
         ▼
┌─────────────────────────┐
│  Data Loading & Merge   │  Polars lazy → eager for memory efficiency
│  Multi-file join        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Text Cleaning          │  Turkish character normalization, lowercasing
│  Label Encoding         │  Categorical → integer IDs
│  Downcast               │  int64 → int32/float32 to reduce memory
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Feature Engineering    │  User history aggregates, product popularity,
│                         │  session-level statistics, text similarity
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Model Training         │  TabNet (validation) + LightGBM (production)
│  Cross-validation       │  Session-stratified split
└──────────┬──────────────┘
           │
           ▼
      submission.csv
```

---

## Model Architecture

### TabNet (Validation)
- Attention-based tabular deep learning model
- Used for validation experiments and AUC scoring
- PyTorch backend

### LightGBM (Production Submission)
- Gradient-boosted trees for final predictions
- Binary cross-entropy loss for click/order targets
- Fast inference for large product catalogs

Both models output a probability score per `(session, product)` pair, which is used directly for ranking.

---

## Project Structure

```
Trendyol-E-Ticaret-Hackathonu/
├── main.ipynb         # Full pipeline: load → feature eng → train → submit
├── data/              # Local parquet data files (place competition files here)
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
python -m venv .venv
.\.venv\Scripts\activate    # Windows
source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

Place the competition parquet files into the `data/` directory. Open `main.ipynb` and run all cells in order. Output: `submission.csv` in the project root.

---

## Notes

- Data paths use the local `data/` directory (not Kaggle's `/kaggle/input/` paths).
- Polars is used for fast lazy evaluation on large parquet files before converting to pandas for model training.
- Diagnostic cells and runtime pip installs have been removed; all dependencies are in `requirements.txt`.

---

## Beyond the Hackathon

The 0.64677 private score is a baseline. These are the architectural steps that would take this pipeline toward production-grade ranking quality:

- [ ] **Two-Tower Neural Network** — Separate user embedding tower + item embedding tower, trained jointly with contrastive loss. Industry standard at Google (YouTube recommendations), Meta (Facebook feed), and Pinterest. Enables sub-millisecond candidate retrieval at millions-of-items scale via approximate nearest neighbor search (FAISS).
- [ ] **Session-Based Sequential Model** — GRU4Rec or SASRec to model the temporal click sequence within a session. "User clicked A, then B, then C → likely to click D." Standard LightGBM treats the session as a bag of features; sequential models capture the ORDER of interactions.
- [ ] **True Learning-to-Rank Loss** — Replace pointwise binary cross-entropy with LambdaRank or ListNet loss, which directly optimize ranking metrics (NDCG, MAP) rather than classification accuracy. This is the loss function used in production ranking systems.
- [ ] **Counterfactual Evaluation** — Implement Inverse Propensity Scoring (IPS) for offline evaluation. Standard AUC overestimates model quality because the training data is itself biased by the previous ranking system (position bias). IPS corrects for this.
- [ ] **Feature Store Simulation** — Redesign feature computation as a real-time feature store using Redis. Benchmark: can session-level and user-level features be served in <10ms to match production SLA requirements?
- [ ] **Semantic Search Layer** — Add a semantic retrieval stage before ranking: embed search terms and product titles using a Turkish sentence transformer, retrieve top-K candidates by cosine similarity, then rerank with the LightGBM model. This mimics the two-stage retrieve-then-rerank architecture used in production.
