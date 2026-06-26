"""
Semantic Search Layer — Two-stage retrieve-then-rerank

Stage 1: Embed search terms and product titles using a Turkish sentence transformer,
         retrieve top-K candidates by cosine similarity.

Stage 2: Rerank candidates with the LightGBM model.

This mimics the production retrieve-then-rerank architecture.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class SemanticSearch:
    def __init__(self, model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"):
        if HAS_ST:
            print(f"[SemanticSearch] Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            print("[SemanticSearch] WARNING: sentence-transformers not installed.")
            print("  Install: pip install sentence-transformers")
            self.model = None
        self.product_ids = None
        self.product_embeddings = None
        self.product_metadata = None

    def encode(self, texts):
        if self.model is None:
            return np.random.randn(len(texts), 384)
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def index_products(self, product_ids, titles, metadata=None):
        print(f"[SemanticSearch] Indexing {len(product_ids)} products...")
        self.product_ids = np.array(product_ids)
        self.product_embeddings = self.encode(titles)
        self.product_metadata = metadata
        print(f"  Embedding matrix: {self.product_embeddings.shape}")

    def retrieve(self, query, top_k=100, return_scores=True):
        if self.product_embeddings is None:
            raise ValueError("No products indexed. Call index_products() first.")

        query_emb = self.encode([query])
        scores = cosine_similarity(query_emb, self.product_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = {
            "indices": self.product_ids[top_indices],
            "scores": scores[top_indices],
        }
        return results

    def retrieve_batch(self, queries, top_k=100):
        query_embs = self.encode(queries)
        scores = cosine_similarity(query_embs, self.product_embeddings)

        results = []
        for i, q in enumerate(queries):
            top_idx = np.argsort(scores[i])[::-1][:top_k]
            results.append({
                "query": q,
                "candidate_ids": self.product_ids[top_idx],
                "scores": scores[i][top_idx],
            })
        return results


class RetrieveThenRerank:
    def __init__(self, semantic_search, rank_model, feature_fn):
        self.semantic = semantic_search
        self.rank_model = rank_model
        self.feature_fn = feature_fn

    def predict(self, queries, top_k_retrieval=100, top_n_final=10):
        candidates = self.semantic.retrieve_batch(queries, top_k=top_k_retrieval)
        results = []
        for c in candidates:
            features = self.feature_fn(c["candidate_ids"], c["query"])
            scores = self.rank_model.predict(features)
            top = np.argsort(scores)[::-1][:top_n_final]
            results.append({
                "query": c["query"],
                "ranked_ids": c["candidate_ids"][top],
                "ranked_scores": scores[top],
            })
        return results


if __name__ == "__main__":
    print("Semantic Search Layer — Retrieve-then-Rerank")
    searcher = SemanticSearch()

    dummy_ids = [f"product_{i}" for i in range(1000)]
    dummy_titles = [f"siyah elbise kadın" if i % 3 == 0 else
                    f"mavi kot pantolon erkek" if i % 3 == 1 else
                    f"kırmızı ayakkabı spor" for i in range(1000)]

    searcher.index_products(dummy_ids, dummy_titles)

    results = searcher.retrieve("siyah elbise", top_k=5)
    print(f"\nQuery: 'siyah elbise'")
    print(f"Top-5 candidates: {results['indices']}")
    print(f"Scores: {np.round(results['scores'][:5], 4)}")
