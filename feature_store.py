"""
Feature Store Simulation — Real-time feature computation with Redis

Production target:  session-level and user-level features served in <10ms

Architecture:
  Feature definitions → Redis hash store → batch precompute + real-time update
  Benchmark: measure feature retrieval latency against production SLA
"""

import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class FeatureStore:
    def __init__(self, host="localhost", port=6379, db=0, use_redis=True):
        self.use_redis = use_redis and HAS_REDIS
        if self.use_redis:
            self.r = redis.Redis(host=host, port=port, db=db,
                                 decode_responses=True)
            self.r.ping()
            print(f"[FeatureStore] Connected to Redis at {host}:{port}")
        else:
            self._local_store = {}
            print("[FeatureStore] Using local dictionary store (no Redis)")

        self._feature_defs = {}
        self._latencies = []

    def register_feature(self, name, compute_fn, dependencies=None,
                         ttl_seconds=3600, feature_type="float"):
        self._feature_defs[name] = {
            "fn": compute_fn,
            "deps": dependencies or [],
            "ttl": ttl_seconds,
            "type": feature_type,
        }

    def _cache_key(self, namespace, key, feature):
        return f"feat:{namespace}:{key}:{feature}"

    def precompute_batch(self, namespace, keys, features=None):
        if features is None:
            features = list(self._feature_defs.keys())

        batch = defaultdict(dict)
        for key in keys:
            for feat in features:
                if feat in self._feature_defs:
                    t0 = time.perf_counter()
                    val = self._feature_defs[feat]["fn"](key)
                    self._latencies.append(time.perf_counter() - t0)
                    batch[key][feat] = val

        if self.use_redis:
            pipe = self.r.pipeline()
            for key, feats in batch.items():
                ck = self._cache_key(namespace, key, "__all__")
                pipe.hset(ck, mapping=feats)
                ttl = max(d["ttl"] for d in self._feature_defs.values())
                pipe.expire(ck, ttl)
            pipe.execute()
        else:
            for key, feats in batch.items():
                self._local_store[self._cache_key(namespace, key, "__all__")] = feats

        return batch

    def get_features(self, namespace, key):
        ck = self._cache_key(namespace, key, "__all__")
        if self.use_redis:
            data = self.r.hgetall(ck)
        else:
            data = self._local_store.get(ck, {})
        return data

    def get_feature(self, namespace, key, feature):
        t0 = time.perf_counter()
        if self.use_redis:
            ck = self._cache_key(namespace, key, "__all__")
            val = self.r.hget(ck, feature)
        else:
            ck = self._cache_key(namespace, key, "__all__")
            data = self._local_store.get(ck, {})
            val = data.get(feature)
        self._latencies.append(time.perf_counter() - t0)
        return val

    def benchmark(self, n_queries=1000):
        if not self._latencies:
            print("[Benchmark] No queries measured yet.")
            return

        lat = np.array(self._latencies[-n_queries:]) * 1000
        print(f"[Benchmark] Last {len(lat)} queries:")
        print(f"  Mean:   {lat.mean():.2f} ms")
        print(f"  Median: {np.median(lat):.2f} ms")
        print(f"  P99:    {np.percentile(lat, 99):.2f} ms")
        print(f"  SLA target: <10 ms")
        print(f"  Within SLA: {(lat < 10).sum()}/{len(lat)} ({100*(lat<10).mean():.1f}%)")

        if lat.mean() > 10:
            print("  WARNING: Mean latency exceeds 10ms SLA target!")

    def clear(self):
        if self.use_redis:
            for k in self.r.keys("feat:*"):
                self.r.delete(k)
        else:
            self._local_store.clear()
        self._latencies.clear()


def example_user_features(user_id):
    return {
        "user_ctr": np.random.random(),
        "user_cart_rate": np.random.random(),
        "user_order_rate": np.random.random(),
        "user_active_days": int(np.random.exponential(30)),
    }


def example_session_features(session_id):
    return {
        "session_click_count": int(np.random.poisson(3)),
        "session_duration_sec": int(np.random.exponential(120)),
        "session_query_length": int(np.random.randint(1, 10)),
    }


if __name__ == "__main__":
    print("Feature Store Simulation — Trendyol Search Ranking")
    fs = FeatureStore(use_redis=False)

    fs.register_feature("user_ctr", lambda uid: np.random.random())
    fs.register_feature("session_click_count", lambda sid: int(np.random.poisson(3)))
    fs.register_feature("session_duration", lambda sid: int(np.random.exponential(120)))

    fs.precompute_batch("user", [f"user_{i}" for i in range(100)])
    fs.precompute_batch("session", [f"session_{i}" for i in range(100)])

    print("\nFeature retrieval benchmark:")
    for _ in range(1000):
        fs.get_feature("user", "user_42", "user_ctr")

    fs.benchmark()
