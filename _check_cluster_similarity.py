"""Check whether strategies assigned to the same KMeans cluster are truly similar.

Pipeline (last generation only):
  1. load final_population of LLM_v3_fermi_z_v3_g100_1000inter_seed2
  2. TF-IDF embed each agent's strategy code
  3. KMeans with K chosen by silhouette (default) or given
  4. for every cluster: pairwise cosine similarity of cluster members
     -> report min/mean/max, the closest & most distant pair, and a
        per-pair similarity matrix, so we can see whether the cluster
        is a real "family" or just a leftover bin.

Usage:
  uv run python _check_cluster_similarity.py [--k 6]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
JSON_PATH = (
    ROOT
    / "results"
    / "quantitative_baseline"
    / "LLM_v3_fermi_z_v3_g100_1000inter_seed2"
    / "evolutionary.json"
)

VECTORIZER = TfidfVectorizer(
    token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)


def fingerprint(code: str) -> str:
    if "q_table" in code and "_get_state" in code:
        return "q-learning"
    if "opponent_models" in code or "recipient_models" in code:
        return "opponent-model"
    if "forgiveness_level" in code and "retaliation_threshold" in code:
        return "rep-forgive"
    if "reputation_scores" in code and "opponent_last_actions" in code:
        return "rep-history"
    if "coop_history" in code and "own_actions" in code:
        return "coop-history"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="fixed K; default = silhouette-best")
    args = ap.parse_args()

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    pop = data["final_population"]
    pop = sorted(pop, key=lambda a: a["agent_id"])
    codes = [a["code"] for a in pop]
    ids = [a["agent_id"] for a in pop]
    print(f"final population: {len(pop)} agents")

    X = VECTORIZER.fit_transform(codes)
    cos = 1.0 - pairwise_distances(X, metric="cosine")  # cosine similarity matrix

    if args.k is None:
        best_k, best_s = None, -1.0
        for k in range(2, min(8, len(pop) - 1) + 1):
            km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
            s = silhouette_score(X, km.labels_)
            print(f"  K={k}: silhouette={s:.4f}")
            if s > best_s:
                best_k, best_s = k, s
        k = best_k
        print(f"-> chosen K={k} (silhouette={best_s:.4f})\n")
    else:
        k = args.k
        print(f"-> fixed K={k}\n")

    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    labels = km.labels_

    print("=" * 78)
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) < 2:
            print(f"\n[cluster {c}] 1 agent (no pair to compare)")
            for i in members:
                print(f"   agent {ids[i]:>3}  {fingerprint(codes[i])}")
            continue

        sub = cos[np.ix_(members, members)]
        iu = np.triu_indices(len(members), 1)
        sims = sub[iu]
        pairs = [(sims[t], members[i], members[j])
                 for t, (i, j) in enumerate(zip(*iu))]
        pairs.sort(key=lambda t: -t[0])
        closest = pairs[0]
        farthest = pairs[-1]

        print(f"\n[cluster {c}] {len(members)} agents  "
              f"pairwise cos-sim: min={sims.min():.3f} mean={sims.mean():.3f} max={sims.max():.3f}")
        print("   members:")
        for i in members:
            print(f"     agent {ids[i]:>3}  fit={pop[i]['fitness']:>4}  "
                  f"coop={pop[i]['cooperation_rate']:.3f}  {fingerprint(codes[i])}")
        print(f"   closest pair : agent {ids[closest[1]]} vs agent {ids[closest[2]]}  sim={closest[0]:.3f}")
        print(f"   farthest pair: agent {ids[farthest[1]]} vs agent {ids[farthest[2]]}  sim={farthest[0]:.3f}")

    # full per-pair table, grouped by cluster
    print("\n" + "=" * 78)
    print("full pairwise cosine-similarity table (agents sorted by cluster)")
    order = np.argsort(labels)
    print("       " + "".join(f"{ids[i]:>6}" for i in order))
    for a in order:
        row = " ".join(f"{cos[a, b]:6.2f}" for b in order)
        print(f"a{ids[a]:>4} " + " ".join(f"{cos[a, b]:6.2f}" for b in order))


if __name__ == "__main__":
    main()
