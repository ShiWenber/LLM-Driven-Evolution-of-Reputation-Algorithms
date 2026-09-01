"""Cluster the final-generation strategies by TF-IDF + K-means and dump per-cluster signatures.

Long-term analysis script (successor of the temporary _tmp_cluster_features.py).

For the last generation only, this script:
  1. loads final_population of an evolutionary.json
  2. TF-IDF embeds each agent's strategy code (unigram + bigram, sublinear tf)
  3. chooses K adaptively by maximizing the silhouette score (or uses --k)
  4. for every cluster, prints its members (agent_id / fitness / cooperation
     rate), a structural fingerprint, and the top TF-IDF identifier features,
     so the "commonality" of each cluster can be read off directly.

Usage:
  uv run python analyze_final_clusters.py
  uv run python analyze_final_clusters.py --json results/quantitative_baseline/LLM_v3_fermi_z_v3_g100_1000inter_seed2/evolutionary.json --k 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = (
    ROOT
    / "results"
    / "quantitative_baseline"
    / "LLM_v3_fermi_z_v3_g100_1000inter_seed2"
    / "evolutionary.json"
)

# Structural keywords used to fingerprint a strategy's mechanism family.
KEYS = [
    "q_table", "_get_state", "epsilon", "learning_rate", "alpha",
    "opponent_models", "recipient_models", "forgiveness_level",
    "retaliation_threshold", "reputation_scores", "opponent_last_actions",
    "coop_history", "own_actions", "reputation_history", "defect",
    "retaliate", "forgive", "threshold",
]


def fingerprint(code: str) -> str:
    """Map strategy code to a coarse archetype (mirrors the paper classifier)."""
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


def pick_default_json(root: Path) -> Path:
    if DEFAULT_JSON.exists():
        return DEFAULT_JSON
    candidates = sorted(root.rglob("evolutionary.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No evolutionary.json found; pass --json explicitly.")
    return candidates[-1]


def choose_k(X, min_k: int, max_k: int, seed: int) -> int:
    """Pick K maximizing silhouette in [min_k, max_k]."""
    best_k, best_s = None, -1.0
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        s = silhouette_score(X, km.labels_)
        print(f"  K={k}: silhouette={s:.4f}")
        if s > best_s:
            best_k, best_s = k, s
    print(f"-> chosen K={best_k} (silhouette={best_s:.4f})")
    return best_k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, default=None, help="path to evolutionary.json")
    ap.add_argument("--k", type=int, default=None, help="fixed K; default = silhouette-best")
    ap.add_argument("--min-k", type=int, default=2, help="minimum K for the sweep")
    ap.add_argument("--max-k", type=int, default=8, help="maximum K for the sweep")
    ap.add_argument("--seed", type=int, default=42, help="K-means random seed")
    ap.add_argument("--top", type=int, default=8, help="number of top TF-IDF features to show")
    args = ap.parse_args()

    json_path = Path(args.json) if args.json else pick_default_json(ROOT)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    pop = sorted(data["final_population"], key=lambda a: a["agent_id"])
    codes = [a["code"] for a in pop]
    print(f"final population: {len(pop)} agents")

    vec = TfidfVectorizer(
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X = vec.fit_transform(codes)
    feat = vec.get_feature_names_out()

    if args.k is None:
        max_k = min(args.max_k, len(pop) - 1)
        k = choose_k(X, args.min_k, max_k, args.seed)
    else:
        k = args.k
        print(f"-> fixed K={k}")

    km = KMeans(n_clusters=k, n_init=20, random_state=args.seed).fit(X)
    labels = km.labels_

    for c in sorted(set(labels)):
        mem = np.where(labels == c)[0]
        print(f"\n===== cluster {c} ({len(mem)} agents) =====")
        for i in mem:
            code = codes[i]
            present = [kw for kw in KEYS if kw in code]
            row = X[i].toarray().ravel()
            idx = np.argsort(row)[::-1][: args.top]
            top = [feat[j] for j in idx if row[j] > 0]
            aid = pop[i]["agent_id"]
            fit = pop[i]["fitness"]
            coop = pop[i]["cooperation_rate"]
            print(f"  agent {aid:>3} fit={fit:>3} coop={coop:.3f} "
                  f"| fingerprint={fingerprint(code)} "
                  f"| feat={present} | top-tfidf={top}")


if __name__ == "__main__":
    main()
