"""Scan final population strategies for RL-like features.

RL signatures to detect:
- Q-table / value dict indexed by (state, action) or by (opponent, action)
- Learning rate / alpha / eta / discount factor / gamma
- Update rule: Q += lr * (reward - Q) or TD(0) delta
- Exploration: epsilon / softmax / UCB / exp3
- Bandit vs full RL: presence of state transition
"""
import json
import os
import re
from collections import defaultdict

base = r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline"

# Regex for RL signatures
RL_PATTERNS = {
    "Q_table":      re.compile(r'\bQ[_\[\.]', re.IGNORECASE),
    "V_table":      re.compile(r'\bV[_\[\.]', re.IGNORECASE),
    "learning_rate":re.compile(r'\b(learning_rate|lr|alpha|eta|_alpha|_eta|_lr)\b'),
    "discount":     re.compile(r'\b(gamma|discount|_gamma|gamma_)\b'),
    "epsilon":      re.compile(r'\b(epsilon|eps|_eps|exploration|explore)\b'),
    "td_error":     re.compile(r'\b(td_error|delta|target|reward)\b'),
    "softmax":      re.compile(r'\b(softmax|boltzmann)\b'),
    "ucb":          re.compile(r'\b(ucb|upper_confidence|confidence_bound)\b'),
    "exp3":         re.compile(r'\bexp3\b', re.IGNORECASE),
    "policy_grad":  re.compile(r'\b(policy[_ ]?grad|reinforce|baseline[_ ]?sub)\b', re.IGNORECASE),
    "experience":   re.compile(r'\b(replay|memory|experience_buffer)\b'),
}

def scan(code: str) -> dict:
    hits = defaultdict(int)
    for label, pat in RL_PATTERNS.items():
        hits[label] = len(pat.findall(code))
    return dict(hits)

def classify(hits: dict) -> str:
    """Rough classification based on hit counts."""
    q_score = hits.get("Q_table", 0) + hits.get("V_table", 0)
    lr_score = hits.get("learning_rate", 0)
    disc_score = hits.get("discount", 0)
    td_score = hits.get("td_error", 0)
    eps_score = hits.get("epsilon", 0)
    sm_score = hits.get("softmax", 0)

    if q_score >= 2 and (lr_score >= 1 or disc_score >= 1 or td_score >= 1):
        return "RL (Q-learning or V-learning)"
    if lr_score >= 1 and td_score >= 1 and eps_score >= 1:
        return "RL (bandit-style with explicit update)"
    if eps_score >= 1 and (lr_score >= 1 or td_score >= 1):
        return "Bandit/RL hybrid (epsilon-greedy + update)"
    if q_score >= 1:
        return "Value-table-like (Q/V naming, no update rule)"
    if sm_score >= 1 or eps_score >= 1:
        return "Stochastic (softmax/epsilon) but no learning"
    if lr_score >= 1 or td_score >= 1:
        return "Update rule present (no Q-table)"
    return "No RL signature"

# Scan all 3 seeds, both main (neutral prompt) and adversarial if available
DATASETS = [
    ("MAIN (neutral prompt, 100 gen × 3 seed × 1000 inter)", "LLM_v3_g100_1000inter_seed{}"),
    ("ADVERSARIAL (in progress)", "LLM_v3_g100_1000inter_ADVERSARIAL_seed{}"),
]
for label, sub in DATASETS:
    print("="*80)
    print(label)
    print("="*80)
    for s in [0, 1, 2]:
        folder = sub.format(s)
        path = os.path.join(base, folder, "evolutionary.json")
        if not os.path.exists(path):
            print(f"\n[seed {s}] (no evolutionary.json yet — adversarial still running)")
            continue
        with open(path) as f:
            j = json.load(f)
        traj = j["trajectory"]
        if not traj:
            print(f"\n[seed {s}] (empty trajectory)")
            continue
        final = traj[-1]
        gen_n = final["generation"]
        pop = final.get("population", [])
        print(f"\n[seed {s}] gen {gen_n}, {len(pop)} agents")
        # Classify each agent
        for a in pop:
            aid = a.get("agent_id", "?")
            code = a.get("code", "")
            hits = scan(code)
            cls = classify(hits)
            non_zero = {k: v for k, v in hits.items() if v > 0}
            print(f"  agent {aid:>2}: {cls}  hits={non_zero}")
