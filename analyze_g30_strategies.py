"""Extract and analyze G=30 final-population strategies.

For each (obs, seed), show the final-pop strategies with their cooperation rate.
Then classify each strategy into 9 archetypes using the regex-based classifier.
Finally, show a few interesting strategy code samples.
"""
import json, re
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
G30 = ROOT / "results" / "exp12_g30_n15"
OUT = ROOT / "results" / "g30_strategies"
OUT.mkdir(exist_ok=True)

OBS = ["private", "partial_0.3", "partial_0.7", "full"]
SEEDS = [0, 1, 2]


# Regex-based classifier (mirrors the paper's Appendix A)
def classify(code: str) -> str:
    has_thresh = bool(re.search(r"\b(?:return\s+.*>=|>=|threshold|recipient_reputation\s*>=\s*-?\d)", code))
    has_history = "my_history" in code
    has_observe = "observation" in code
    has_round = "round_num" in code
    has_random = "random.random" in code
    if has_random:
        return "Random"
    if has_observe and has_thresh and has_history:
        return "Hybrid(F2b)"  # IS + history
    if has_observe and has_thresh and not has_history:
        return "ImageScoring"
    if has_observe and has_history and not has_thresh:
        return "DirectExperience"
    if has_thresh and not has_observe:
        return "ThresholdOnly"
    if has_round and not has_thresh:
        return "RoundDependent"
    if "return True" in code and not has_thresh and not has_random:
        return "ALLC"
    if "return False" in code and not has_thresh and not has_random and not has_observe:
        return "ALLD"
    return "Other"


def load_final(d):
    for f in (d / "evolutionary.json",):
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                pop = data.get("final_population", [])
                if pop:
                    return pop
            except Exception:
                pass
    for f in d.glob("evo_*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            pop = data.get("final_population", [])
            if pop:
                return pop
        except Exception:
            pass
    return []


# Per-trial: list all 15 strategies with classification + cooperation
print("=" * 100)
print("Per-trial final-population strategies (15 each, 12 trials = 180 total)")
print("=" * 100)

all_strategies = []
trial_summary = []
for obs in OBS:
    for seed in SEEDS:
        pop = load_final(G30 / f"{obs}_seed{seed}")
        if not pop:
            print(f"\n[SKIP] {obs}_seed{seed}: no data")
            continue
        classifications = []
        for s in pop:
            code = s.get("code", "")
            arch = classify(code)
            coop = s.get("cooperation_rate", 0)
            classifications.append((arch, coop, s.get("agent_id"), s.get("fitness"), code))
        cnt = Counter(c[0] for c in classifications)
        mean_coop = sum(c[1] for c in classifications) / len(classifications)
        max_coop = max(c[1] for c in classifications)
        trial_summary.append((obs, seed, len(pop), mean_coop, max_coop, dict(cnt)))
        print(f"\n--- {obs}_seed{seed}: 15 strategies, mean coop {mean_coop:.3f}, max {max_coop:.3f} ---")
        for arch, coop, aid, fit, _ in sorted(classifications, key=lambda x: -x[1]):
            print(f"  agent{aid:>2}  coop={coop:.3f}  fit={fit:>6.1f}  {arch}")
        for s in pop:
            all_strategies.append({
                "obs": obs, "seed": seed,
                "agent_id": s.get("agent_id"),
                "fitness": s.get("fitness"),
                "cooperation_rate": s.get("cooperation_rate", 0),
                "archetype": classify(s.get("code", "")),
                "code": s.get("code", ""),
            })

# Aggregate
print("\n" + "=" * 100)
print("Aggregate archetype distribution (180 strategies)")
print("=" * 100)
arch_cnt = Counter(s["archetype"] for s in all_strategies)
total = len(all_strategies)
for arch, n in sorted(arch_cnt.items(), key=lambda x: -x[1]):
    print(f"  {arch:<20} {n:>4}  ({100*n/total:.1f}%)")

# Per-obs
print("\nPer-obs archetype distribution:")
for obs in OBS:
    obs_strats = [s for s in all_strategies if s["obs"] == obs]
    cnt = Counter(s["archetype"] for s in obs_strats)
    print(f"\n  {obs} (n={len(obs_strats)}):")
    for arch, n in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"    {arch:<20} {n:>3}  ({100*n/len(obs_strats):.1f}%)")

# Save
out_path = OUT / "all_g30_final_strategies.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_strategies, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(all_strategies)} strategies to {out_path}")

# Show top 3 highest-coop strategies across all trials
print("\n" + "=" * 100)
print("Top 5 highest-cooperation final-pop strategies")
print("=" * 100)
top5 = sorted(all_strategies, key=lambda s: -s["cooperation_rate"])[:5]
for s in top5:
    print(f"\n[{s['obs']}_seed{s['seed']} agent{s['agent_id']}] coop={s['cooperation_rate']:.3f} fit={s['fitness']:.1f} ({s['archetype']})")
    print("-" * 80)
    print(s["code"])
    print("-" * 80)
