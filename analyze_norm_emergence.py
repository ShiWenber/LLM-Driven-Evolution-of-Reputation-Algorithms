"""
Analyze final-population strategy distribution at the LAST generation.
Question: did the group develop a "social norm"?

A norm (in the IR literature) means:
  (1) consensus on what the population is doing (low strategy diversity)
  (2) behavioral alignment across agents (uniform cooperation rate)
  (3) norm expression as an implicit rule that the population collectively enforces
  (4) resistance to invasion by mutants (defectors don't gain)

We measure (1), (2), (3) empirically per final population.
"""
import json, glob, os, re
from collections import Counter
import statistics

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"

# Only use v15 main plan + robustness (the trial types reported in paper)
# 1) exp1_method_n10 (v15 standard, N=15)
# 2) exp5_robustness (v18 partial replication with deepseek-coder)
# 3) exp6_sweep_AB_n5 / exp6_sweep_CD_n5 (label ablation, deepseek-v4-flash)
# 4) exp7_algorithmic_ceiling (probes A-E)
# 5) exp8_intern_ceiling_v18 (intern)
target_dirs = [
    "exp1_method_n10",
    "exp5_robustness",
    "exp6_sweep_AB_n5",
    "exp6_sweep_CD_n5",
    "exp7_algorithmic_ceiling",
    "exp8_intern_ceiling_v18",
    "exp8_intern_ceiling_v19_A",
    "exp9_bc_scan",
]

def classify_strategy(code: str) -> str:
    """Heuristic 9-archetype classifier — matches paper's regex scheme."""
    if code is None: return "INVALID"
    has_evaluate = "def evaluate" in code
    has_decide = "def decide" in code
    if not (has_evaluate and has_decide): return "INVALID"
    # Did the agent cooperate at all (using cooperation_rate from data, but here infer from code)
    # Archetypes:
    always_c = code.count("return True") > 1 or "return True\n" in code
    always_d = code.count("return False") > 1
    has_my_history = "my_history" in code
    has_recent_window = "recent_window" in code
    has_decay = re.search(r"0\.\d+\s*\*\s*current_reputation", code) or "0.9 * current_reputation" in code
    has_emwa = re.search(r"alpha\s*=\s*0\.\d+", code) or re.search(r"\(1\s*-\s*alpha\)\s*\*\s*current_reputation", code)
    has_simple_is = ("current_reputation + 1" in code or "+= 1" in code or "+ 1.0" in code) \
                    and ("current_reputation - 1" in code or "-= 1" in code or "- 1.0" in code)
    has_threshold_zero = ">= 0" in code or "> 0" in code
    has_round_dependent = re.search(r"round_num\s*[<>=]+\s*\d+", code)
    if always_c and not has_my_history: return "ALLC"
    if always_d: return "ALLD"
    if has_simple_is and not has_my_history: return "ImageScoring"
    if has_simple_is and has_my_history: return "Hybrid_IS_with_history"
    if has_emwa and has_my_history: return "Hybrid_EMA_with_history"
    if has_decay and has_my_history: return "Hybrid_Decay_with_history"
    if has_my_history: return "Hybrid_Other"
    if has_simple_is: return "ImageScoring"
    if has_recent_window: return "RecentWindow"
    if has_round_dependent: return "RoundDependent"
    return "ThresholdOnly"

# Walk all trials, get final-pop distribution
trials = []
for d in target_dirs:
    pattern = os.path.join(base, d, "**", "evo_*.json")
    for f in glob.glob(pattern, recursive=True):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue
        if "final_population" not in data: continue
        final = data["final_population"]
        if not final: continue
        cfg = data.get("config", {})
        # Co-op rate distribution
        coop_rates = [a.get("cooperation_rate", 0) for a in final]
        fit = [a.get("fitness", 0) for a in final]
        # Archetype distribution
        archetypes = Counter(classify_strategy(a.get("code", "")) for a in final)
        # Strategy uniqueness: # distinct strategy_ids
        unique_strats = len(set(a.get("strategy_id", "?") for a in final))
        # Most common archetype
        n = len(final)
        most_common, mc_count = archetypes.most_common(1)[0]
        consensus_pct = mc_count / n
        trials.append({
            "file": os.path.relpath(f, base),
            "obs": cfg.get("observability", "?"),
            "p": cfg.get("observability_p"),
            "llm": cfg.get("llm_model", "?"),
            "N": cfg.get("population_size"),
            "n_rounds": cfg.get("num_rounds_per_gen"),
            "b/c": f"{cfg.get('benefit', '?')}/{cfg.get('cost', '?')}",
            "n_final": n,
            "unique_strats": unique_strats,
            "archetypes": dict(archetypes),
            "most_common": most_common,
            "consensus_pct": consensus_pct,
            "mean_coop": statistics.mean(coop_rates) if coop_rates else 0,
            "std_coop": statistics.stdev(coop_rates) if len(coop_rates) > 1 else 0,
            "min_coop": min(coop_rates) if coop_rates else 0,
            "max_coop": max(coop_rates) if coop_rates else 0,
        })

print(f"Analyzed {len(trials)} trials\n")
print("=" * 100)

# Per-trial summary
print("\nPer-trial consensus + cooperation uniformity:")
print(f"{'obs':10s} {'N':3s} {'LLM':25s} {'n':3s} {'uniq':5s} {'top_archetype':35s} {'cons%':6s} {'mean_coop':9s} {'std_coop':8s} {'min':5s} {'max':5s}")
for t in trials[:60]:
    print(f"{t['obs']:10s} {t['N']:3d} {t['llm'][:25]:25s} {t['n_final']:3d} {t['unique_strats']:5d} {t['most_common'][:35]:35s} {t['consensus_pct']:6.1%} {t['mean_coop']:9.3f} {t['std_coop']:8.3f} {t['min_coop']:5.2f} {t['max_coop']:5.2f}")

# Aggregate: norm-consensus indicator = (consensus_pct >= 0.8) AND (mean_coop >= 0.7)
print(f"\n{'='*100}")
print("Norm-consensus indicator (consensus% ≥ 80% AND mean_coop ≥ 70%):")
norm_strong = [t for t in trials if t['consensus_pct'] >= 0.8 and t['mean_coop'] >= 0.7]
print(f"  Strong norm (high consensus + high coop): {len(norm_strong)} / {len(trials)} trials ({100*len(norm_strong)/len(trials):.1f}%)")

# Consensus alone
high_consensus = [t for t in trials if t['consensus_pct'] >= 0.8]
print(f"  High strategy consensus (≥80% same archetype): {len(high_consensus)} / {len(trials)} trials ({100*len(high_consensus)/len(trials):.1f}%)")

# All-agents-coop
all_coop = [t for t in trials if t['min_coop'] >= 0.99]
print(f"  All 10 agents cooperate (min_coop ≥ 0.99): {len(all_coop)} / {len(trials)} trials")

# Bimodal (some agents at 0, some at 1)
bimodal = [t for t in trials if t['min_coop'] < 0.1 and t['max_coop'] > 0.9]
print(f"  Bimodal (some at 0, some at 1): {len(bimodal)} / {len(trials)} trials")

# Average consensus percentage across all trials
mean_cons = statistics.mean(t['consensus_pct'] for t in trials)
mean_uniq = statistics.mean(t['unique_strats'] for t in trials)
print(f"\n  Mean consensus %: {mean_cons:.2%}")
print(f"  Mean unique strategies per trial: {mean_uniq:.2f} (out of N={trials[0]['n_final']})")

# Show the strong-norm trials
if norm_strong:
    print(f"\n{'='*100}")
    print("Strong-norm trials (consensus ≥ 80% AND mean_coop ≥ 70%):")
    for t in norm_strong:
        print(f"  {t['file']}")
        print(f"     obs={t['obs']}  N={t['N']}  LLM={t['llm']}  b/c={t['b/c']}")
        print(f"     archetypes: {t['archetypes']}")
        print(f"     mean_coop={t['mean_coop']:.3f}  std_coop={t['std_coop']:.3f}  consensus={t['consensus_pct']:.0%}")
        print()
