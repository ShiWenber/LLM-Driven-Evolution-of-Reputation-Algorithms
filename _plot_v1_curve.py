"""Plot v1 (with canonical hints) 3-seed cooperation trajectory.

Uses whatever gens are in the log file (partial or full). For each seed
we parse the log for "Gen N: coop=X fitness=Y" lines.
"""
import re
import os
from collections import defaultdict

base = r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline"

# Parse log files (they have Gen N: coop=X, fitness=Y lines)
per_seed = {}  # seed -> {gen: (coop, fitness)}
for s in [0, 1, 2]:
    log_path = os.path.join(base, f"LLM_v3_g100_1000inter_ADVERSARIAL_seed{s}", f"seed{s}.log")
    if not os.path.exists(log_path):
        print(f"seed {s}: no log")
        continue
    with open(log_path) as f:
        content = f.read()
    # Match "Gen N: coop=0.123, fitness=12.3"
    pattern = re.compile(r'Gen (\d+):\s*coop=([0-9.]+),\s*fitness_mean=([0-9.]+)')
    matches = pattern.findall(content)
    data = {}
    for m in matches:
        gen = int(m[0])
        coop = float(m[1])
        fit = float(m[2])
        data[gen] = (coop, fit)
    per_seed[s] = data
    print(f"seed {s}: parsed {len(data)} gens ({min(data.keys()) if data else 0} to {max(data.keys()) if data else 0})")

# Aggregate
all_gens = sorted(set().union(*[set(d.keys()) for d in per_seed.values()]))
print(f"\nAvailable gens: {min(all_gens)} to {max(all_gens)}")

# Print summary table
print(f"\n{'gen':<5} {'seed0':<15} {'seed1':<15} {'seed2':<15} {'mean':<10} {'std':<10}")
for g in all_gens[::5] + [all_gens[-1]]:  # every 5 gens + last
    vals = [per_seed[s][g][0] for s in [0,1,2] if g in per_seed.get(s, {})]
    if not vals:
        continue
    parts = [f"seed{s}={per_seed[s][g][0]:.3f}" if g in per_seed.get(s, {}) else f"seed{s}=--" for s in [0,1,2]]
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        std = (sum((v - mean)**2 for v in vals) / (len(vals) - 1)) ** 0.5
    else:
        std = 0
    print(f"{g:<5} {parts[0]:<15} {parts[1]:<15} {parts[2]:<15} {mean:<10.3f} {std:<10.3f}")

# Now plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, s in enumerate([0, 1, 2]):
        if s not in per_seed or not per_seed[s]:
            continue
        gens = sorted(per_seed[s].keys())
        coops = [per_seed[s][g][0] for g in gens]
        ax.plot(gens, coops, label=f'seed {s} (gen {gens[-1]})', color=colors[i], linewidth=1.5, alpha=0.85)
        ax.scatter(gens[-1], coops[-1], color=colors[i], s=50, zorder=5)

    # Mean band
    gens_all = sorted(set().union(*[set(d.keys()) for d in per_seed.values()]))
    means, stds = [], []
    for g in gens_all:
        v = [per_seed[s][g][0] for s in [0,1,2] if g in per_seed.get(s, {})]
        if len(v) >= 2:
            means.append(sum(v)/len(v))
            stds.append((sum((x - means[-1])**2 for x in v)/(len(v)-1))**0.5)
        elif len(v) == 1:
            means.append(v[0])
            stds.append(0)
        else:
            continue
    if means:
        ax.plot(gens_all[:len(means)], means, 'k--', linewidth=2, label='mean', alpha=0.7)
        ax.fill_between(gens_all[:len(means)],
                        [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)],
                        color='gray', alpha=0.2, label='±1 std')

    # Reference: Neutral prompt (main run) endpoints
    main_path = os.path.join(base, "LLM_v3_g100_1000inter_seed0", "evolutionary.json")
    if os.path.exists(main_path):
        import json
        for ss, color, lbl in [(0, 'red', 'main s0=0.000'), (1, 'orange', 'main s1=0.474'), (2, 'darkred', 'main s2=0.851')]:
            try:
                mj = json.load(open(os.path.join(base, f"LLM_v3_g100_1000inter_seed{ss}", "evolutionary.json")))
                mg = [t['generation'] for t in mj['trajectory']]
                mc = [t['cooperation_rate_mean'] for t in mj['trajectory']]
                ax.plot(mg, mc, color=color, linewidth=1.5, linestyle=':', alpha=0.6, label=lbl)
            except Exception as e:
                pass

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cooperation rate')
    ax.set_title('Ablation A v1: ADVERSARIAL prompt (with canonical hints: Nowak-Sigmund, leading-eight)\n100 gen × 3 seed × 1000 inter/gen')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 102)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    out_png = os.path.join(base, "plots", "v1_adversarial_partial_curve.png")
    out_pdf = os.path.join(base, "plots", "v1_adversarial_partial_curve.pdf")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")
except ImportError as e:
    print(f"matplotlib not available: {e}")
