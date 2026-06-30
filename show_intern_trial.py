"""Show 1 detailed Intern trial + compare to v15 deepseek-v4-flash same probe."""
import json, os

# Intern B_recent_window full seed1 (best, 0.97)
d = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp8_intern_ceiling\B_recent_window\full_seed1'
files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
with open(os.path.join(d, files[0])) as f:
    t = json.load(f)

print('=== Intern B_recent_window / full / seed1 (final coop 0.97) ===\n')
print('Trajectory:')
for g in t['trajectory']:
    print(f'  gen {g["generation"]:2d}: coop={g["cooperation_rate_mean"]:.3f}  '
          f'fitness_mean={g.get("fitness_mean", 0):.2f}  '
          f'best_fitness={g.get("best_fitness", 0):.2f}')

print(f'\nNum unique strategies: {len(set(a["code"] for a in t["final_population"]))} / {len(t["final_population"])}')
mean_len = sum(len(a["code"]) for a in t["final_population"]) / len(t["final_population"])
print(f'Mean code length: {mean_len:.0f} chars')
print(f'Max code length: {max(len(a["code"]) for a in t["final_population"])}')
print(f'Min code length: {min(len(a["code"]) for a in t["final_population"])}')

# Show 1 sample
print('\n--- Sample final-pop strategy (agent 0) ---')
print(t['final_population'][0]['code'][:2000])

# v15 deepseek-v4-flash B_recent_window full seed0 (best v15 B at full)
d2 = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\B_recent_window\full_seed0'
files2 = [f for f in os.listdir(d2) if f.startswith('evo_') and f.endswith('.json')]
files2.sort(key=lambda f: os.path.getmtime(os.path.join(d2, f)), reverse=True)
with open(os.path.join(d2, files2[0])) as f:
    t2 = json.load(f)

print('\n\n=== v15 DeepSeek-V4-Flash B_recent_window / full / seed0 (final coop 0.06) ===\n')
print('Trajectory:')
for g in t2['trajectory']:
    print(f'  gen {g["generation"]:2d}: coop={g["cooperation_rate_mean"]:.3f}')

# Compare both
print('\n\n=== Side-by-side compare ===')
print(f'{"":20s} {"Intern":>15s} {"DeepSeek":>15s}')
print(f'{"final coop":20s} {t["trajectory"][-1]["cooperation_rate_mean"]:15.3f} {t2["trajectory"][-1]["cooperation_rate_mean"]:15.3f}')
print(f'{"mean code len":20s} {mean_len:15.0f} {sum(len(a["code"]) for a in t2["final_population"])/len(t2["final_population"]):15.0f}')
print(f'{"num agents":20s} {len(t["final_population"]):15d} {len(t2["final_population"]):15d}')