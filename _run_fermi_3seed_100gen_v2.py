"""Production run: 100 gen x target_interactions_per_gen=1000 x 3 seeds.

M7 Fermi Z-like scheme (commit b3468e5):
  per imitated event (with prob mu=0.1): offspring = LLM init (independent)
  per imitated event (with prob 1-mu=0.9): offspring = LLM small-mutate(j)

Each seed expected ~5-7h based on Z-like LLM call count:
  ~15 init + ~7.5 imitations/gen x 99 gens x 1 LLM/imitation = ~758 calls/seed
  at ~11s/call + ~20s game/gen -> ~7h/seed.

Run detached in parallel (3 processes). Sequential would be ~21h, parallel
~7h. Risk: deepseek-v4-flash API rate limit; if hit, fall back to
sequential.
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

ROOT = Path(r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUT_BASE = ROOT / 'results' / 'quantitative_baseline'

api_key = get_api_key('deepseek')
base_url = get_base_url('deepseek')

LABEL = "LLM_v3_fermi_z_v2_g100_1000inter"

# Accept seed override via argv: pass a single int to run only that seed.
if len(sys.argv) > 1:
    seeds = [int(sys.argv[1])]
else:
    seeds = [0, 1, 2]

NUM_GENS = 100

print(f"=== {LABEL} seeds={seeds} (sequential) ===", flush=True)
print(f"  api_key: {api_key[:8]}...{api_key[-4:]}", flush=True)
print(f"  num_gens: {NUM_GENS}, target_interactions: 1000", flush=True)
print(f"  Z-like: mu=0.1, beta=5.0, updates_per_gen=15", flush=True)
print(f"  estimated: ~5-7h/seed x {len(seeds)} seed(s)", flush=True)
print(flush=True)

summary = []
overall_t0 = time.time()

for seed in seeds:
    seed_dir = OUT_BASE / f"{LABEL}_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_path = seed_dir / 'evolutionary.json'
    if out_path.exists():
        out_path.unlink()
        print(f"  [seed {seed}] removed existing {out_path}", flush=True)

    print(f"=== seed {seed} start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    t0 = time.time()
    seed_summary = {'seed': seed, 'completed': False, 'error': None}

    try:
        pop = V2EvolutionaryPopulation(
            population_size=15,
            target_interactions_per_gen=1000,
            benefit=2.0,
            cost=1.0,
            observability='full',
            observability_p=1.0,
            elite_count=2,           # legacy fields, ignored when use_fermi=True
            num_eliminate=5,         # legacy fields, ignored when use_fermi=True
            tournament_size=3,       # legacy fields, ignored when use_fermi=True
            llm_model='deepseek-v4-flash',
            api_key=api_key,
            api_base_url=base_url,
            mutation_temperature=0.8,
            seed=seed,
            results_dir=str(OUT_BASE),
            use_baseline=None,
            agent_type='v3',
            llm_thinking=False,
            # === Fermi Z-like scheme ===
            use_fermi=True,
            fermi_beta=5.0,
            mutation_rate_on_adoption=0.1,  # mu
            updates_per_gen=15,              # = N (Moran standard)
            forbid_self_pairing=True,
        )
        result = pop.run_evolution(num_generations=NUM_GENS)
        elapsed = time.time() - t0

        # Save
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Summarize
        last = result['trajectory'][-1]
        seed_summary.update({
            'completed': True,
            'elapsed_sec': elapsed,
            'elapsed_min': elapsed / 60,
            'final_coop': last['cooperation_rate_mean'],
            'final_fitness': last['fitness_mean'],
            'fallback_init': result['config']['fallback_init_count'],
            'fallback_mutation': result['config']['fallback_mutation_count'],
            'gen0_coop': result['trajectory'][0]['cooperation_rate_mean'],
        })
        print(f"=== seed {seed} done in {elapsed/60:.1f} min ===", flush=True)
        print(f"  gen 0 coop: {seed_summary['gen0_coop']:.3f}", flush=True)
        print(f"  final coop: {seed_summary['final_coop']:.3f}", flush=True)
        print(f"  final fitness: {seed_summary['final_fitness']:.1f}", flush=True)
        print(f"  FALLBACK: init={seed_summary['fallback_init']}/15, "
              f"mutation={seed_summary['fallback_mutation']}/{(NUM_GENS-1)*15}", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        seed_summary.update({
            'completed': False,
            'elapsed_sec': elapsed,
            'error': f"{type(e).__name__}: {e}",
            'traceback': traceback.format_exc(),
        })
        print(f"=== seed {seed} FAILED after {elapsed/60:.1f} min ===", flush=True)
        print(f"  {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    summary.append(seed_summary)
    # Save the running summary after each seed so partial state survives crashes
    summary_path = OUT_BASE / f"{LABEL}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'label': LABEL,
            'num_gens': NUM_GENS,
            'target_interactions_per_gen': 1000,
            'scheme': 'fermi_z_like (mu=0.1, beta=5.0, updates_per_gen=15)',
            'seeds': summary,
            'overall_elapsed_sec': time.time() - overall_t0,
        }, f, indent=2)

# Final summary
total_elapsed = time.time() - overall_t0
n_done = sum(1 for s in summary if s['completed'])
print(f"\n=== ALL SEEDS DONE in {total_elapsed/3600:.2f} h ({total_elapsed/60:.0f} min) ===", flush=True)
print(f"  {n_done}/{len(seeds)} seeds completed successfully", flush=True)
for s in summary:
    if s['completed']:
        print(f"  seed {s['seed']}: final_coop={s['final_coop']:.3f}, "
              f"final_fitness={s['final_fitness']:.1f}, "
              f"min={s['elapsed_min']:.1f}, "
              f"FALLBACK init/mut={s['fallback_init']}/{s['fallback_mutation']}")
    else:
        print(f"  seed {s['seed']}: FAILED ({s.get('error', '?')})")
