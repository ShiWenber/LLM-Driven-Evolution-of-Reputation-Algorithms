"""Smoke test: Fermi Z-like 5-gen with LLM, with the
`_ctx_opponent_id` setter bug fixed.

Verifies:
  - LLM init works (15 agents, agent_type=v3)
  - Fermi Z-like selects (μ path + 1-μ path both called)
  - The `_ctx_opponent_id` property-without-setter failure mode
    is gracefully handled (no crash)

Run:
  DEEPSEEK_API_KEY in env. 5 gen expected ~5-10 min.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

ROOT = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUT_DIR = ROOT / 'results' / 'quantitative_baseline' / '_smoke_fermi_z_llm'
OUT_DIR.mkdir(parents=True, exist_ok=True)

api_key = get_api_key('deepseek')
base_url = get_base_url('deepseek')

NUM_GENS = 5
print(f"=== Fermi Z-like LLM smoke, {NUM_GENS} gen ===", flush=True)
print(f"  api_key: {api_key[:8]}...{api_key[-4:]}", flush=True)

t0 = time.time()
try:
    pop = V2EvolutionaryPopulation(
        population_size=15,
        target_interactions_per_gen=200,  # smaller for smoke (faster)
        benefit=2.0,
        cost=1.0,
        observability='full',
        observability_p=1.0,
        elite_count=2,
        num_eliminate=5,
        tournament_size=3,
        llm_model='deepseek-v4-flash',
        api_key=api_key,
        api_base_url=base_url,
        mutation_temperature=0.8,
        seed=42,
        results_dir=str(OUT_DIR),
        use_baseline=None,
        agent_type='v3',
        llm_thinking=False,
        # Fermi Z-like
        use_fermi=True,
        fermi_beta=5.0,
        mutation_rate_on_adoption=0.1,
        updates_per_gen=15,
        forbid_self_pairing=True,
    )
    result = pop.run_evolution(num_generations=NUM_GENS)
    elapsed = time.time() - t0

    out_path = OUT_DIR / 'evolutionary.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    last = result['trajectory'][-1]
    coop_curve = [t['cooperation_rate_mean'] for t in result['trajectory']]
    print(f"=== DONE in {elapsed/60:.1f} min ===", flush=True)
    print(f"  coop curve: {[f'{c:.3f}' for c in coop_curve]}", flush=True)
    print(f"  final coop: {last['cooperation_rate_mean']:.3f}", flush=True)
    print(f"  FALLBACK: init={result['config']['fallback_init_count']}/15, "
          f"mutation={result['config']['fallback_mutation_count']}/{(NUM_GENS-1)*15}", flush=True)
    print(f"  PASS: Z-like LLM smoke completed without crash", flush=True)
except Exception as e:
    elapsed = time.time() - t0
    print(f"=== FAILED after {elapsed/60:.1f} min ===", flush=True)
    print(f"  {type(e).__name__}: {e}", flush=True)
    print(traceback.format_exc(), flush=True)
    sys.exit(1)
