"""Smoke test: verify burn-in produces all-good reputation state."""
import sys
sys.path.insert(0, '.')
from _run_invasion_sweep import run_invasion, _mean_reputation

ALLD = '''
def evaluate(target_reputation, target_action, my_reputation):
    if target_action == 'cooperate':
        new = target_reputation + 0.333
    else:
        new = target_reputation - 0.333
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return False
'''

print('=== Smoke: ALLD vs IS with burn-in=10 ===')
traj = run_invasion('IS', resident_count=14,
                    invader_code=ALLD, invader_count=1,
                    num_gens=20, seed=42, population_size=15,
                    burn_in_gens=10)
for t in [0, 1, 5, 10, 15, 20]:
    if t < len(traj):
        d = traj[t]
        print(f'  gen={t}: inv_freq={d["invader_freq"]:.3f}  '
              f'mean_coop={d["mean_coop"]:.3f}  '
              f'mean_rep={d["mean_rep"]:.3f}')
print(f'\nBurn-in metadata: burn_in_gens={traj[0].get("burn_in_gens")}')

print('\n=== Smoke: ALLD vs IS with burn-in=0 (no burn-in) for comparison ===')
traj0 = run_invasion('IS', resident_count=14,
                     invader_code=ALLD, invader_count=1,
                     num_gens=20, seed=42, population_size=15,
                     burn_in_gens=0)
for t in [0, 1, 5, 10, 15, 20]:
    if t < len(traj0):
        d = traj0[t]
        print(f'  gen={t}: inv_freq={d["invader_freq"]:.3f}  '
              f'mean_coop={d["mean_coop"]:.3f}  '
              f'mean_rep={d["mean_rep"]:.3f}')
