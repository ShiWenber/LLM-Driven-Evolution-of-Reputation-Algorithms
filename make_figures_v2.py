"""Figure generation for the new (real-LLM) Standard run data.

Reads from results/exp{1,2,3,4}_* and produces Figure 1/2/3/4 as PNG.
Output also produces a summary.md table.

Run:  python make_figures_v2.py
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

RESULTS = Path('results')
SUBDIRS = {
    'evolutionary': 'exp1_method',
    'threshold':    'exp2_threshold',
    'static':       'exp3_static',
    'random-mutation': 'exp4_random_mut',
}


def load_all_trials() -> List[Dict[str, Any]]:
    """Load every trial record. Each record: {run, obs, seed, trajectory, final_coop}."""
    out = []
    for run, sub in SUBDIRS.items():
        d = RESULTS / sub
        if not d.exists():
            continue
        for trial_dir in sorted(d.iterdir()):
            if not trial_dir.is_dir():
                continue
            m = re.match(r'([a-z_0-9.]+)_seed(\d+)', trial_dir.name)
            if not m:
                continue
            obs = m.group(1)
            seed = int(m.group(2))
            agg_files = (
                list(trial_dir.glob('evolutionary_*.json')) +
                list(trial_dir.glob('static_control_*.json')) +
                list(trial_dir.glob('aggregate_*.json'))
            )
            if not agg_files:
                continue
            agg = json.loads(agg_files[0].read_text())
            ts = agg.get('trials_summary', [])
            if ts:
                tr = ts[0]
                traj = tr.get('trajectory', [])
                final = tr.get('final_mean_cooperation')
            else:
                traj = agg.get('trajectory', [])
                final = agg.get('final_mean_cooperation')
            # Map partial_X / private / full to numeric p
            p = obs_label_to_p(obs)
            out.append({
                'run': run, 'obs': obs, 'seed': seed, 'p': p,
                'trajectory': traj, 'final_coop': final,
            })
    return out


def obs_label_to_p(obs: str) -> Optional[float]:
    if obs == 'private': return 0.0
    if obs == 'full': return 1.0
    m = re.match(r'partial[_ ]?([0-9.]+)', obs)
    if m: return float(m.group(1))
    return None


# ---------- Figure rendering ----------

def make_fig1(trials, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    obs_to_color = {'private': '#d62728', 'full': '#1f77b4'}
    obs_to_title = {'private': 'PRIVATE (p=0.0)', 'full': 'FULL (p=1.0)'}
    for ax, obs in zip(axes, ['private', 'full']):
        obs_trials = [t for t in trials if t['obs'] == obs and t['run'] == 'evolutionary']
        if not obs_trials:
            continue
        for t in obs_trials:
            traj = t['trajectory']
            if not traj:
                continue
            xs = [g['generation'] for g in traj]
            ys = [g['cooperation_rate_mean'] for g in traj]
            ax.plot(xs, ys, color=obs_to_color[obs], alpha=0.4, linewidth=1)
        # Mean
        max_len = max(len(t['trajectory']) for t in obs_trials)
        all_ys = []
        for t in obs_trials:
            ys = [g['cooperation_rate_mean'] for g in t['trajectory']]
            all_ys.append(ys + [np.nan] * (max_len - len(ys)))
        arr = np.array(all_ys)
        mean_ys = np.nanmean(arr, axis=0)
        ax.plot(range(max_len), mean_ys, color='black', linewidth=2.5,
                label=f'mean (n={len(obs_trials)})')
        ax.set_xlabel('Generation')
        ax.set_title(obs_to_title[obs])
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Mean cooperation rate')
    fig.suptitle('Fig 1: Cooperation Trajectories — LLM-driven Evolution (3 seeds per condition)')
    fig.tight_layout()
    p = output_dir / 'fig1_obs_contrast.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  saved {p}')


def make_fig2(trials, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Final coop by p, all four runs
    fig, ax = plt.subplots(figsize=(8, 5))
    runs = [
        ('evolutionary',    'LLM-driven evolution',   '#1f77b4', 'o'),
        ('random-mutation', 'Random mutation control', '#ff7f0e', 's'),
        ('static',          'Static control (no sel.)','#2ca02c', '^'),
    ]
    for run, label, color, marker in runs:
        run_trials = [t for t in trials if t['run'] == run and isinstance(t['p'], (int, float))]
        run_trials.sort(key=lambda t: t['p'])
        if not run_trials:
            continue
        # group by p
        by_p = defaultdict(list)
        for t in run_trials:
            by_p[t['p']].append(t['final_coop'])
        ps = sorted(by_p.keys())
        means = [np.mean(by_p[p]) for p in ps]
        stds = [np.std(by_p[p]) for p in ps]
        ax.errorbar(ps, means, yerr=stds, fmt=marker + '-', color=color,
                    capsize=4, linewidth=2, markersize=8, label=label)
    ax.set_xlabel('Observability p')
    ax.set_ylabel('Final-generation mean cooperation rate')
    ax.set_title('Fig 2: Cooperation vs Observability (LLM-evo vs controls)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'fig2_observability_scan.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  saved {p}')


def make_fig3(trials, output_dir):
    """Control comparison: final coop by observability, three conditions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    obs_to_x = {'private': 0.0, 'partial_0.3': 0.3, 'full': 1.0}
    runs = [
        ('evolutionary',    'LLM-driven evolution',   '#1f77b4', 'o'),
        ('random-mutation', 'Random mutation',         '#ff7f0e', 's'),
        ('static',          'Static (no selection)',   '#2ca02c', '^'),
    ]
    width = 0.25
    for i, (run, label, color, marker) in enumerate(runs):
        run_trials = [t for t in trials if t['run'] == run and t['obs'] in obs_to_x]
        run_trials.sort(key=lambda t: obs_to_x[t['obs']])
        xs = [obs_to_x[t['obs']] + (i - 1) * width for t in run_trials]
        ys = [t['final_coop'] for t in run_trials]
        ax.bar(xs, ys, width=width, color=color, label=label, alpha=0.85)
    ax.set_xticks([0.0, 0.3, 1.0])
    ax.set_xticklabels(['PRIVATE\n(p=0)', 'PARTIAL\n(p=0.3)', 'FULL\n(p=1)'])
    ax.set_ylabel('Final-generation mean cooperation rate')
    ax.set_title('Fig 3: Control Comparison (Exp 1 / 3 / 4)')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    p = output_dir / 'fig3_control_comparison.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  saved {p}')


def make_fig4(trials, output_dir):
    """Selection pressure: LLM-evo vs static, trajectories side-by-side at p=0/0.3/1.0."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    obs_panels = [
        ('private',     'p=0.0  (PRIVATE)',   '#d62728'),
        ('partial_0.3', 'p=0.3  (PARTIAL)',   '#ff7f0e'),
        ('full',        'p=1.0  (FULL)',      '#1f77b4'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (obs, title, color) in zip(axes, obs_panels):
        for run, color_alt, label in [
            ('evolutionary', '#1f77b4', 'LLM-driven evolution (n=3)'),
            ('static',       '#2ca02c', 'Static (no sel./mut., n=2)'),
        ]:
            obs_trials = [t for t in trials if t['run'] == run and t['obs'] == obs]
            if not obs_trials:
                continue
            for t in obs_trials:
                traj = t['trajectory']
                if not traj:
                    continue
                xs = [g['generation'] for g in traj]
                ys = [g['cooperation_rate_mean'] for g in traj]
                ax.plot(xs, ys, color=color_alt, alpha=0.3, linewidth=1)
            max_len = max(len(t['trajectory']) for t in obs_trials)
            all_ys = []
            for t in obs_trials:
                ys = [g['cooperation_rate_mean'] for g in t['trajectory']]
                all_ys.append(ys + [np.nan] * (max_len - len(ys)))
            arr = np.array(all_ys)
            mean_ys = np.nanmean(arr, axis=0)
            ax.plot(range(max_len), mean_ys, color=color_alt, linewidth=2.5, label=label)
        ax.set_xlabel('Generation')
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    axes[0].set_ylabel('Mean cooperation rate')
    fig.suptitle('Fig 4: Selection Pressure — LLM-driven Evolution vs Static Control')
    fig.tight_layout()
    p = output_dir / 'fig4_selection_comparison.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  saved {p}')


def write_summary(trials, output_dir):
    """Markdown summary table grouped by run and obs."""
    by_k = defaultdict(list)
    for t in trials:
        by_k[(t['run'], t['obs'])].append(t['final_coop'])
    lines = ['# Experimental Summary (Standard plan, real LLM-driven runs)\n']
    lines.append('| Run | Observability | n | Mean | Std | Min | Max |')
    lines.append('|---|---|---|---|---|---|---|')
    for k in sorted(by_k.keys()):
        vals = [v for v in by_k[k] if isinstance(v, (int, float))]
        if not vals:
            continue
        run, obs = k
        m, s, lo, hi = np.mean(vals), np.std(vals), min(vals), max(vals)
        lines.append(f'| {run} | {obs} | {len(vals)} | {m:.3f} | {s:.3f} | {lo:.3f} | {hi:.3f} |')
    out = output_dir / 'summary.md'
    out.write_text('\n'.join(lines))
    print(f'  saved {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='results/figures')
    args = p.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = load_all_trials()
    print(f'Loaded {len(trials)} trials.')

    print('\n[Fig 1] PRIVATE vs FULL trajectories')
    make_fig1(trials, output_dir)
    print('\n[Fig 2] Cooperation vs observability (LLM-evo vs controls)')
    make_fig2(trials, output_dir)
    print('\n[Fig 3] Control comparison (3 obs × 3 conditions)')
    make_fig3(trials, output_dir)
    print('\n[Fig 4] Selection pressure: LLM-evo vs static')
    make_fig4(trials, output_dir)
    print('\n[Summary] writing summary.md')
    write_summary(trials, output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
