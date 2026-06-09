"""Publication-quality figure generation for the LLM-reputation paper.

Produces 4 figures as PDF (vector graphics) suitable for IEEE TCSS.
All figures use:
- a consistent colour palette (ColorBrewer)
- a consistent style: clean grid, light axes, bold mean line, faded seeds
- 8pt x 6pt fig size (single column 3.5 in x 2.625 in scale)
- sans-serif font (Helvetica/Arial fallback)
- PDF output via PdfPages

Output: results/figures/fig{1,2,3,4}_*.pdf
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS = Path('results')
OUT = Path('results/figures')
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Style ----------
# ColorBrewer-inspired palette
COLORS = {
    'flash':      '#1f77b4',  # deep blue
    'coder':      '#ff7f0e',  # orange
    'static':     '#2ca02c',  # green
    'random':     '#d62728',  # red
    'private':    '#9467bd',  # purple
    'partial':    '#8c564b',  # brown
    'full':       '#1f77b4',  # blue
    'mean':       '#000000',  # black
}

# Light versions for individual seed trajectories
def lighten(hexcolor, factor=0.55):
    h = hexcolor.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'

# Set up matplotlib defaults
plt.rcParams.update({
    'font.family': ['Helvetica', 'Arial', 'sans-serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,  # TrueType (not Type 3) for proper rendering
    'ps.fonttype': 42,
})

# ---------- Data loading ----------
def load_trials(root):
    rows = []
    for trial_dir in sorted(root.iterdir()):
        if not trial_dir.is_dir():
            continue
        m = re.match(r'([a-z_0-9.]+)_seed(\d+)', trial_dir.name)
        if not m:
            continue
        obs, seed = m.group(1), int(m.group(2))
        # Try multiple possible aggregate file names
        agg_candidates = (
            list(trial_dir.glob('evolutionary_*.json')) +
            list(trial_dir.glob('static_control_*.json')) +
            list(trial_dir.glob('aggregate_*.json'))
        )
        if not agg_candidates:
            continue
        d = json.loads(agg_candidates[0].read_text())
        # Exp 1/2/4/5: trials_summary[0] structure
        # Exp 3 (static): top-level trajectory + final_mean_cooperation
        ts_list = d.get('trials_summary')
        if ts_list:
            ts = ts_list[0]
            traj = ts.get('trajectory', [])
            final = ts.get('final_mean_cooperation')
        else:
            traj = d.get('trajectory', [])
            final = d.get('final_mean_cooperation')
        rows.append({
            'obs': obs, 'seed': seed, 'run': root.name,
            'trajectory': traj,
            'final_coop': final,
        })
    return rows


flash = load_trials(RESULTS / 'exp1_method')
threshold = load_trials(RESULTS / 'exp2_threshold')
static = load_trials(RESULTS / 'exp3_static_g10')  # G=10 to match LLM-evo
random_mut = load_trials(RESULTS / 'exp4_random_mut')
coder = load_trials(RESULTS / 'exp5_robustness')

obs_to_p = lambda obs: (
    0.0 if obs == 'private'
    else 1.0 if obs == 'full'
    else float(re.match(r'partial[_ ]?([0-9.]+)', obs).group(1))
    if re.match(r'partial[_ ]?([0-9.]+)', obs) else None
)


def aggregate_by_obs(trials):
    """Return dict[obs] -> list of (trajectory, final_coop)."""
    out = defaultdict(list)
    for t in trials:
        out[t['obs']].append(t)
    return out


# ---------- Figure 1: PRIVATE vs FULL (cleaner trajectories) ----------
def make_fig1():
    """
    Two-panel trajectory plot: PRIVATE (p=0) vs FULL (p=1) for the
    LLM-driven evolutionary runs. Each panel shows individual seed
    trajectories (faded) and the cross-seed mean (bold).
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.20, wspace=0.10)

    panels = [
        ('private', 'PRIVATE  (p = 0)', COLORS['private']),
        ('full',    'FULL  (p = 1)',    COLORS['full']),
    ]
    for ax, (obs, title, color) in zip(axes, panels):
        obs_trials = [t for t in flash if t['obs'] == obs]
        if not obs_trials:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue
        # individual seeds (faded)
        for t in obs_trials:
            traj = t['trajectory']
            if not traj: continue
            xs = [g['generation'] for g in traj]
            ys = [g['cooperation_rate_mean'] for g in traj]
            ax.plot(xs, ys, color=lighten(color, 0.6), linewidth=0.9, alpha=0.7, zorder=1)
        # mean (bold)
        max_len = max(len(t['trajectory']) for t in obs_trials)
        all_ys = []
        for t in obs_trials:
            ys = [g['cooperation_rate_mean'] for g in t['trajectory']]
            all_ys.append(ys + [np.nan] * (max_len - len(ys)))
        mean_ys = np.nanmean(all_ys, axis=0)
        ax.plot(range(max_len), mean_ys, color=color, linewidth=2.4, zorder=2,
                label=f'mean (n={len(obs_trials)})')
        # format
        ax.set_title(title, fontweight='bold', pad=4)
        ax.set_xlabel('Generation')
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(loc='upper right', frameon=False)
        # final value annotation
        final = mean_ys[-1]
        ax.annotate(f'final = {final:.2f}',
                    xy=(max_len - 1, final), xytext=(-2, 10),
                    textcoords='offset points',
                    fontsize=7, color=color, ha='right')

    axes[0].set_ylabel('Mean cooperation rate')
    fig.suptitle('Fig. 1.   Cooperation trajectories under PRIVATE and FULL observability',
                 y=0.99, fontsize=9)
    return fig


# ---------- Figure 2: Cooperation vs observability (with threshold scan) ----------
def make_fig2():
    """
    Final cooperation rate as a function of observability p, for three
    conditions: LLM-driven evolution, static control, random mutation.
    The LLM-driven line combines the four Exp 1 observability levels
    (private, partial_0.3, partial_0.7, full) AND the four mid-level
    p values from the threshold scan (0.1, 0.5, etc.).
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.20)

    # Aggregate by (run, p)
    def by_p(trials):
        out = defaultdict(list)
        for t in trials:
            p = obs_to_p(t['obs'])
            if p is not None and t['final_coop'] is not None:
                out[p].append(t['final_coop'])
        return out

    flash_by_p = by_p(flash)
    threshold_by_p = by_p(threshold)
    static_by_p = by_p(static)
    random_by_p = by_p(random_mut)

    # Merge flash and threshold: flash is Exp 1 (4 levels, 3 seeds), threshold
    # is Exp 2 (6 levels, 2 seeds). Use threshold for the 6-level scan
    # (broader coverage) and flash for the same levels. Combine.
    llm_by_p = defaultdict(list)
    for d in (flash_by_p, threshold_by_p):
        for p, vals in d.items():
            llm_by_p[p].extend(vals)

    # Plot in order
    runs = [
        ('LLM-driven evolution', llm_by_p, COLORS['flash'], 'o', 1.6),
        ('Static (no selection)', static_by_p, COLORS['static'], '^', 1.6),
        ('Random mutation',      random_by_p, COLORS['random'], 's', 1.6),
    ]
    for label, by_p_, color, marker, lw in runs:
        ps = sorted(by_p_.keys())
        if not ps: continue
        means = np.array([np.mean(by_p_[p]) for p in ps])
        stds = np.array([np.std(by_p_[p]) for p in ps])
        ns = np.array([len(by_p_[p]) for p in ps])
        ax.errorbar(ps, means, yerr=stds, fmt=marker + '-',
                    color=color, linewidth=lw, markersize=5,
                    markeredgecolor=color, markerfacecolor='white',
                    markeredgewidth=1.2,
                    capsize=2.5, capthick=1.0, elinewidth=1.0,
                    label=label, zorder=3)

    # Annotate n per point
    for label, by_p_, color, marker, lw in runs:
        for p, vals in by_p_.items():
            ax.annotate(f'n={len(vals)}', (p, np.mean(vals)),
                        textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=6, color='gray', zorder=1)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.3, 0.7, 1.0])
    ax.set_xticklabels(['0.0', '0.3', '0.7', '1.0'])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('Observability  p')
    ax.set_ylabel('Final cooperation rate')
    ax.legend(loc='upper left', frameon=False, fontsize=7, ncol=1)
    ax.set_title('Fig. 2.   Cooperation vs observability',
                 fontsize=9, loc='left', pad=4)
    return fig


# ---------- Figure 3: Control comparison bar chart ----------
def make_fig3():
    """
    Grouped bar chart: final cooperation rate for LLM-driven evolution,
    static, and random mutation, at the three observability levels
    where all three conditions were tested.
    """
    obs_levels = [('private', 'PRIVATE\np = 0.0'),
                  ('partial_0.3', 'PARTIAL\np = 0.3'),
                  ('full', 'FULL\np = 1.0')]
    runs = [
        ('LLM-driven evolution', COLORS['flash']),
        ('Static (no selection)', COLORS['static']),
        ('Random mutation', COLORS['random']),
    ]
    # mean final per (run, obs)
    def mean_final(trials, obs):
        vals = [t['final_coop'] for t in trials if t['obs'] == obs and t['final_coop'] is not None]
        return np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.18)

    n_obs = len(obs_levels)
    n_runs = len(runs)
    group_width = 0.78
    bar_w = group_width / n_runs
    x_centers = np.arange(n_obs)

    for i, (run_label, color) in enumerate(runs):
        offset = (i - (n_runs - 1) / 2) * bar_w
        vals = []
        for obs, _ in obs_levels:
            src = flash if run_label == 'LLM-driven evolution' else (static if 'Static' in run_label else random_mut)
            vals.append(mean_final(src, obs))
        bars = ax.bar(x_centers + offset, vals, width=bar_w * 0.95,
                      color=color, label=run_label, edgecolor='none',
                      alpha=0.92, zorder=2)
        # value labels
        for j, (bar, v) in enumerate(zip(bars, vals)):
            h = bar.get_height()
            label_y = max(h, 0.02) + 0.03
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7,
                    color=color, fontweight='bold')

    ax.set_xticks(x_centers)
    ax.set_xticklabels([lbl for _, lbl in obs_levels], fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel('Final cooperation rate')
    ax.legend(loc='upper right', frameon=False, fontsize=7, ncol=1)
    ax.set_title('Fig. 3.   Control comparison',
                 fontsize=9, loc='left', pad=4)
    return fig


# ---------- Figure 4: Selection pressure trajectories ----------
def make_fig4():
    """
    Three-panel trajectory plot: p=0, p=0.3, p=1.0.
    Each panel overlays LLM-driven evolution (3 seeds) with
    static control (2 seeds). Bold mean lines + faded individuals.
    """
    obs_panels = [
        ('private',     r'PRIVATE   ($p = 0.0$)'),
        ('partial_0.3', r'PARTIAL   ($p = 0.3$)'),
        ('full',        r'FULL   ($p = 1.0$)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.18, wspace=0.10)

    for ax, (obs, title) in zip(axes, obs_panels):
        evo_trials = [t for t in flash if t['obs'] == obs]
        sta_trials = [t for t in static if t['obs'] == obs]

        for run, color, label in [
            ('evo',  COLORS['flash'],  f'LLM-driven (n={len(evo_trials)})'),
            ('sta',  COLORS['static'], f'Static (n={len(sta_trials)})'),
        ]:
            trials = evo_trials if run == 'evo' else sta_trials
            if not trials: continue
            for t in trials:
                traj = t['trajectory']
                if not traj: continue
                xs = [g['generation'] for g in traj]
                ys = [g['cooperation_rate_mean'] for g in traj]
                ax.plot(xs, ys, color=lighten(color, 0.55), linewidth=0.9, alpha=0.7, zorder=1)
            # mean
            max_len = max(len(t['trajectory']) for t in trials)
            all_ys = []
            for t in trials:
                ys = [g['cooperation_rate_mean'] for g in t['trajectory']]
                all_ys.append(ys + [np.nan] * (max_len - len(ys)))
            mean_ys = np.nanmean(all_ys, axis=0)
            ax.plot(range(max_len), mean_ys, color=color, linewidth=2.0, zorder=2, label=label)
        # final-value annotation: place LLM and Static on opposite sides
        # to avoid overlap. Use a small leader offset that does not collide
        # with the x-axis labels.
        if evo_trials:
            m_evo = np.mean([t['trajectory'][-1]['cooperation_rate_mean'] for t in evo_trials if t['trajectory']])
        else:
            m_evo = None
        if sta_trials:
            m_sta = np.mean([t['trajectory'][-1]['cooperation_rate_mean'] for t in sta_trials if t['trajectory']])
        else:
            m_sta = None
        # Place annotations above the rightmost point, away from x-axis
        if m_evo is not None:
            label_y_evo = min(0.97, m_evo + 0.18)
            ax.annotate(f'LLM final\n= {m_evo:.2f}', xy=(max_len - 1, m_evo), xytext=(-2, 22),
                        textcoords='offset points', fontsize=6.5,
                        color=COLORS['flash'], ha='right', fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=COLORS['flash'], lw=0.6))
        if m_sta is not None:
            label_y_sta = min(0.97, m_sta + 0.05)
            ax.annotate(f'Static final\n= {m_sta:.2f}', xy=(max_len - 1, m_sta), xytext=(-2, 8),
                        textcoords='offset points', fontsize=6.5,
                        color=COLORS['static'], ha='right', fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=COLORS['static'], lw=0.6))
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('Generation')
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # Legend in upper area (less likely to collide with data)
        ax.legend(loc='upper right', frameon=False, fontsize=6.5,
                  bbox_to_anchor=(0.98, 0.98))

    axes[0].set_ylabel('Mean cooperation rate')
    fig.suptitle('Fig. 4.   Selection pressure: LLM-driven evolution vs static control',
                 y=0.99, fontsize=9)
    return fig


# ---------- Render & save ----------
def save_fig(fig, name):
    pdf_path = OUT / f'{name}.pdf'
    png_path = OUT / f'{name}.png'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.02, dpi=200)
    print(f'  saved {pdf_path}  ({pdf_path.stat().st_size // 1024} KB)')
    print(f'  saved {png_path}  ({png_path.stat().st_size // 1024} KB)')
    plt.close(fig)


print('=== generating publication-quality figures ===\n')
save_fig(make_fig1(), 'fig1_obs_contrast')
save_fig(make_fig2(), 'fig2_observability_scan')
save_fig(make_fig3(), 'fig3_control_comparison')
save_fig(make_fig4(), 'fig4_selection_comparison')
print('\nDone.')
