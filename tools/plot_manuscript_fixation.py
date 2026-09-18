"""Paper-sized payoff/fixation panels using the current b=3 benchmark."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_paper_fixation_b3 import drift_checks
DATA = ROOT / 'results/manuscript_v2/fixation_b3_20260918'
PROBES = ('L1', 'ALLC', 'ALLD')
FORWARD, REVERSE = '#2878B5', '#E07A5F'
STYLES = (('#247A91', '--', 's'), ('#111111', '-', 'o'), ('#999999', '-', '^'))


def build_figure():
    """Return all five candidates, retaining raw probabilities and sensitivity data."""
    fig, axes = plt.subplots(5, 3, figsize=(7.15, 7.5),
                             gridspec_kw={'width_ratios': [1.18, 1, 1.05]})
    fig.subplots_adjust(left=.08, right=.985, bottom=.055, top=.92,
                        hspace=.48, wspace=.51)
    records = []
    for seed, (payoff, full, zoom) in enumerate(axes):
        path = DATA / f'seed{seed}/fixation_benchmark.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        n = data['config']['population_size']
        neutral = data['config']['neutral_fixation_probability']
        assert n == 20 and abs(neutral - .05) < 1e-12
        assert data['config']['benefit'] == 3 and data['config']['cost'] == 1
        assert data['config']['observation_schedule'] == 'asynchronous'
        forward, reverse, lower, upper = [], [], [], []
        drifting_by_probe = {}
        payoff.axhline(0, color='#555555', lw=.7, ls='--')
        for probe, (color, linestyle, marker) in zip(PROBES, STYLES):
            side = data['results'][probe]['candidate_invades_probe']
            curve = side['curve']
            x = np.array([row['mutant_count'] / n for row in curve])
            y = np.array([row['payoff_difference'] for row in curve])
            sd = np.array([row['payoff_difference_std'] for row in curve])
            assert len(x) == n - 1 and all(row['replicates'] == 5 for row in curve)
            payoff.fill_between(x, y-sd, y+sd, color=color, alpha=.12, lw=0)
            payoff.plot(x, y, color=color, ls=linestyle, marker=marker,
                        ms=2.2, markevery=2, lw=.95)
            bad_k = sorted({case['k'] for case in drift_checks(curve)['flagged']})
            drifting_by_probe[probe] = bad_k
            if bad_k:
                payoff.scatter([k/n for k in bad_k], [y[k-1] for k in bad_k],
                               s=19, marker='D', facecolors='none', edgecolors='#333333',
                               linewidths=.7, zorder=6)
            forward.append(side['rho'])
            reverse.append(data['results'][probe]['probe_invades_candidate']['rho'])
            lower.append(side['rho_at_minus_1sd'])
            upper.append(side['rho_at_plus_1sd'])
            assert lower[-1] <= forward[-1] <= upper[-1]
        forward, reverse, lower, upper = map(np.array, (forward, reverse, lower, upper))
        error = np.vstack((forward-lower, upper-forward))
        x = np.arange(3)
        full.bar(x-.18, forward, .34, color=FORWARD, linewidth=.4, edgecolor=FORWARD)
        full.bar(x+.18, reverse, .34, color=REVERSE, hatch='///', linewidth=.4, edgecolor=REVERSE)
        full.errorbar(x-.18, forward, yerr=error, fmt='none', ecolor='#222222', lw=.6, capsize=1.5)
        full.axhline(neutral, color='#555555', lw=.8, ls='--')
        ymax = max(upper.max(), reverse.max(), neutral) * 1.32
        full.set(ylim=(0, ymax), xticks=x, xticklabels=PROBES, xlim=(-.55, 2.65))
        # The near-neutral values are labelled in the zoom; ALLD values appear here.
        for offset, value, color in [(-.18, forward[2], FORWARD), (.18, reverse[2], REVERSE)]:
            label = f'{value:.3g}' if value < .0001 else f'{value:.4f}'
            full.annotate(label, (2+offset, value), xytext=(0, 5), textcoords='offset points',
                          ha='center', va='bottom', fontsize=8, color=color)
        zoom.axhline(neutral, color='#555555', lw=.8, ls='--')
        for j in range(2):
            zoom.errorbar(j-.13, forward[j], yerr=error[:, j:j+1], fmt='o', color=FORWARD,
                          ms=3, lw=.6, capsize=2)
            zoom.plot(j+.13, reverse[j], marker='s', ms=3, color=REVERSE)
            for offset, value, color, dy, va in [
                (-.13, forward[j], FORWARD, 7, 'bottom'),
                (.13, reverse[j], REVERSE, -7, 'top'),
            ]:
                zoom.annotate(f'{value:.6f}', (j+offset, value), xytext=(0, dy),
                              textcoords='offset points', ha='center', va=va, fontsize=8, color=color,
                              bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .3, 'alpha': .9})
        half_range = max(abs(lower[:2]-neutral).max(), abs(upper[:2]-neutral).max(),
                         abs(reverse[:2]-neutral).max(), .0001) * 1.55
        zoom.set(ylim=(neutral-half_range, neutral+half_range), xticks=[0, 1],
                 xticklabels=['L1', 'ALLC'], xlim=(-.58, 1.58))
        zoom.set_yticks([neutral-half_range*.65, neutral, neutral+half_range*.65])
        zoom.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        payoff.set(xlim=(0, 1), xticks=[0, .5, 1])
        payoff.text(-.31, .5, f'Seed {seed}', transform=payoff.transAxes,
                    rotation=90, ha='center', va='center', fontsize=8.5, weight='bold')
        for ax in (payoff, full, zoom):
            ax.tick_params(labelsize=8, length=2, pad=2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#dddddd', lw=.4)
            ax.set_axisbelow(True)
        payoff.yaxis.set_major_locator(MaxNLocator(3))
        full.yaxis.set_major_locator(MaxNLocator(3))
        if seed == 4:
            payoff.set_xlabel(r'Candidate share $k/N$', fontsize=8.5)
            full.set_xlabel('Opponent', fontsize=8.5)
            zoom.set_xlabel('Opponent', fontsize=8.5)
        records.append({'seed': seed, 'source': str(path.relative_to(ROOT)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'probes': list(PROBES), 'forward': forward.tolist(), 'reverse': reverse.tolist(),
                        'forward_sensitivity_lower': lower.tolist(), 'forward_sensitivity_upper': upper.tolist(),
                        'drifting_compositions': drifting_by_probe,
                        'zoom_ylim': list(zoom.get_ylim())})
    for ax, title in zip(axes[0], [r'(a) Payoff difference $\Delta\pi$', '(b) Fixation probability', '(c) Near-neutral detail']):
        ax.set_title(title, fontsize=8.5, pad=9)
    handles = [Line2D([], [], color=c, ls=ls, marker=m, ms=3, lw=1, label=p)
               for p, (c, ls, m) in zip(PROBES, STYLES)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, 1.002),
               ncol=3, frameon=False, fontsize=8, handlelength=1.5, columnspacing=.7)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=FORWARD, label='Candidate invades'),
                        Line2D([], [], marker='s', ls='', color=REVERSE, label='Opponent invades'),
                        Line2D([], [], color='#555555', ls='--', lw=.8, label=r'$1/N=0.05$')],
               loc='upper right', bbox_to_anchor=(1, 1.002), frameon=False, ncol=3,
               fontsize=8, handlelength=1.1, columnspacing=.6)
    return fig, records


def main():
    out = ROOT/'paper_zh/figures/manuscript_v2'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, records = build_figure()
        fig.savefig(out/'fixation.pdf', bbox_inches='tight')
        fig.savefig(out/'fixation.png', bbox_inches='tight', dpi=220)
        plt.close(fig)
    (out/'fixation_detail_provenance.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    print('Saved payoff curves, both fixation directions and near-neutral detail for five seeds.')


if __name__ == '__main__':
    main()
