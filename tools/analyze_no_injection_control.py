"""Compare completed official-DeepSeek mu=0 runs to their historical controls."""
from pathlib import Path
import sys
import json
import statistics as stats
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_manuscript_v2_evidence import audit_run, write_csv, write_json, digest, signature
from experiments.v2_quantitative.executor import V2StrategyExecutor
import random

OUT = ROOT / 'results/no_baseline_injection_20260917'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    plan = read(OUT / 'plan.json')
    assert len(plan['jobs']) == 5 and {j['provider'] for j in plan['jobs']} == {'deepseek'}
    all_rows, all_norms, all_roots, paired = [], [], [], []
    trajectories = {'Historical injection (mu=0.1)': [], 'No injection (mu=0)': []}
    api_usage = Counter()
    model_names = Counter()
    audit_counts = Counter()
    checks = []
    for job in plan['jobs']:
        folder = Path(job['folder'])
        source = read(Path(job['source']))
        data = read(folder / 'evolutionary.json')
        assert read(folder / 'status.json')['state'] == 'completed'
        cfg = data['config']
        assert cfg['mutation_rate_on_adoption'] == 0 and cfg['control']['gen0_replay_verified']
        assert digest(Path(job['source'])) == cfg['control']['source_sha256']
        assert cfg['control']['provider'] == 'deepseek'
        assert cfg['control']['base_url'].rstrip('/') == 'https://api.deepseek.com'
        unchanged = ('population_size', 'num_generations', 'benefit', 'cost', 'seed',
                     'target_interactions_per_gen', 'fitness_window_fraction', 'fermi_beta',
                     'updates_per_gen', 'llm_concurrency', 'agent_type', 'llm_model',
                     'fermi_init_source', 'imitation_learning_mode', 'mutation_temperature',
                     'llm_thinking', 'llm_max_tokens', 'action_error_probability',
                     'observation_error_probability', 'observability', 'observability_p')
        for key in unchanged:
            assert cfg[key] == source['config'][key], (job['seed'], key)
        assert data['trajectory'][0] == source['trajectory'][0]
        assert all(e['origin'] != 'independent_init' for e in data['lineage_events'])
        arm_rows = {}
        for condition, result in [('Historical injection (mu=0.1)', source), ('No injection (mu=0)', data)]:
            row, norms, roots = audit_run(result, 'v4_1_flash', job['seed'])
            for item in [row, *norms, *roots]:
                item['condition'] = condition
            all_rows.append(row)
            all_norms.extend(norms)
            all_roots.extend(roots)
            trajectories[condition].append([g['cooperation_rate_mean'] for g in result['trajectory']])
            arm_rows[condition] = row
        original_row, new_row = arm_rows.values()
        paired.append(dict(seed=job['seed'], gen0_cooperation=new_row['gen0_cooperation'],
                           historical_last20=original_row['last20_cooperation'],
                           no_injection_last20=new_row['last20_cooperation'],
                           difference_no_injection_minus_historical=new_row['last20_cooperation']-original_row['last20_cooperation'],
                           historical_final=original_row['gen99_cooperation'],
                           no_injection_final=new_row['gen99_cooperation'],
                           no_injection_surviving_initial_roots=new_row['surviving_roots'],
                           no_injection_fallbacks=new_row['fallback_mutation']))
        for line in (folder / 'api_audit.jsonl').read_text(encoding='utf-8').splitlines():
            call = json.loads(line)
            audit_counts['successful' if call['ok'] else 'failed'] += 1
            if call['ok']:
                model_names[call['response_model']] += 1
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    api_usage[key] += (call.get('usage') or {}).get(key, 0) or 0
        checks.append(f"seed{job['seed']}: 100 generations, exact gen0 match, unchanged comparison settings, source hash, complete ancestry and zero injected events verified.")
    summary = {}
    for condition in trajectories:
        rows = [r for r in all_rows if r['condition'] == condition]
        norms = [r for r in all_norms if r['condition'] == condition and r['generation'] == 99]
        summary[condition] = dict(
            last20_mean=stats.mean(r['last20_cooperation'] for r in rows),
            last20_sample_sd=stats.stdev(r['last20_cooperation'] for r in rows),
            final_mean=stats.mean(r['gen99_cooperation'] for r in rows),
            final_sample_sd=stats.stdev(r['gen99_cooperation'] for r in rows),
            initial_root_final_agents=sum(r['initial_root_final_agents'] for r in rows),
            injected_root_final_agents=sum(r['injected_root_final_agents'] for r in rows),
            final_pooled_signatures=len({r['signature'] for r in norms}),
            final_image_scoring_signature=sum(r['signature']=='GGGGBBBB|CDCD' for r in norms),
            fallback_mutations=sum(r['fallback_mutation'] for r in rows))
    summary['paired_difference'] = dict(
        mean=stats.mean(r['difference_no_injection_minus_historical'] for r in paired),
        sample_sd=stats.stdev(r['difference_no_injection_minus_historical'] for r in paired), n=5)
    summary['api'] = dict(calls={k:audit_counts[k] for k in ['successful', 'failed']}, returned_models=dict(model_names), usage=dict(api_usage),
                          note='Evolution requests only; excludes the single preflight request and any SDK-internal retries.')
    analysis = OUT / 'analysis'
    write_csv(analysis / 'paired_comparison.csv', paired)
    write_csv(analysis / 'run_summary.csv', all_rows)
    write_csv(analysis / 'norm_probes.csv', all_norms)
    write_csv(analysis / 'surviving_roots.csv', all_roots)
    write_json(analysis / 'summary.json', summary)

    # An observed collapse is retained and inspected, not excluded as an outlier.
    case = read(OUT / 'v4_1_flash/seed4/evolutionary.json')
    case_rows = []
    for generation in case['trajectory']:
        neutral_cooperators = 0
        scoring_like = 0
        for member in generation['population']:
            random.seed(0)
            neutral_cooperators += bool(V2StrategyExecutor(member['code']).decide(0, 0))
            scoring_like += '|'.join(signature(member['code'])) == 'GGGGBBBB|CDCD'
        case_rows.append(dict(generation=generation['generation'],
            cooperation=generation['cooperation_rate_mean'], neutral_cooperators=neutral_cooperators,
            image_scoring_like_signatures=scoring_like))
    write_csv(analysis / 'seed4_transition_probes.csv', case_rows)
    write_json(analysis / 'provenance.json', dict(
        analysis_script_sha256=digest(Path(__file__)),
        probe_script_sha256=digest(ROOT / 'tools/prepare_manuscript_v2_evidence.py'),
        inputs=[dict(path=str(Path(j['folder'])/'evolutionary.json'),
                     sha256=digest(Path(j['folder'])/'evolutionary.json')) for j in plan['jobs']]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios':[1.5, 1]})
    colors = ['#62788a', '#ca5434']
    for (condition, curves), color in zip(trajectories.items(), colors):
        arr = np.array(curves)
        for curve in arr:
            axes[0].plot(range(100), curve, color=color, alpha=.16, lw=.6)
        axes[0].plot(range(100), arr.mean(axis=0), color=color, lw=2, label=condition)
    axes[0].set(xlabel='Generation', ylabel='Cooperation rate', ylim=(-.025, 1.025))
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    label_y = {}
    previous_y = -1
    for r in sorted(paired, key=lambda x:x['no_injection_last20']):
        previous_y = max(r['no_injection_last20'], previous_y + .034)
        label_y[r['seed']] = previous_y
    shift = max(0, max(label_y.values()) - .995)
    for r in paired:
        ys = [r['historical_last20'], r['no_injection_last20']]
        axes[1].plot([0, 1], ys, color='#999999', alpha=.6, lw=.8)
        axes[1].scatter([0, 1], ys, c=colors, s=28, zorder=3)
        axes[1].annotate(f"s{r['seed']}", (1, ys[1]), xytext=(1.16, label_y[r['seed']] - shift),
                         fontsize=8, va='center', arrowprops={'arrowstyle':'-', 'lw':.5, 'color':'#777777'})
    axes[1].set(xticks=[0, 1], xticklabels=['Injection\n(historical)', 'No injection\n(new)'],
                ylabel='Mean cooperation, generations 80–99', xlim=(-.25, 1.5), ylim=(-.025, 1.025))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    fig.savefig(analysis / 'cooperation_comparison.png', dpi=180)
    fig.savefig(analysis / 'cooperation_comparison.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for (condition, curves), color in zip(trajectories.items(), colors):
        axes[0].plot(range(100), curves[4], color=color, lw=1.5, label=condition)
    axes[0].axvline(84, color='#444444', ls=':', lw=.8)
    axes[0].set(xlabel='Generation', ylabel='Seed 4 cooperation', ylim=(-.025, 1.025))
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    axes[1].plot(range(100), [r['neutral_cooperators'] for r in case_rows], label='Cooperate at neutral (0, 0)', color='#2a7d72')
    axes[1].plot(range(100), [r['image_scoring_like_signatures'] for r in case_rows], label='IS-like coarse signature', color='#7a589d')
    axes[1].set(xlabel='Generation', ylabel='Seed 4 programs (out of 16)', ylim=(-.5, 16.5))
    axes[1].legend(frameon=False, fontsize=8, loc='lower left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    fig.savefig(analysis / 'seed4_transition.png', dpi=180)
    fig.savefig(analysis / 'seed4_transition.pdf')
    plt.close(fig)

    old, new = [summary[k] for k in trajectories]
    difference = summary['paired_difference']['mean']
    lines = ['# DeepSeek官方模型：无基线注入对照结果', '',
        '5个种子均完成100代。初始程序逐种子复用原官方参考组，且第0代完整记录匹配；独立注入概率从0.1改为0，亲代条件LLM改写保留。', '',
        f"主要终点（第80–99代平均合作率；跨种子均值±样本标准差）：历史注入组 **{old['last20_mean']:.4f} ± {old['last20_sample_sd']:.4f}**，无注入组 **{new['last20_mean']:.4f} ± {new['last20_sample_sd']:.4f}**。逐种子差值的平均为 **{difference:+.4f}**（{difference*100:+.2f}个百分点）。", '',
        '| seed | 历史注入组后20代 | 无注入组后20代 | 差值 | 无注入末代 | fallback次数 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['seed']} | {r['historical_last20']:.4f} | {r['no_injection_last20']:.4f} | {r['difference_no_injection_minus_historical']:+.4f} | {r['no_injection_final']:.4f} | {r['no_injection_fallbacks']} |")
    lines += ['', '![合作轨迹及逐种子对照](cooperation_comparison.png)', '',
        '左图粗线表示五种子均值、细线表示各条轨迹。四个种子的后20代合作率高于各自历史对照，seed 4大幅下降，因而整体均值下降8.48个百分点。本报告不作统计显著性声明。', '',
        f"无注入组末代80个程序中，{new['initial_root_final_agents']}个来自初代根，{new['injected_root_final_agents']}个来自独立注入根；后者为0是干预正确执行的校验，不应当作额外的机制发现。末代跨种子合计有{new['final_pooled_signatures']}种粗探针签名（历史组{old['final_pooled_signatures']}种）。探针相同不代表完整程序等价。", '',
        '## seed 4的合作下降', '',
        'seed 4第83代合作率为0.98365，第84代降至0.24310，末代为0.00985；全程没有fallback。末代13/16个程序仍具有`GGGGBBBB|CDCD`粗签名。这是有限探针不能替代完整行为评估的具体例子，不意味着这些程序等同于经典Image Scoring。', '',
        '补充的中性状态探针`decide(0,0)`显示，选择合作的程序数从第83代10/16下降到末代2/16。这与合作下降同期发生，但尚未通过状态访问测量或干预实验识别因果机制。该探针属于观察到下降后的探索性分析，不是预先指定的主要终点。', '',
        '![seed4合作轨迹与探索性探针](seed4_transition.png)', '',
        '## 解释边界', '',
        '本结果检验官方参考组在移除外源基线注入后的表现。种子间存在明显分化，因此不能笼统断言移除注入没有影响，也不能断言所有种子都需要外源注入才能维持合作。它不能回答从v4低合作初始群体出发是否也能达到高合作。当前没有运行Ark/v4组。', '',
        '历史对照和新实验不在同一时间运行，官方模型别名不能独立保证服务版本一致；收益选择的独立作用仍需中性替换对照。五个种子是重复单位，80个程序不是80个独立重复。此处报告描述性差值，不将其直接表述为无混杂的因果效应。', '',
        f"API返回模型计数：`{dict(model_names)}`。生成调用成功{audit_counts['successful']}次、失败{audit_counts['failed']}次；token用量见[summary.json](summary.json)，不含连通性检查及SDK内部重试。", '',
        '## 验证', '', *['- '+check for check in checks], '',
        '- [逐种子对照](paired_comparison.csv)', '- [完整统计](run_summary.csv)',
        '- [规则探针](norm_probes.csv)', '- [祖先来源](surviving_roots.csv)']
    (analysis / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
