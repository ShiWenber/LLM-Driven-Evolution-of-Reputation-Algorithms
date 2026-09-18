"""Compare the completed synchronous control with its archived async cohort."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/synchronous_control_20260917'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(data):
    trajectory = data['trajectory']
    assert len(trajectory) == 100
    assert [t['generation'] for t in trajectory] == list(range(100))
    assert all(t['n_interactions'] == 10000 for t in trajectory)
    curve = [t['cooperation_rate_mean'] for t in trajectory]
    return dict(final=curve[-1], late20=statistics.mean(curve[-20:]),
                whole100=statistics.mean(curve), minimum=min(curve),
                low_generations=sum(c <= .3 for c in curve),
                sharp_drops=sum(a >= .8 and b <= .3 for a, b in zip(curve, curve[1:])))


def main():
    plan = read(OUT / 'plan.json')
    for item in plan['code_manifest']:
        assert hashlib.sha256((OUT / 'code_snapshot' / item['source']).read_bytes()).hexdigest() == item['sha256']
    rows, audits, curves = [], [], {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for job in plan['jobs']:
        folder = Path(job['folder'])
        assert read(folder / 'status.json')['state'] == 'completed'
        source = Path(job['source'])
        old, new = read(source), read(folder / 'evolutionary.json')
        cfg, old_cfg = new['config'], old['config']
        assert cfg['observation_schedule'] == 'synchronous'
        assert old_cfg['observation_schedule'] == 'asynchronous'
        assert cfg['num_rounds_per_gen'] == 1250
        for key in ('population_size', 'target_interactions_per_gen', 'num_generations',
                    'fitness_window_fraction', 'benefit', 'cost', 'seed', 'llm_model',
                    'action_error_probability', 'observation_error_probability',
                    'mutation_rate_on_adoption', 'fermi_init_source', 'fermi_beta',
                    'imitation_learning_mode', 'updates_per_gen', 'mutation_temperature',
                    'llm_thinking', 'llm_max_tokens', 'observability', 'observability_p'):
            assert cfg[key] == old_cfg[key], (job['seed'], key)
        assert [(a['agent_id'], a['code']) for a in old['trajectory'][0]['population']] == [
            (a['agent_id'], a['code']) for a in new['trajectory'][0]['population']]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == cfg['control']['source_sha256']
        requests = [json.loads(line) for line in (folder / 'api_audit.jsonl').read_text(encoding='utf-8').splitlines()]
        births = [json.loads(line) for line in (folder / 'offspring_jobs.jsonl').read_text(encoding='utf-8').splitlines()]
        assert all(b['operator'] in ('llm_init', 'llm_mutate') for b in births)
        audits.append(dict(seed=job['seed'], requests=len(requests),
                           errors=sum(not r['ok'] for r in requests),
                           fallback_mutation=cfg['fallback_mutation_count'],
                           fallback_init=cfg['fallback_init_count'],
                           returned_models=sorted({r.get('response_model') for r in requests if r['ok']}),
                           tokens=sum((r.get('usage') or {}).get('total_tokens', 0) for r in requests),
                           births=len(births), fresh_initializations=sum(b['operator']=='llm_init' for b in births),
                           source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           output_sha256=hashlib.sha256((folder / 'evolutionary.json').read_bytes()).hexdigest()))
        a, b = metrics(old), metrics(new)
        row = {'seed': job['seed']}
        row.update({'async_' + k: v for k, v in a.items()})
        row.update({'sync_' + k: v for k, v in b.items()})
        row['late20_difference'] = b['late20'] - a['late20']
        rows.append(row)
        ax = axes.flat[job['seed']]
        curves[job['seed']] = {}
        for name, data, color in [('Asynchronous', old, '#d97a28'), ('Synchronous', new, '#2874a6')]:
            values = [t['cooperation_rate_mean'] for t in data['trajectory']]
            curves[job['seed']][name] = values
            ax.plot(range(100), values, color=color, lw=1.3, alpha=.85, label=name)
        ax.axvspan(80, 99, color='grey', alpha=.1)
        ax.set(title=f'Seed {job["seed"]}', xlabel='Generation', ylabel='Cooperation', ylim=(-.03, 1.03))
    ax = axes.flat[5]
    for row in rows:
        ax.plot([0, 1], [row['async_late20'], row['sync_late20']], marker='o', label=f'Seed {row["seed"]}')
    ax.set(xticks=[0, 1], xticklabels=['Asynchronous', 'Synchronous'], ylabel='Last 20 generations: mean cooperation', ylim=(-.03, 1.03))
    ax.legend(fontsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Matched initial strategies: interaction protocol comparison (5 seeds)')
    fig.savefig(OUT / 'comparison.png', dpi=180)
    fig.savefig(OUT / 'comparison.pdf')
    plt.close(fig)
    with (OUT / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for label in ('async', 'sync'):
        summary[label] = {metric: {'mean': statistics.mean(r[label+'_'+metric] for r in rows),
                                  'sample_sd': statistics.stdev(r[label+'_'+metric] for r in rows)}
                          for metric in ('late20', 'final', 'whole100')}
        summary[label]['sharp_drops_total'] = sum(r[label+'_sharp_drops'] for r in rows)
        summary[label]['low_generations_total'] = sum(r[label+'_low_generations'] for r in rows)
    summary['paired_late20_difference'] = statistics.mean(r['late20_difference'] for r in rows)
    (OUT / 'comparison_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUT / 'verification.json').write_text(json.dumps(dict(initial_programs_and_matched_parameters_verified=True,
        audits=audits, code_manifest=plan['code_manifest']), indent=2), encoding='utf-8')
    lines = ['# 同步交互对照实验', '',
        '对照：`results/llm_reinitialization_20260917/` 的异步实验。新实验使用同一批初始策略、相同 5 个种子和 DeepSeek 官方接口；N=16、100 代、每代 10,000 次配对交互、μ=0.1，独立新策略由 LLM 初始化。', '',
        '同步恢复为历史随机匹配：每轮 8 对全部行动之后，再按原顺序更新参与者及第三方观察；每代 1,250 轮。保留当前执行/观察噪声各 0.01、末 20% 的每行动平均适应度，以及每代声誉归零。并未回退历史代码中旧的适应度总和算法。', '',
        '历史实现核验：以 `29503a7^` 为参照，在 N=15/16、full/private/partial 观察和 3 个随机种子的 18 组无噪声检查中，行动日志、收益、声誉矩阵和 RNG 状态均逐项一致。记录见 historical_equivalence.json。', '',
        '主终点为第 80–99 代合作率的均值，统计单位为独立演化种子；表中 SD 为 5 个种子的样本标准差。', '',
        '| 指标 | 异步 | 同步 |', '|---|---:|---:|']
    for metric, title in [('late20', '末20代合作率'), ('final', '末代合作率'), ('whole100', '全100代合作率')]:
        a, b = summary['async'][metric], summary['sync'][metric]
        lines.append(f'| {title}，均值 ± SD | {a["mean"]:.5f} ± {a["sample_sd"]:.5f} | {b["mean"]:.5f} ± {b["sample_sd"]:.5f} |')
    for metric, title in [('sharp_drops_total', '从 ≥0.8 跌至 ≤0.3 的相邻代事件数'), ('low_generations_total', '合作率 ≤0.3 的代数（总计500代）')]:
        lines.append(f'| {title} | {summary["async"][metric]} | {summary["sync"][metric]} |')
    lines += ['', '| 种子 | 异步末20代 | 同步末20代 | 同步−异步 |', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["async_late20"]:.5f} | {r["sync_late20"]:.5f} | {r["late20_difference"]:+.5f} |')
    lines += ['', f'平均配对差：{summary["paired_late20_difference"]:+.5f}。', '',
        '**解释边界：**此干预同时改变了配对（独立抽样→每轮无重复匹配）和观察更新时间，LLM 的规则说明也相应改变。相同随机种子不代表相同交互事件流；LLM 输出本身亦有随机性。因此，这批结果比较的是完整演化协议，不能单独证明“交互随机种子影响”增加或减少。只有 5 个演化重复，SD 的变化是描述性证据，不作显著性或普遍稳定性结论。急降事件包括随后恢复的暂时低谷，不等于永久崩溃。', '',
        f'API 审计：{sum(a["requests"] for a in audits)} 次生成请求，{sum(a["errors"] for a in audits)} 次请求错误；{sum(a["fallback_mutation"] for a in audits)} 次突变回退；报告总 token 数 {sum(a["tokens"] for a in audits):,}（不含接口预检）。返回模型名称见 verification.json，服务别名不等于独立固定的模型版本。', '',
        '![合作曲线及逐种子比较](comparison.png)', '',
        '结果：comparison.csv、comparison_summary.json；审计：verification.json、plan.json、code_snapshot/、各种子 api_audit.jsonl 与 offspring_jobs.jsonl。']
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
