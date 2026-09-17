"""Audit completed LLM reinitialization and compare the three matched cohorts."""
from pathlib import Path
from collections import Counter
import json
import statistics as stats
import sys
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_manuscript_v2_evidence import signature, digest, write_csv, write_json
from experiments.v2_quantitative.executor import V2StrategyExecutor

OUT = ROOT / 'results/llm_reinitialization_20260917'
CONDITIONS = {
    'baseline': ('Classical injection (mu=0.1)', '基线注入0.1', ROOT/'results/manuscript_v2/v4_1_flash/runs', .1, 'baseline'),
    'none': ('No injection (mu=0)', '无注入', ROOT/'results/no_baseline_injection_20260917/v4_1_flash', 0., 'baseline'),
    'llm': ('Fresh LLM injection (mu=0.1)', 'LLM独立初始化0.1', OUT/'v4_1_flash', .1, 'llm'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def jsonlines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def summarize(data, condition, seed):
    tr = data['trajectory']
    assert len(tr) == 100 and [g['generation'] for g in tr] == list(range(100))
    assert data['final_population'] == tr[-1]['population']
    cfg = data['config']
    assert cfg['mutation_rate_on_adoption'] == CONDITIONS[condition][3]
    assert cfg['fermi_init_source'] == CONDITIONS[condition][4]
    events = {e['lineage_id']:e for e in data['lineage_events']}
    assert len(events) == len(data['lineage_events'])
    def root(lid):
        seen = set()
        while events[lid]['parent_lineage_id'] is not None:
            assert lid not in seen
            seen.add(lid)
            parent = events[lid]['parent_lineage_id']
            assert events[parent]['birth_gen'] < events[lid]['birth_gen']
            lid = parent
        return lid
    roots = Counter(root(a['lineage_id']) for a in data['final_population'])
    row = dict(condition=condition, seed=seed, gen0=tr[0]['cooperation_rate_mean'],
        last20=stats.mean(g['cooperation_rate_mean'] for g in tr[-20:]),
        final=tr[-1]['cooperation_rate_mean'], final_fitness=tr[-1]['fitness_mean'],
        independent_births=sum(e['origin']=='independent_init' for e in events.values()),
        parent_conditioned_births=sum(e['origin']=='imitate' for e in events.values()),
        initial_root_descendants=sum(n for lid,n in roots.items() if events[lid]['origin']=='initial'),
        injected_root_descendants=sum(n for lid,n in roots.items() if events[lid]['origin']=='independent_init'),
        surviving_roots=len(roots), fallback_init=cfg['fallback_init_count'],
        fallback_mutation=cfg['fallback_mutation_count'])
    norms = []
    for generation in (tr[0],tr[-1]):
        for agent in generation['population']:
            assessment, action = signature(agent['code'])
            random.seed(0)
            neutral = bool(V2StrategyExecutor(agent['code']).decide(0.,0.))
            norms.append(dict(condition=condition,seed=seed,generation=generation['generation'],
                agent_id=agent['agent_id'],signature=assessment+'|'+action,neutral_cooperate=neutral))
    root_rows = [dict(condition=condition,seed=seed,root_lineage_id=lid,
                      root_origin=events[lid]['origin'],root_birth_gen=events[lid]['birth_gen'],final_members=n)
                 for lid,n in roots.most_common()]
    return row,norms,root_rows


def validate_jobs(data, records):
    births = [e for e in data['lineage_events'] if e['birth_gen']>0]
    assert len(records)==len(births)
    assert len({(r['generation'],r['output_index']) for r in records})==len(records)
    assert {r['operator'] for r in records}=={'llm_init','llm_mutate'}
    by_origin = Counter(e['origin'] for e in births)
    by_operator = Counter(r['operator'] for r in records)
    assert by_origin['independent_init']==by_operator['llm_init']
    assert by_origin['imitate']==by_operator['llm_mutate']
    for job in records:
        generation=data['trajectory'][job['generation']]
        agent=generation['population'][job['output_index']]
        assert agent['birth_gen']==job['generation'] and agent['origin']==job['origin']
        assert agent['parent_id']==job['parent_id'] and agent['parent_lineage_id']==job['parent_lineage_id']
        if job['operator']=='llm_init':
            assert job['parent_id'] is None and job['parent_lineage_id'] is None
        if data['config']['fallback_mutation_count']==0:
            import hashlib
            assert job['code_generated'] and hashlib.sha256(agent['code'].encode()).hexdigest()==job['code_sha256']
    return by_operator


def main():
    plan=read(OUT/'plan.json')
    assert len(plan['jobs'])==5 and {j['provider'] for j in plan['jobs']}=={'deepseek'}
    for c in plan['code_manifest']:
        assert digest(ROOT/c['source'])==c['sha256']
    rows=[]; norms=[]; roots=[]; verified=[]; inputs=[]
    curves={k:[] for k in CONDITIONS}
    operator_totals=Counter(); api_counts=Counter(); api_models=Counter(); usage=Counter()
    for seed in range(5):
        old=read(CONDITIONS['baseline'][2]/f'seed{seed}/evolutionary.json')
        for condition,(_,_,folder,_,_) in CONDITIONS.items():
            path=folder/f'seed{seed}/evolutionary.json'
            data=read(path)
            inputs.append(dict(path=str(path),sha256=digest(path)))
            assert data['trajectory'][0]==old['trajectory'][0]
            keys=('population_size','num_generations','benefit','cost','seed','llm_model',
                  'target_interactions_per_gen','fitness_window_fraction','fermi_beta',
                  'updates_per_gen','llm_concurrency','agent_type','imitation_learning_mode',
                  'mutation_temperature','llm_thinking','llm_max_tokens',
                  'action_error_probability','observation_error_probability','observability','observability_p')
            for key in keys:
                assert data['config'][key]==old['config'][key],(condition,seed,key)
            row,ns,rs=summarize(data,condition,seed)
            rows.append(row);norms.extend(ns);roots.extend(rs)
            curves[condition].append([g['cooperation_rate_mean'] for g in data['trajectory']])
            if condition=='llm':
                assert read(path.parent/'status.json')['state']=='completed'
                assert digest(CONDITIONS['baseline'][2]/f'seed{seed}/evolutionary.json')==data['config']['control']['source_sha256']
                counts=validate_jobs(data,jsonlines(path.parent/'offspring_jobs.jsonl'))
                operator_totals.update(counts)
                for call in jsonlines(path.parent/'api_audit.jsonl'):
                    assert call['operator'] in ('llm_init','llm_mutate')
                    api_counts['success' if call['ok'] else 'failure']+=1
                    api_counts[call['operator']]+=1
                    if call['ok']:
                        api_models[call['response_model']]+=1
                        for key in ('prompt_tokens','completion_tokens','total_tokens'):
                            usage[key]+=(call.get('usage') or {}).get(key,0) or 0
                verified.append(f"seed {seed}: gen0精确匹配，100代完成，{counts['llm_init']}次独立LLM初始化、{counts['llm_mutate']}次亲代改写，无baseline_init；出生记录及亲代信息逐项核验。")
    summary={}
    for condition in CONDITIONS:
        selected=[r for r in rows if r['condition']==condition]
        final_norms=[n for n in norms if n['condition']==condition and n['generation']==99]
        summary[condition]=dict(n=5,last20_mean=stats.mean(r['last20'] for r in selected),
            last20_sample_sd=stats.stdev(r['last20'] for r in selected),
            final_mean=stats.mean(r['final'] for r in selected),final_sample_sd=stats.stdev(r['final'] for r in selected),
            final_pooled_signatures=len({n['signature'] for n in final_norms}),
            final_image_scoring_like=sum(n['signature']=='GGGGBBBB|CDCD' for n in final_norms),
            initial_root_descendants=sum(r['initial_root_descendants'] for r in selected),
            independent_root_descendants=sum(r['injected_root_descendants'] for r in selected),
            independent_births=sum(r['independent_births'] for r in selected),
            parent_conditioned_births=sum(r['parent_conditioned_births'] for r in selected),
            fallback_init=sum(r['fallback_init'] for r in selected),fallback_mutation=sum(r['fallback_mutation'] for r in selected))
    paired=[]
    for seed in range(5):
        values={r['condition']:r for r in rows if r['seed']==seed}
        paired.append(dict(seed=seed,baseline_last20=values['baseline']['last20'],
            no_injection_last20=values['none']['last20'],llm_init_last20=values['llm']['last20'],
            llm_minus_baseline=values['llm']['last20']-values['baseline']['last20'],
            llm_minus_no_injection=values['llm']['last20']-values['none']['last20'],
            llm_init_final=values['llm']['final'],llm_independent_births=values['llm']['independent_births']))
    summary['new_run_audit']=dict(operators=dict(operator_totals),
        empirical_independent_share=operator_totals['llm_init']/sum(operator_totals.values()),
        calls={k:api_counts[k] for k in ('success','failure','llm_init','llm_mutate')},
        response_models=dict(api_models),usage=dict(usage),
        usage_scope='Excludes preflight and SDK-internal retries.')
    analysis=OUT/'analysis'
    for filename,records in [('run_summary',rows),('norm_probes',norms),('surviving_roots',roots),('paired_comparison',paired)]:
        write_csv(analysis/f'{filename}.csv',records)
    write_json(analysis/'summary.json',summary)
    write_json(analysis/'provenance.json',dict(inputs=inputs,analysis_script_sha256=digest(Path(__file__)),
        probe_script_sha256=digest(ROOT/'tools/prepare_manuscript_v2_evidence.py')))
    plot_results(analysis,curves,paired)
    write_report(analysis,summary,paired,verified)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def plot_results(analysis,curves,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    colors={'baseline':'#657b8d','none':'#ce5738','llm':'#167d76'}
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True,sharey=True)
    for i,ax in enumerate(axes.flat):
        for condition in CONDITIONS:
            ys=curves[condition][i] if i<5 else np.mean(curves[condition],axis=0)
            ax.plot(range(100),ys,color=colors[condition],lw=1.2,label=CONDITIONS[condition][0])
        ax.set(title=f'Seed {i}' if i<5 else 'Mean of five seeds',ylim=(-.025,1.025),xlabel='Generation',ylabel='Cooperation rate')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1))
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','pdf'):
        fig.savefig(analysis/f'three_condition_trajectories.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4))
    for row in paired:
        ys=[row['baseline_last20'],row['no_injection_last20'],row['llm_init_last20']]
        ax.plot([0,1,2],ys,marker='o',ms=5,lw=1,label=f"Seed {row['seed']}")
    ax.set(xticks=[0,1,2],xticklabels=['Classical injection\nmu=0.1','No injection\nmu=0','Fresh LLM injection\nmu=0.1'],
           ylabel='Mean cooperation, generations 80–99',ylim=(-.025,1.025))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    ax.legend(frameon=False,loc='lower left',ncol=2,fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(analysis/f'paired_late_cooperation.{ext}',dpi=180)
    plt.close(fig)


def write_report(analysis,summary,paired,verified):
    lines=['# LLM独立初始化0.1：三组对照结果','',
        '本次仅使用DeepSeek官方接口，5个种子全部完成100代。初代程序及第0代完整记录与原参考组一致。每次Fermi接受替换后，以0.1概率使用原始初始化提示词重新生成策略，不提供亲代代码/收益；0.9概率进行亲代条件改写。','',
        '## 主要终点','',
        '第80–99代平均合作率。下表的标准差是五个种子间的样本标准差，不是标准误或置信区间。','',
        '| 条件 | 后20代均值 | 种子间标准差 | 末代均值 |','|---|---:|---:|---:|']
    for key,(_,label,_,_,_) in CONDITIONS.items():
        s=summary[key]
        lines.append(f"| {label} | {s['last20_mean']:.5f} | {s['last20_sample_sd']:.5f} | {s['final_mean']:.5f} |")
    lines+=['','| seed | 基线注入 | 无注入 | LLM独立初始化 | 新组−基线 | 新组−无注入 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['seed']} | {r['baseline_last20']:.5f} | {r['no_injection_last20']:.5f} | {r['llm_init_last20']:.5f} | {r['llm_minus_baseline']:+.5f} | {r['llm_minus_no_injection']:+.5f} |")
    d1=summary['llm']['last20_mean']-summary['baseline']['last20_mean']
    d2=summary['llm']['last20_mean']-summary['none']['last20_mean']
    lines+=['',f'新组相对历史基线注入组的均值差为{d1*100:+.2f}个百分点；相对本日无注入组为{d2*100:+.2f}个百分点。以上为描述性比较，不作统计显著性声明。','',
        '![各个种子与组均值的完整轨迹](three_condition_trajectories.png)','',
        '完整轨迹显示，新组部分种子也经历过较深的合作低谷，随后恢复。高后期合作率不等于全程稳定；尚不能将恢复直接归因于某一次LLM注入，需进一步进行事件与状态分析或受控重放。', '',
        '![后20代逐种子比较](paired_late_cooperation.png)','',
        '## 规则与谱系','',
        '| 条件 | 末代合计粗签名数 | 初代根后代/80 | 独立注入根后代/80 |','|---|---:|---:|---:|']
    for key,(_,label,_,_,_) in CONDITIONS.items():
        s=summary[key]
        lines.append(f"| {label} | {s['final_pooled_signatures']} | {s['initial_root_descendants']} | {s['independent_root_descendants']} |")
    lines+=['','所有条件使用相同8位评估+4位行动探针；这不是完整程序等价判断。基线组的独立根来自经典池，新组的独立根来自LLM重新初始化，两者来源不同，不能仅凭相同的`independent_init`标签混为一类。','',
        '## 执行核验','',*['- '+v for v in verified]]
    audit=summary['new_run_audit']
    lines+=['',f"实际独立初始化占所有已接受出生事件的{audit['empirical_independent_share']:.2%}。0.1是每次接受后的抽样概率，不要求每代或每条轨迹恰有10%。",
        f"API调用成功{audit['calls']['success']}次、失败{audit['calls']['failure']}次；返回模型名：`{audit['response_models']}`。新组fallback：初代{summary['llm']['fallback_init']}次，演化期间{summary['llm']['fallback_mutation']}次。",
        '', '## 解释边界','',
        '三组比较可以区分“没有独立更新”和“独立更新来自不同来源”的表现。历史基线组与本日两组并非同期随机分配，官方模型别名也不独立固定服务版本；随机过程会在不同分支后发生分歧。不能仅凭均值差确认基线注入或LLM注入的因果机制。', '',
        '本次复用了初始已高度合作的官方参考组，不能据此回答从v4低合作初始群体出发会发生什么。谱系比例不是选择机制证据，合作率也不是外部入侵稳定性的替代指标。没有新增中性选择或外部入侵实验。','',
        '- [逐种子比较](paired_comparison.csv)','- [完整统计](run_summary.csv)','- [API用量及汇总](summary.json)',
        '- [功能探针](norm_probes.csv)','- [末代祖先来源](surviving_roots.csv)','- [输入哈希与分析溯源](provenance.json)']
    (analysis/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
