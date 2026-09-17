"""Local 2x2 replay of severe cooperation drops; no LLM calls or new evolution."""
from pathlib import Path
import sys
import random
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_llm_reinitialization_control import CONDITIONS
from tools.prepare_manuscript_v2_evidence import read,write_json,write_csv,digest
from experiments.v2_quantitative.evolution_architecture import AgentSnapshot,FermiEvolutionRule,ReputationPrisonersDilemmaScenario
from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.executor import V2StrategyExecutor

OUT=ROOT/'results/cooperation_drop_replay_20260917'


def recover_seeds(data):
    cfg=data['config']
    assert cfg['agent_type']=='agent-type1'
    assert cfg['fallback_init_count']==cfg['fallback_mutation_count']==0
    rng=random.Random(cfg['seed'])
    rule=FermiEvolutionRule(beta=cfg['fermi_beta'],mutation_rate=cfg['mutation_rate_on_adoption'],
                           updates_per_gen=cfg['updates_per_gen'],init_source=cfg['fermi_init_source'])
    seeds=[];plans={}
    for g,entry in enumerate(data['trajectory']):
        seeds.append(rng.randrange(10**9))
        if g==99:
            break
        cohort=entry['population']
        snapshots=tuple(AgentSnapshot(a['agent_id'],a['code'],a['fitness'],a['lineage_id']) for a in cohort)
        plan=rule.plan(snapshots,rng=rng,next_gen=g+1,lineage_by_agent_id={a['agent_id']:a['lineage_id'] for a in cohort})
        nxt=data['trajectory'][g+1]['population']
        actual={i for i,a in enumerate(nxt) if a['birth_gen']==g+1}
        assert actual=={j.output_index for j in plan.jobs},(cfg['seed'],g,'slots')
        for job in plan.jobs:
            a=nxt[job.output_index]
            assert (a['origin'],a['parent_id'],a['parent_lineage_id'])==(job.origin,job.parent_id,job.parent_lineage_id)
        plans[g+1]=plan
    assert json.dumps(rng.getstate())==json.dumps(cfg['rng_state']), 'Final RNG state mismatch'
    return seeds,plans


def play(cohort,cfg,seed):
    random.seed(0)
    agents=[QuantitativeAgent(a['agent_id'],a['code'],V2StrategyExecutor(a['code'])) for a in cohort]
    scenario=ReputationPrisonersDilemmaScenario(**{k:cfg[k] for k in (
        'population_size','benefit','cost','observability','observability_p','fitness_window_fraction',
        'num_rounds_per_gen','action_error_probability','observation_error_probability')})
    return scenario.evaluate(agents,generation_seed=seed,num_rounds=cfg['num_rounds_per_gen'])


def replay(case):
    data=read(Path(case['source']))
    cfg=data['config'];g=case['generation']
    cohorts=[data['trajectory'][g-1]['population'],data['trajectory'][g]['population']]
    values={}
    for c in range(2):
        for s in range(2):
            result=play(cohorts[c],cfg,case['game_seeds'][s])
            values[f'c{c}_s{s}']=result.cooperation_rate_mean
            if c==s:
                actual=data['trajectory'][g-1+c]
                assert abs(result.cooperation_rate_mean-actual['cooperation_rate_mean'])<1e-12,(case['id'],c,'cooperation')
                assert all(abs(a['fitness']-p)<1e-12 for a,p in zip(actual['population'],result.payoffs)),(case['id'],c,'payoffs')
    row={k:v for k,v in case.items() if k not in ('game_seeds','source')}
    row.update(values)
    row['old_population_also_low_under_new_seed']=values['c0_s1']<=.3
    row['replacement_effect_at_new_seed']=values['c1_s1']-values['c0_s1']
    row['seed_effect_at_old_population']=values['c0_s1']-values['c0_s0']
    return row


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[];inputs=[]
    for condition,(_,_,folder,_,_) in CONDITIONS.items():
        for seed in range(5):
            path=folder/f'seed{seed}/evolutionary.json'
            data=read(path);seeds,plans=recover_seeds(data)
            inputs.append(dict(source=str(path),sha256=digest(path),rng_and_all_birth_plans_verified=True))
            for g in range(1,100):
                before=data['trajectory'][g-1]['cooperation_rate_mean']
                after=data['trajectory'][g]['cooperation_rate_mean']
                if before>=.8 and after<=.3:
                    jobs=plans[g].jobs
                    counts=Counter(j.operator for j in jobs)
                    before_pop=data['trajectory'][g-1]['population'];after_pop=data['trajectory'][g]['population']
                    cases.append(dict(id=f'{condition}_s{seed}_g{g}',condition=condition,seed=seed,generation=g,
                        source=str(path),game_seeds=[seeds[g-1],seeds[g]],
                        archived_before=before,archived_after=after,
                        llm_rewrites=counts['llm_mutate'],fresh_llm=counts['llm_init'],baseline_injections=counts['baseline_init'],
                        replaced_slots=len(jobs),changed_code_slots=sum(a['code']!=b['code'] for a,b in zip(before_pop,after_pop)),
                        following5_mean=sum(t['cooperation_rate_mean'] for t in data['trajectory'][g:g+5])/len(data['trajectory'][g:g+5])))
    write_json(OUT/'plan.json',dict(screen='previous cooperation >=0.8 and next <=0.3',
        design='old/new population programs crossed with old/new reconstructed generation seed; reset state in all cells',
        selected_cases=cases,inputs=inputs,script_sha256=digest(Path(__file__))))
    print(f'All 15 RNG streams and birth plans verified; replaying {len(cases)} severe-drop events ({4*len(cases)} fixed-population evaluations).',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(replay,c) for c in cases]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            print(row['id'], 'old/new at new seed:',round(row['c0_s1'],5),round(row['c1_s1'],5),flush=True)
    rows.sort(key=lambda r:(r['condition'],r['seed'],r['generation']))
    write_csv(OUT/'replay_results.csv',rows)
    summary=dict(n_events=len(rows),historical_cells_exactly_reproduced=2*len(rows),
        old_population_also_low=sum(r['old_population_also_low_under_new_seed'] for r in rows),
        by_condition={c:dict(n=sum(r['condition']==c for r in rows),
            old_population_also_low=sum(r['condition']==c and r['old_population_also_low_under_new_seed'] for r in rows)) for c in CONDITIONS})
    write_json(OUT/'summary.json',summary)
    lines=['# 合作骤降的本地交叉重放','',
        '筛查三组各5种子、共15条100代轨迹，选取前代合作率≥0.8、后代≤0.3的转折。这是探索性“骤降”定义，包括短暂低谷，不等于全部持续崩溃事件。','',
        '根据存档收益和Fermi规则恢复每代交互种子，逐代核对接受槽位、出生类型和亲代，并验证末态随机数状态。随后交叉运行前/后代程序群体 × 前/后代交互种子；每格仍从中性声誉重新开始，没有LLM调用，没有新演化。','',
        f"共{len(rows)}个骤降事件，{2*len(rows)}个历史对角格精确重现合作率与个体收益。其中{summary['old_population_also_low']}个事件中，保留前代所有程序、只更换为后代交互种子，合作率也降至≤0.3。",'',
        '| 条件/种子/代 | 原前代 | 原后代 | 前代群体+后代种子 | 后代群体+前代种子 | 改写/新LLM/基线注入 |',
        '|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['c0_s0']:.5f} | {r['c1_s1']:.5f} | {r['c0_s1']:.5f} | {r['c1_s0']:.5f} | {r['llm_rewrites']}/{r['fresh_llm']}/{r['baseline_injections']} |")
    lines+=['','这并未分离“改写了榜样代码”与“该榜样的后代替换了原槽位”的作用，也不能从同步多个替换中锁定某一个罪魁策略。交互种子同时决定配对、行动噪声与观察噪声；在不同程序下，实际执行及后续状态可能产生交互作用。','',
        '本重放能判断某个固定随机条件下整批替换是否促成下降，不能证明单个突变的普遍因果作用，也没有单独操纵代际重置。原程序本身可能已具备脆弱性，即使当前代不替换仍会下降；这不排除它的脆弱性来自更早的规则演化。','',
        '- [全部数值](replay_results.csv)','- [事件及输入溯源](plan.json)','- [汇总](summary.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
