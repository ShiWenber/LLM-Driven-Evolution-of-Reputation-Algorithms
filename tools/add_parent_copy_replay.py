"""Cancel contemporaneous parental rewriting while retaining accepted slots/injections."""
from pathlib import Path
import sys
import csv
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_cooperation_drops import OUT,read,play,write_csv,write_json,digest


def replay_parent_copy(case):
    data=read(Path(case['source']));g=case['generation']
    old={a['agent_id']:a for a in data['trajectory'][g-1]['population']}
    cohort=[dict(a) for a in data['trajectory'][g]['population']]
    count=0
    for agent in cohort:
        if agent['birth_gen']==g and agent['origin']=='imitate':
            agent['code']=old[agent['parent_id']]['code']
            count+=1
    assert count==case['llm_rewrites']
    result=play(cohort,data['config'],case['game_seeds'][1])
    return dict(id=case['id'],condition=case['condition'],seed=case['seed'],generation=g,
                canceled_rewrites=count,actual_cooperation=case['archived_after'],
                copy_parent_cooperation=result.cooperation_rate_mean,
                still_low_without_rewriting=result.cooperation_rate_mean<=.3,
                preserved_independent_injections=case['fresh_llm']+case['baseline_injections'])


def main():
    cases=read(OUT/'plan.json')['selected_cases']
    with ProcessPoolExecutor(max_workers=5) as pool:
        rows=list(pool.map(replay_parent_copy,cases))
    rows.sort(key=lambda r:(r['condition'],r['seed'],r['generation']))
    write_csv(OUT/'parent_copy_results.csv',rows)
    write_json(OUT/'parent_copy_provenance.json',dict(script_sha256=digest(Path(__file__)),
        design='Preserve actual accepted replacement slots and actual independent injections; replace every parent-conditioned child with its role model original code; use actual post-drop generation seed and neutral initial state.',
        n_evaluations=len(rows),still_low_without_rewriting=sum(r['still_low_without_rewriting'] for r in rows)))
    lines=['# 取消当代亲代改写的补充重放','',
        '固定实际接受的替换槽位、榜样和独立注入程序；将所有亲代条件子代换成榜样原代码，使用实际后一代的交互种子及中性声誉起点。不重新调用LLM，不改变此前演化。', '',
        '| 事件 | 实际合作率 | 取消亲代改写后 | 保留的独立注入数 |','|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['actual_cooperation']:.5f} | {r['copy_parent_cooperation']:.5f} | {r['preserved_independent_injections']} |")
    lines+=['','如果取消改写后仍骤降，说明在这个固定条件下，当代亲代代码改写不是骤降的必要条件。榜样原程序的复制仍会改变群体组成；独立初始化/基线注入和交互随机性依旧保留。反之，恢复高合作支持当代这批改写在本随机条件下促成下降，但不能锁定单个策略，也不意味着该策略在所有环境中都会破坏合作。','',
            '[原始2×2交叉重放](REPORT.md)']
    (OUT/'PARENT_COPY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Without rewriting, still low:',sum(r['still_low_without_rewriting'] for r in rows),'/',len(rows))
    for r in rows:
        print(r)


if __name__=='__main__':
    main()
