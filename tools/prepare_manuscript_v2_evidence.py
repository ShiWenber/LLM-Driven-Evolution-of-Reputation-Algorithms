"""Copy and audit selected batches, without moving sources or calling an LLM."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import random
import re
import shutil
import statistics as stats
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.v2_quantitative.executor import V2StrategyExecutor
SOURCE = ROOT / 'results/quantitative_baseline'
OUT = ROOT / 'results/manuscript_v2'
BASE = 'LLM_agent-type1_fermi_baseline_init_g100_10000inter_N16_genreset_5seed'
ARMS = {
 'v4_flash': dict(model='v4-flash', label=BASE+'_arkv4flash',
  analysis='analysis_agent-type1_fermi_baseline_init_g100_10000inter_N16_arkv4flash_5seed',
  reports=['ARK_V4FLASH_BASELINE_INIT_5SEED_REPORT.md','ARK_V4FLASH_INVASION_FIXATION_REPORT.md'],
  invasion='n20_fermi_baseline_init_arkv4flash_10000inter_ae0p01_oe0p01',
  fixation='n20_fermi_baseline_init_arkv4flash_seed{seed}_ae0p01_oe0p01'),
 'v4_1_flash': dict(model='v4.1-flash',label=BASE,
  analysis='analysis_agent-type1_fermi_baseline_init_g100_10000inter_N16_5seed',
  reports=['SOCIAL_NORM_FORMATION_ASSESSMENT.md','LINEAGE_TREE_NORM_ANALYSIS.md',
           'FINAL_DOMINANT_FAMILY_JUDGEMENT.md','ALLD_MUTANT_AND_EXPLOITATION_REPORT.md',
           'EVOLUTION_INVASION_FIXATION_REPORT.md'],
  invasion='n20_fermi_baseline_init_10000inter_ae0p01_oe0p01',
  fixation='n20_fermi_baseline_init_seed{seed}_long_100k_300k_ae0p01_oe0p01')}

def read(path):
 return json.loads(path.read_text(encoding='utf-8-sig'))

def write_json(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def digest(path):
 with path.open('rb') as f:
  return hashlib.file_digest(f,'sha256').hexdigest()

def signature(code):
 """Reports' coarse 8+4 probe; NOT full continuous-program equivalence."""
 ex=V2StrategyExecutor(code); assessment=[]; action=[]
 for ar,act,br in [(0.5,'cooperate',1),(0.5,'cooperate',-1),(-0.5,'cooperate',1),
  (-0.5,'cooperate',-1),(0.5,'defect',1),(0.5,'defect',-1),(-0.5,'defect',1),(-0.5,'defect',-1)]:
  random.seed(0)
  delta=float(ex.observe(ar,act,br,'cooperate',0.0))-ar
  assessment.append('G' if delta>1e-9 else 'B' if delta < -1e-9 else '0')
 for sr,rr in [(1,1),(1,-1),(-1,1),(-1,-1)]:
  random.seed(0); action.append('C' if ex.decide(sr,rr) else 'D')
 return ''.join(assessment),''.join(action)

def audit_run(data,arm,seed):
 tr=data['trajectory']; cfg=data['config']
 assert len(tr)==100 and [g['generation'] for g in tr]==list(range(100))
 for k,v in dict(population_size=16,benefit=3.0,cost=1.0,updates_per_gen=16,
                fermi_init_source='baseline',action_error_probability=0.01,observation_error_probability=0.01).items():
  assert cfg[k]==v,(arm,seed,k,cfg[k])
 events={e['lineage_id']:e for e in data['lineage_events']}
 def root_of(lid):
  seen=set()
  while events[lid].get('parent_lineage_id') is not None:
   assert lid not in seen
   seen.add(lid); lid=events[lid]['parent_lineage_id']
  return lid
 final=data['final_population']
 assert [a['code'] for a in final]==[a['code'] for a in tr[-1]['population']]
 roots=Counter(root_of(a['lineage_id']) for a in final)
 row=dict(model=ARMS[arm]['model'],seed=seed,gen0_cooperation=tr[0]['cooperation_rate_mean'],
  gen99_cooperation=tr[-1]['cooperation_rate_mean'],
  last20_cooperation=stats.mean(g['cooperation_rate_mean'] for g in tr[-20:]),
  last20_within_run_sd=stats.stdev(g['cooperation_rate_mean'] for g in tr[-20:]),
  gen99_fitness=tr[-1]['fitness_mean'],fallback_init=cfg.get('fallback_init_count',0),
  fallback_mutation=cfg.get('fallback_mutation_count',0),birth_events=len(events),
  surviving_roots=len(roots),initial_root_final_agents=sum(n for lid,n in roots.items() if events[lid]['origin']=='initial'),
  injected_root_final_agents=sum(n for lid,n in roots.items() if events[lid]['origin']=='independent_init'),
  largest_root_family=max(roots.values()))
 ns=[]
 for g in [tr[0],tr[-1]]:
  for agent in g['population']:
   a,d=signature(agent['code'])
   ns.append(dict(model=row['model'],seed=seed,generation=g['generation'],agent_id=agent['agent_id'],
    assessment=a,action=d,signature=a+'|'+d,code_sha256=hashlib.sha256(agent['code'].encode()).hexdigest()))
 ls=[dict(model=row['model'],seed=seed,root_lineage_id=lid,root_origin=events[lid]['origin'],
  root_birth_gen=events[lid]['birth_gen'],final_members=n) for lid,n in roots.most_common()]
 return row,ns,ls

def copy_arm(arm,info):
 target=OUT/arm; mapping={}
 def add(src,dst):
  assert src.exists(),src
  if src.is_dir():
   for f in sorted(src.rglob('*')):
    if f.is_file() and '__pycache__' not in f.parts:
     mapping[f.resolve()]=(dst/f.relative_to(src)).resolve()
  else: mapping[src.resolve()]=dst.resolve()
 add(SOURCE/info['analysis'],target/'analysis')
 for seed in range(5):
  add(SOURCE/f"{info['label']}_seed{seed}",target/'runs'/f'seed{seed}')
  add(SOURCE/'fixation'/info['fixation'].format(seed=seed),target/'benchmarks/fixation'/f'seed{seed}')
 add(SOURCE/'invasion'/info['invasion'],target/'benchmarks/invasion')
 for report in info['reports']: add(SOURCE/report,target/'reports'/report)
 if arm=='v4_flash': add(SOURCE/'report_ark',target/'reports/report_ark')
 summary=SOURCE/(info['label']+'_summary.json')
 if summary.exists(): add(summary,target/'runs/original_summary.json')
 for src in list(mapping):
  if src.suffix.lower()!='.md': continue
  for link in re.findall(r'\]\(([^\n)]+)\)',src.read_text(encoding='utf-8-sig')):
   rel=link.split('#')[0].strip('<>')
   if not rel or '://' in rel: continue
   dep=(src.parent/rel).resolve()
   if dep.exists() and dep.is_file() and dep.suffix.lower() in {'.png','.pdf','.gif','.svg','.csv'} and dep not in mapping and dep.is_relative_to(ROOT):
    add(dep,target/'assets'/dep.relative_to(ROOT))
 records=[]
 for src,dst in mapping.items():
  dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
  sha=digest(dst); assert digest(src)==sha
  records.append(dict(source=src.relative_to(ROOT).as_posix(),copy=dst.relative_to(target).as_posix(),bytes=dst.stat().st_size,sha256=sha))
 # Navigable copies are separate from the byte-identical archived originals.
 for src,dst in mapping.items():
  if src.suffix.lower()!='.md': continue
  nav=dst.with_name(dst.stem+'.linked.md')
  def fix_link(m):
   raw=m.group(1)
   if '://' in raw or raw.startswith('#'): return m.group(0)
   path,sep,frag=raw.partition('#'); dep=(src.parent/path.strip('<>')).resolve()
   resolved=mapping.get(dep,dep)
   if resolved.exists(): return ']('+os.path.relpath(resolved,nav.parent).replace('\\','/')+(sep+frag if sep else '')+')'
   return m.group(0)
  text=re.sub(r'\]\(([^\n)]+)\)',fix_link,src.read_text(encoding='utf-8-sig'))
  nav.write_text('> 导航副本：仅调整有效本地链接。原报告保存在不带 `.linked` 的同名文件。\n\n'+text,encoding='utf-8')
 write_json(target/'source_manifest.json',dict(model_identity=info['model'],model_identity_authority='user-confirmed; raw aliases unchanged',source_run_label=info['label'],files=records))
 return target,records

AUDIT='''# 论文证据勘误与适用范围

本文件优先于历史报告中的推断性措辞；数值以 tables/ 中从原始 JSON 重算的表为准。

1. `fermi_init_source=baseline` 控制采纳后的0.1概率注入分支，gen0仍为LLM初始化。注入池为50% ALLD、50%均匀抽取L1–L8，不是gen0基线混合。
2. 实际主实验b=3、c=1、updates/gen=16、每代10000交互、窗口0.2、双噪声0.01。fitness为计入窗口的每次行动的平均收益。旧稿b=2、updates=4、无噪声及累计收益不可用于本批。
3. 最后一代与最后20代均值不同，run_summary.csv分别列出。独立复现单位是种子，不是80个程序。
4. 两模型后期均高合作不能识别选择的独立因果贡献。初始化、后代改写、注入和服务商差异未完全拆分；无中性替换对照，未做等效性检验。
5. 8位评估+4位行动探针固定对方动作为合作、自身评价为0；不覆盖完整连续程序，不能称精确规范等价，不能据此断言不存在新规范。
6. 谱系记录亲代条件生成来源。gen0祖先灭绝不等于gen0没有历史环境影响；来自注入根也不等于后代仍实现经典规范。
7. 两批聚类分别拟合，K及簇编号不能跨模型直接比较；簇名称是描述标签。
8. 归档入侵/fixation为b=2、c=1，主实验b=3，属于环境转移。2026-09-16后引擎默认随来源配置，复现旧数字必须显式--benefit 2 --cost 1。本次未重跑大规模实验。
9. 有限时间扫描是同步严格高收益模仿并每代重置；fixation是固定组成收益测量与beta=1概率过程。不能将其差异单独归因于选择强度。严格高收益平局不复制，也不完全等于Fermi极限。
10. 半窗口gap<=0.1只是漂移诊断通过，不是数学平稳证明；标准差敏感性带不是95%CI。rho低于中性仅表示稀有固定受抑制。两个方向必须分别读取。
11. 声誉矩阵图是末代程序的短程重放，不是训练直接保存的末代矩阵或稳态估计。
12. thinking失败/中止、type2、旧观察率批次不进入本文主证据。报告提到它们的历史段落保留，但不引用未完成终点。
'''

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 totals=[]; all_norms=[]; all_lineages=[]; all_fix=[]; curves={}; summary={}
 for arm,info in ARMS.items():
  target,manifest=copy_arm(arm,info)
  rows=[]; norms=[]; lineages=[]; fixes=[]; configs=[]; curves[arm]=[]
  for seed in range(5):
   data=read(target/'runs'/f'seed{seed}'/'evolutionary.json')
   row,ns,ls=audit_run(data,arm,seed)
   rows.append(row); norms.extend(ns); lineages.extend(ls)
   curves[arm].append([g['cooperation_rate_mean'] for g in data['trajectory']])
   configs.append(dict(seed_index=seed,**{k:v for k,v in data['config'].items() if k!='rng_state'}))
   bm=read(target/'benchmarks/fixation'/f'seed{seed}'/'fixation_benchmark.json')
   for probe,result in bm['results'].items():
    fixes.append(dict(model=info['model'],seed=seed,probe=probe,
     rho_candidate_to_probe=result['candidate_invades_probe']['rho'],
     rho_probe_to_candidate=result.get('probe_invades_candidate',{}).get('rho',''),
     neutral_rho=bm['config']['neutral_fixation_probability'],
     max_half_gap=bm['stationarity'][probe]['max_half_to_half_gap'],
     candidate_code_sha256=bm['candidate']['code_sha256'],evaluation_benefit=2,evaluation_cost=1))
   assert any(hashlib.sha256(a['code'].encode()).hexdigest()==bm['candidate']['code_sha256'] for a in data['final_population']),(arm,seed,'candidate hash mismatch')
  for name,items in [('run_summary',rows),('norm_probes',norms),('surviving_roots',lineages),('fixation_summary',fixes)]: write_csv(target/'tables'/f'{name}.csv',items)
  write_json(target/'tables/configs.json',configs)
  totals.extend(rows); all_norms.extend(norms); all_lineages.extend(lineages); all_fix.extend(fixes)
  pooled={}
  for gen in [0,99]:
   ns=[r for r in norms if r['generation']==gen]; cnt=Counter(r['signature'] for r in ns)
   pooled[str(gen)]=dict(n=len(ns),distinct_signatures=len(cnt),modal=cnt.most_common(1)[0],image_scoring_signature=cnt['GGGGBBBB|CDCD'],scoring_assessment=sum(r['assessment']=='GGGGBBBB' for r in ns),signature_counts=dict(cnt))
  summary[arm]=dict(model=info['model'],n_seeds=5,norms=pooled,
   metrics={k:dict(mean=stats.mean(r[k] for r in rows),sample_sd=stats.stdev(r[k] for r in rows)) for k in ['gen0_cooperation','gen99_cooperation','last20_cooperation','gen99_fitness']},
   initial_root_final_agents=sum(r['initial_root_final_agents'] for r in rows),injected_root_final_agents=sum(r['injected_root_final_agents'] for r in rows),
   fallback_mutation=sum(r['fallback_mutation'] for r in rows),stationarity_pass_pairs=sum(r['max_half_gap']<=0.1 for r in fixes),copied_files=len(manifest),copied_bytes=sum(r['bytes'] for r in manifest))
  text=f"# {info['model']}：实验结果与分析\n\n模型身份按用户确认归档；原始日志历史别名保留。\n\n"
  text+='## 从这里阅读\n\n- [本批分析导航](analysis/README.linked.md)\n- [原始分析报告](analysis/README.md)\n- [逐种子统计](tables/run_summary.csv)\n- [联合规范探针](tables/norm_probes.csv)\n- [末代谱系来源](tables/surviving_roots.csv)\n- [双向固定概率](tables/fixation_summary.csv)\n- [论文证据勘误](AUDIT.md)\n- [来源及SHA-256清单](source_manifest.json)\n\n## 专题报告\n\n'
  text+=''.join(f'- [{r}](reports/{Path(r).stem}.linked.md)\n' for r in info['reports'])
  text+='\n## 原始记录\n\n'+''.join(f'- [seed {s}](runs/seed{s}/evolutionary.json)；[fixation](benchmarks/fixation/seed{s}/fixation_benchmark.json)\n' for s in range(5))
  text+='\n[入侵扫描](benchmarks/invasion/summary.json)。runs目录保留原图表、动画、谱系及JSON。原文件仅复制，未移动或删除；旧绝对路径由来源清单与候选代码哈希追溯。\n'
  (target/'README.md').write_text(text,encoding='utf-8')
  extra='\n本批历史报告还存在“没有规范过半”与69%的种子数值矛盾、“无初代L规范匹配”与L3匹配矛盾等问题；论文使用统一重算值，避免绝对化结论。\n' if arm=='v4_1_flash' else ''
  (target/'AUDIT.md').write_text(AUDIT+extra,encoding='utf-8')
 write_json(OUT/'comparison_summary.json',summary); write_json(OUT/'cooperation_trajectories.json',curves)
 for name,rows in [('comparison_by_seed',totals),('norm_probes_all',all_norms),('surviving_roots_all',all_lineages),('fixation_all',all_fix)]: write_csv(OUT/f'{name}.csv',rows)
 (OUT/'README.md').write_text('# manuscript_v2：两模型实验归档\n\n- [v4-flash](v4_flash/README.md)\n- [v4.1-flash](v4_1_flash/README.md)\n- [统计汇总](comparison_summary.json)\n- [逐种子比较](comparison_by_seed.csv)\n\n原始资料仅复制。用户重复列出的规范报告收录一次。模型按用户确认分类。\n',encoding='utf-8')
 print(json.dumps({k:{x:y for x,y in v.items() if x!='norms'} for k,v in summary.items()},indent=2))
 print('norms',json.dumps({k:{g:{x:y for x,y in n.items() if x!='signature_counts'} for g,n in v['norms'].items()} for k,v in summary.items()}))

if __name__=='__main__': main()
