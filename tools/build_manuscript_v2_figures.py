"""Publication figures and tables derived only from the organized experiment files."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plot_manuscript_fixation import build_figure as build_fixation_figure
DATA=ROOT/'results/manuscript_v2'
MAIN_OUT=ROOT/'paper_zh/figures/manuscript_v2'
SUPP_OUT=ROOT/'paper_zh/figures/manuscript_v2_supplement'
MAIN_OUT.mkdir(parents=True,exist_ok=True)
OUT=SUPP_OUT
WORK=ROOT/'paper_zh/manuscript_v2_work'
OUT.mkdir(parents=True,exist_ok=True)
WORK.mkdir(parents=True,exist_ok=True)
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def csvread(path):
 with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
summary=read(DATA/'comparison_summary.json')
trajectories=read(DATA/'cooperation_trajectories.json')
rows=csvread(DATA/'comparison_by_seed.csv')
norms=csvread(DATA/'norm_probes_all.csv')
roots=csvread(DATA/'surviving_roots_all.csv')
fixes=csvread(DATA/'fixation_all.csv')
ARMS=['v4_flash','v4_1_flash']; NAMES=['v4-flash','v4.1-flash']; COLORS=['#247A91','#C96B35']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
 'axes.spines.right':False,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240})
provenance={}
def save(fig,name,inputs,note):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(OUT/(name+'.png'),bbox_inches='tight')
 plt.close(fig)
 provenance[OUT.name+'/'+name]={'inputs':inputs,'note':note}

fig,axs=plt.subplots(1,3,figsize=(10,2.8),gridspec_kw={'width_ratios':[1,1,0.8]},layout='constrained')
for idx,(arm,name,col) in enumerate(zip(ARMS,NAMES,COLORS)):
 arr=np.array(trajectories[arm]); ax=axs[idx]
 for seed,line in enumerate(arr): ax.plot(range(100),line,color=col,alpha=0.25,lw=0.7)
 mean=arr.mean(0); sd=arr.std(0,ddof=1)
 ax.fill_between(range(100),np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=col,alpha=.15)
 ax.plot(mean,color=col,lw=1.7,label='Mean; band = seed SD')
 ax.set(xlabel='Generation',ylabel='Cooperation' if idx==0 else '',ylim=(-.02,1.04),title=f'({chr(97+idx)}) {name}')
 ax.legend(loc='lower right',frameon=False)
 selected=[r for r in rows if r['model']==name]
 for r in selected:
  axs[2].plot([idx-.13,idx+.13],[float(r['gen0_cooperation']),float(r['last20_cooperation'])],color=col,alpha=.55,lw=.8)
 axs[2].scatter([idx-.13]*5,[float(r['gen0_cooperation']) for r in selected],marker='o',facecolors='white',edgecolors=col,s=25,zorder=3)
 axs[2].scatter([idx+.13]*5,[float(r['last20_cooperation']) for r in selected],marker='s',color=col,s=25,zorder=3)
axs[2].set(xticks=[0,1],xticklabels=NAMES,ylim=(-.02,1.04),ylabel='Cooperation',title='(c) Initial and late')
axs[2].text(.5,.07,'Circle: generation 0\nSquare: mean of 80–99',ha='center',transform=axs[2].transAxes,fontsize=7.5)
save(fig,'cooperation',['comparison_by_seed.csv','cooperation_trajectories.json'],'Five seeds per model; descriptive sample SD; no thinking trajectories.')

fig,axs=plt.subplots(1,2,figsize=(9,2.8),layout='constrained')
for ax,name,col in zip(axs,NAMES,COLORS):
 for seed in range(5):
  selected=[r for r in roots if r['model']==name and int(r['seed'])==seed]
  selected.sort(key=lambda r:int(r['final_members']),reverse=True)
  bottom=0
  for j,r in enumerate(selected):
   count=int(r['final_members']); init=r['root_origin']=='initial'
   color='#687B8A' if init else plt.cm.Blues(.35+.5*j/max(len(selected)-1,1))
   ax.bar(seed,count,bottom=bottom,color=color,edgecolor='white',width=.68,hatch='///' if init else None)
   if count>=2: ax.text(seed,bottom+count/2,str(count),ha='center',va='center',fontsize=8,color='white' if not init and j>2 else '#132B3A')
   bottom+=count
  ax.text(seed,16.3,f'{len(selected)} roots',ha='center',fontsize=7)
 ax.set(xticks=range(5),xlabel='Evolution seed',ylabel='Final population members',ylim=(0,18),title=name)
save(fig,'ancestry',['surviving_roots_all.csv'],'Segments are recorded ancestry, not inferred code clusters. Root colors do not identify shared roots across seeds.')

fig,axs=plt.subplots(1,2,figsize=(9,2.8),layout='constrained')
rho_max=max(float(r[k])/.05 for r in fixes if r['model']=='v4-flash' for k in ['rho_candidate_to_probe','rho_probe_to_candidate'])
for idx,(arm,name,col) in enumerate(zip(ARMS,NAMES,COLORS)):
 for gen,x in [(0,idx-.18),(99,idx+.18)]:
  n=summary[arm]['norms'][str(gen)]; frac=n['image_scoring_signature']/80
  axs[0].bar(x,frac,width=.3,color=col,alpha=.45 if gen==0 else 1)
  axs[0].text(x,frac+.025,f"{n['image_scoring_signature']}/80",ha='center',fontsize=8)
  vals=[]
  for seed in range(5):
   selected=[r for r in norms if r['model']==name and int(r['generation'])==gen and int(r['seed'])==seed]
   vals.append(len(set(r['signature'] for r in selected)))
  axs[1].scatter(np.linspace(x-.05,x+.05,5),vals,color=col,marker='o' if gen==0 else 's',s=26,alpha=.75)
 axs[1].plot([idx-.18,idx+.18],[statistics.mean([len({r['signature'] for r in norms if r['model']==name and int(r['generation'])==g and int(r['seed'])==s}) for s in range(5)]) for g in [0,99]],color=col,lw=1)
axs[0].set(title='(a) Image-scoring-like joint signature',ylabel='Fraction of 80 programs',ylim=(0,.85),xticks=[0,1],xticklabels=NAMES)
axs[1].set(title='(b) Within-population signature diversity',ylabel='Distinct signatures (of 16)',xticks=[0,1],xticklabels=NAMES,ylim=(0,17))
for ax in axs: ax.text(.5,.98,'Left: generation 0   |   Right: generation 99',transform=ax.transAxes,ha='center',va='top',fontsize=7)
save(fig,'norm_signatures',['norm_probes_all.csv','comparison_summary.json'],'Coarse 8+4 evaluation only; pooled program counts are descriptive, not independent replication.')

# Main-paper plots use the v4 cohort only. Comparison plots above are supplemental.
OUT=MAIN_OUT
name='v4-flash'; col=COLORS[0]
selected=[r for r in rows if r['model']==name]
fig,axs=plt.subplots(1,2,figsize=(8,2.7),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
arr=np.array(trajectories['v4_flash'])
for line in arr: axs[0].plot(range(100),line,color=col,alpha=.3,lw=.7)
mean=arr.mean(0); sd=arr.std(0,ddof=1)
axs[0].fill_between(range(100),np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=col,alpha=.15)
axs[0].plot(mean,color=col,lw=1.7,label='Mean; band = seed SD')
axs[0].set(xlabel='Generation',ylabel='Cooperation',ylim=(-.02,1.04),title='(a) Cooperation trajectories')
axs[0].legend(loc='lower right',frameon=False)
for r in selected:
 vals=[float(r['gen0_cooperation']),float(r['last20_cooperation'])]
 axs[1].plot([0,1],vals,color=col,alpha=.5,lw=.9)
 axs[1].scatter([0,1],vals,color=col,s=24,zorder=3)
axs[1].set(xticks=[0,1],xticklabels=['Generation 0','Mean of 80–99'],xlim=(-.2,1.2),ylim=(-.02,1.04),ylabel='Cooperation',title='(b) Initial and late cooperation')
save(fig,'cooperation',['v4_flash/tables/run_summary.csv','v4_flash/runs/seed0-4/evolutionary.json'],'Main paper: v4-flash only; all five archived seeds; sample SD, not confidence intervals.')

fig,ax=plt.subplots(figsize=(6.5,2.5),layout='constrained')
for seed in range(5):
 selected=sorted([r for r in roots if r['model']==name and int(r['seed'])==seed],key=lambda r:int(r['final_members']),reverse=True)
 bottom=0
 for j,r in enumerate(selected):
  count=int(r['final_members']); init=r['root_origin']=='initial'
  color='#687B8A' if init else plt.cm.Blues(.35+.5*j/max(len(selected)-1,1))
  ax.bar(seed,count,bottom=bottom,color=color,edgecolor='white',width=.68,hatch='///' if init else None)
  if count>=2: ax.text(seed,bottom+count/2,str(count),ha='center',va='center',fontsize=8,color='white' if not init and j>2 else '#132B3A')
  bottom+=count
 ax.text(seed,16.3,f'{len(selected)} roots',ha='center',fontsize=8)
ax.set(xticks=range(5),xlabel='Evolution seed',ylabel='Final population members',ylim=(0,18))
save(fig,'ancestry',['v4_flash/tables/surviving_roots.csv'],'Main paper: v4-flash only; hatch denotes initial ancestry; colors identify roots within a seed.')

fig,axs=plt.subplots(1,2,figsize=(8,2.7),layout='constrained')
for i,gen in enumerate([0,99]):
 count=summary['v4_flash']['norms'][str(gen)]['image_scoring_signature']
 axs[0].bar(i,count/80,width=.55,color=col,alpha=.45 if gen==0 else 1)
 axs[0].text(i,count/80+.015,f'{count}/80',ha='center',fontsize=9)
for seed in range(5):
 vals=[len({r['signature'] for r in norms if r['model']==name and int(r['generation'])==g and int(r['seed'])==seed}) for g in [0,99]]
 axs[1].plot([0,1],vals,marker='o',color=col,alpha=.5,lw=.9)
for ax in axs: ax.set(xticks=[0,1],xticklabels=['Generation 0','Generation 99'],xlim=(-.55,1.55))
axs[0].set(title='(a) Image-scoring-like joint signature',ylabel='Fraction of 80 programs',ylim=(0,.5))
axs[1].set(title='(b) Within-population signature diversity',ylabel='Distinct signatures (of 16)',ylim=(0,17))
save(fig,'norm_signatures',['v4_flash/tables/norm_probes.csv'],'Main paper: v4-flash only; five paired seed counts; pooled counts are not independent replicates.')

scan=read(DATA/'v4_flash/benchmarks/invasion/summary.json')
bm=read(DATA/'fixation_b3_20260918/seed2/fixation_benchmark.json')
assert bm['config']['benefit'] == 3 and bm['config']['cost'] == 1
fig,axs=plt.subplots(1,2,figsize=(9,2.8),layout='constrained')
for probe,col in [('L1','#247A91'),('ALLC','#8664A5'),('ALLD','#C96B35')]:
 cells=scan['groups']['seed2'][probe]; counts=sorted(map(int,cells))
 axs[0].plot(np.array(counts)/20,[cells[str(k)]['mean_final_invader_frequency'] for k in counts],marker='o',ms=3,lw=1.2,color=col,label=probe)
axs[0].plot([0,1],[0,1],ls='--',color='#999999',lw=.8)
axs[0].set(xlabel='Initial candidate fraction',ylabel='Mean fraction after 50 generations',title='(a) Finite-time scan (b=2, c=1)',xlim=(0,1),ylim=(-.04,1.04))
axs[0].legend(frameon=False,loc='lower right')
for probe,col in [('L1','#247A91'),('ALLC','#8664A5')]:
 curve=bm['results'][probe]['candidate_invades_probe']['curve']
 axs[1].plot([r['mutant_count']/20 for r in curve],[r['payoff_difference'] for r in curve],marker='o',ms=2.5,lw=1,color=col,label=probe)
axs[1].axhline(0,color='#999999',ls='--',lw=.8)
axs[1].set(xlabel='Fixed candidate fraction',ylabel='Mean payoff difference',title='(b) Fixed-composition payoffs (b=3, c=1)')
axs[1].legend(frameon=False,loc='lower right')
save(fig,'transfer_case',['v4_flash/benchmarks/invasion/summary.json','fixation_b3_20260918/seed2/fixation_benchmark.json'],'Same candidate hash; scan b=2 and fixation b=3, c=1, N=20, 1% errors. Different update and reset protocols; scan n=3, payoff n=5 replicates.')

with plt.rc_context({'font.size': 8}):
 fig,fixation_details=build_fixation_figure()
 save(fig,'fixation',[record['source'] for record in fixation_details],
      'Five v4-flash seeds at b=3, c=1: payoff mean +/- replicate SD, both fixation directions, and near-neutral L1/ALLC zooms with six-decimal labels. Row-specific limits. Forward whiskers are payoff-spread sensitivity, not CIs.')
(OUT/'fixation_detail_provenance.json').write_text(json.dumps(fixation_details,indent=2)+'\n',encoding='utf-8')

from tools.analyze_synchronous_evolution import analyze as build_evolution_comparison
with plt.rc_context({'font.size':8.5,'axes.titlesize':9}):
 evolution_comparison=build_evolution_comparison()
provenance[MAIN_OUT.name+'/evolution_protocol_comparison']={
 'inputs':[record['path'] for record in evolution_comparison['sources']],
 'note':'One current synchronous evolutionary run and five historical asynchronous runs; model endpoints and initial programs differ. No synchronous seed-variance estimate.'}

# Figure generation never overwrites authored manuscripts or bibliographies.
from tools.analyze_worst_fitness_injection import analyze as build_worst_fitness_comparison
with plt.rc_context():
 worst_comparison=build_worst_fitness_comparison()
provenance[MAIN_OUT.name+'/worst_fitness_injection']={
 'inputs':[record['path'] for record in worst_comparison['sources']],
 'note':'Ten new paired runs; once-per-generation worst-fitness injection versus per-accepted-update initialization. Both placement and injection supply differ.'}
from tools.plot_injection_comparison import build as build_injection_comparison
with plt.rc_context():
 injection_inputs=build_injection_comparison()
for name in ['injection_source_comparison','injection_reputation_matrices']:
 provenance[SUPP_OUT.name+'/'+name]={
  'inputs':list(injection_inputs),
  'note':'Matched high-initial-cooperation cohort; five evolutionary seeds per injection source. Reputation measurements are frozen-population replays, not historical matrices. Nested replicates averaged within evolutionary seed.'}
for folder in [MAIN_OUT,SUPP_OUT]:
 records={key.split('/',1)[1]:value for key,value in provenance.items() if key.startswith(folder.name+'/')}
 (folder/'provenance.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
print('Wrote seven main-paper and five supplementary PDF/PNG figures.')
