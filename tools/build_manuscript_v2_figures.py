"""Publication figures and tables derived only from the organized experiment files."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/manuscript_v2'
OUT=ROOT/'paper_zh/figures/manuscript_v2'
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
 provenance[name]={'inputs':inputs,'note':note}

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

scan=read(DATA/'v4_flash/benchmarks/invasion/summary.json')
bm=read(DATA/'v4_flash/benchmarks/fixation/seed2/fixation_benchmark.json')
fig,axs=plt.subplots(1,2,figsize=(9,2.8),layout='constrained')
for probe,col in [('L1','#247A91'),('ALLC','#8664A5'),('ALLD','#C96B35')]:
 cells=scan['groups']['seed2'][probe]; counts=sorted(map(int,cells))
 axs[0].plot(np.array(counts)/20,[cells[str(k)]['mean_final_invader_frequency'] for k in counts],marker='o',ms=3,lw=1.2,color=col,label=probe)
axs[0].plot([0,1],[0,1],ls='--',color='#999999',lw=.8)
axs[0].set(xlabel='Initial candidate fraction',ylabel='Mean fraction after 50 generations',title='(a) v4-flash seed 2: finite-time scan',xlim=(0,1),ylim=(-.04,1.04))
axs[0].legend(frameon=False,loc='lower right')
for probe,col in [('L1','#247A91'),('ALLC','#8664A5')]:
 curve=bm['results'][probe]['candidate_invades_probe']['curve']
 axs[1].plot([r['mutant_count']/20 for r in curve],[r['payoff_difference'] for r in curve],marker='o',ms=2.5,lw=1,color=col,label=probe)
axs[1].axhline(0,color='#999999',ls='--',lw=.8)
axs[1].set(xlabel='Fixed candidate fraction',ylabel='Mean payoff difference',title='(b) Fixed-composition payoff measurement',ylim=(-.001,.020))
axs[1].legend(frameon=False,loc='lower right')
save(fig,'transfer_case',['v4_flash/benchmarks/invasion/summary.json','v4_flash/benchmarks/fixation/seed2/fixation_benchmark.json'],'Same candidate hash; different update and reset protocols. N=20, b=2, c=1, 1% errors. Scan n=3; payoff n=5 replicates.')

fig,axs=plt.subplots(1,2,figsize=(9,2.8),layout='constrained')
for ax,key,title in [(axs[0],'rho_candidate_to_probe','(a) Candidate invades probe'),(axs[1],'rho_probe_to_candidate','(b) Probe invades candidate')]:
 for j,(probe,col,marker) in enumerate([('L1','#247A91','o'),('ALLC','#8664A5','s'),('ALLD','#C96B35','^')]):
  selected=sorted([r for r in fixes if r['model']=='v4-flash' and r['probe']==probe],key=lambda r:int(r['seed']))
  vals=[float(r[key])/.05 for r in selected]
  ax.scatter(np.arange(5)+(j-1)*.17,vals,s=33,color=col,marker=marker,label=probe,zorder=3)
 ax.axhline(1,color='#777777',ls='--',lw=1)
 ax.set(xticks=range(5),xlabel='Candidate evolution seed',ylabel='Fixation probability / (1/N)',title=title,ylim=(-.05,max(3.6,rho_max*1.25)))
 ax.legend(frameon=False,ncol=3,loc='upper left')
save(fig,'fixation',['fixation_all.csv'],'v4-flash only; both directions. L1 represents coincident L1-L8 results in this archive, not universal norm equivalence. Point estimates, not CIs.')

# Fully enumerated values retain small nonzero probabilities in scientific notation.
def texnum(x):
 x=float(x)
 if x==0: return '$0$'
 if 0<x<.0001:
  m,e=f'{x:.2e}'.split('e'); return f'${m}\\times10^{{{int(e)}}}$'
 return f'${x:.4f}$'
lines=[]
for name in NAMES:
 for seed in range(5):
  s=[r for r in fixes if r['model']==name and int(r['seed'])==seed]
  values=[]
  for probe in ['L1','ALLC','ALLD']:
   r=next(r for r in s if r['probe']==probe)
   for key in ['rho_candidate_to_probe','rho_probe_to_candidate']:
    v=texnum(r[key]); values.append('${'+v[1:-1]+'}^{\\dagger}$' if float(r['max_half_gap'])>.1 else v)
  lines.append(name+' & '+str(seed)+' & '+' & '.join(values)+' \\\\')
(WORK/'fixation_table_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
draft=(WORK/'new_manuscript.tex').read_text(encoding='utf-8')
(ROOT/'paper_zh/manuscript_v2.tex').write_text(draft.replace('\\input{manuscript_v2_work/fixation_table_rows.tex}', '\n'.join(lines)),encoding='utf-8')

keys={'nowak1998','nowak2005','ohtsuki2006','hilbe2018','schmid2023','fujimoto2024','traulsen2007','vallinder2024','horibe2026','funsearch2024','lear2025','rahwan2019','brinkmann2023','akata2025'}
bib=(ROOT/'paper_zh/references.bib').read_text(encoding='utf-8')
entries=[]
for entry in re.split(r'(?=^@)',bib,flags=re.M):
 match=re.match(r'@\w+\{([^,]+),',entry)
 if match and match.group(1) in keys: entries.append(entry.strip())
assert len(entries)==len(keys)
(ROOT/'paper_zh/manuscript_v2_refs.bib').write_text('\n\n'.join(entries)+'\n',encoding='utf-8')
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
print('Wrote five PDF/PNG figures, fixation table, and 14-entry bibliography.')
print('scan seed2 L1 counts',[(k,v['mean_final_invader_frequency']) for k,v in scan['groups']['seed2']['L1'].items()])
print('fixation seed2 ALLD',bm['results']['ALLD']['candidate_invades_probe']['rho'])
