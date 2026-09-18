"""Plot verified injection-source trajectories and frozen-population reputations."""
from pathlib import Path
import json, hashlib, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.analysis.consensus.core import disagreement
DATA=ROOT/'results/manuscript_v2/injection_source_comparison_20260918'
OUT=ROOT/'paper_zh/figures/manuscript_v2_supplement'
COLORS={'baseline':'#247A91','llm':'#C96B35'}
LABELS={'baseline':'Baseline injection','llm':'Fresh LLM injection'}

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    evolution=read(DATA/'evolution_summary.json');consensus=read(DATA/'consensus_summary.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':10,
        'axes.labelsize':9.5,'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240})
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
    for c in COLORS:
        arr=np.array(evolution['curves'][c]);mean=arr.mean(0);sd=arr.std(0,ddof=1)
        axs[0,0].plot(range(100),mean,color=COLORS[c],label=LABELS[c],lw=1.5,
                      linestyle='-' if c=='llm' else '--')
        axs[0,0].fill_between(range(100),np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=COLORS[c],alpha=.13)
    axs[0,0].axvspan(80,99,color='gray',alpha=.1)
    axs[0,0].set(title='(a) Evolutionary cooperation',xlabel='Generation',ylabel='Cooperation',ylim=(0,1.03),xlim=(0,99))
    axs[0,0].legend(frameon=False,loc='lower right')
    marks=['o','s','^','D','v']
    for row,marker in zip(evolution['paired'],marks):
        y=[row['baseline_late'],row['llm_late']]
        axs[0,1].plot([0,1],y,color='#aaa',lw=.9)
        axs[0,1].scatter([0,1],y,c=list(COLORS.values()),marker=marker,s=29,zorder=3)
        axs[0,1].annotate(str(row['seed']),(.5,np.mean(y)),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    axs[0,1].set(title='(b) Paired late cooperation',ylabel='Mean of generations 80–99',
                  xticks=[0,1],xticklabels=['Baseline','Fresh LLM'],xlim=(-.22,1.3),ylim=(.8,1.01))
    axs[0,1].text(.03,.96,'Labels: evolutionary seed',transform=axs[0,1].transAxes,va='top',fontsize=8)
    for k,c in enumerate(COLORS):
        rows=[r for r in consensus['trajectory'] if r['condition']==c]
        x=np.array([r['generation'] for r in rows]);mean=np.array([r['mean'] for r in rows]);sd=np.array([r['sample_sd'] for r in rows])
        axs[1,0].plot(x,mean,color=COLORS[c],marker='o' if k==0 else 's',ms=3,lw=1.2,label=LABELS[c])
        axs[1,0].fill_between(x,np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=COLORS[c],alpha=.13)
    axs[1,0].set(title='(c) Frozen-population agreement',xlabel='Generation of saved programs',ylabel='Consensus C = 1 − D',ylim=(0,1.03),xlim=(0,99))
    categories=[('Initial','#aaa',evolution['pooled']['baseline']['norms']['0']),
                ('Baseline',COLORS['baseline'],evolution['pooled']['baseline']['norms']['99']),
                ('Fresh LLM',COLORS['llm'],evolution['pooled']['llm']['norms']['99'])]
    x=np.arange(2)
    for k,(label,col,stats) in enumerate(categories):
        heights=[stats['image_scoring_joint'],stats['conditional_action']]
        bars=axs[1,1].bar(x+(k-1)*.25,heights,width=.23,color=col,label=label)
        axs[1,1].bar_label(bars,fontsize=8,padding=2)
    axs[1,1].set(title='(d) Assessment and action probes',ylabel='Programs (of 80)',xticks=x,
                  xticklabels=['Scoring-like\njoint rule','Conditional\naction'],ylim=(0,104))
    axs[1,1].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.01),fontsize=8,handlelength=.8,columnspacing=.6)
    for ext in ('pdf','png'):fig.savefig(OUT/f'injection_source_comparison.{ext}',bbox_inches='tight')
    plt.close(fig)

    fig,axs=plt.subplots(2,5,figsize=(7.2,3.65),layout='constrained')
    cmap=plt.colormaps['RdBu'].copy();cmap.set_bad('#bdbdbd')
    for row,c in enumerate(COLORS):
        for s in range(5):
            rep=read(DATA/'replays'/c/f'seed{s}_g99_r0.json')
            matrix=np.array(rep['matrix_final']);mask=np.eye(16,dtype=bool)
            ax=axs[row,s]
            im=ax.imshow(np.ma.array(matrix,mask=mask),vmin=-1,vmax=1,cmap=cmap,interpolation='nearest')
            score=1-disagreement(matrix,np.array(rep['observed_final']))
            ax.set_title(f'Seed {s}\nC = {score:.3f}',fontsize=9)
            ax.set(xticks=[0,7,15],yticks=[0,7,15],xticklabels=[1,8,16],yticklabels=[1,8,16])
            ax.tick_params(length=2,labelsize=8.5)
            if s==0:ax.set_ylabel(('Baseline' if c=='baseline' else 'Fresh LLM')+'\nObserver',fontsize=9)
            if row==1:ax.set_xlabel('Target',fontsize=9)
    cbar=fig.colorbar(im,ax=axs,location='bottom',shrink=.65,pad=.045,aspect=35,ticks=[-1,-.5,0,.5,1])
    cbar.set_label('Private reputation (self scores masked in gray)',fontsize=9)
    for ext in ('pdf','png'):fig.savefig(OUT/f'injection_reputation_matrices.{ext}',bbox_inches='tight')
    plt.close(fig)
    inputs=[DATA/'evolution_summary.json',DATA/'consensus_summary.json']+[
        DATA/'replays'/c/f'seed{s}_g99_r0.json' for c in COLORS for s in range(5)]
    records={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT/'injection_comparison_provenance.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print('Wrote two supplementary injection-source figures; all five evolutionary seeds included.')
    return records

if __name__=='__main__':build()
