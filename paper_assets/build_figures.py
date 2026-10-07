"""Regenerate publication figures and numeric tables from the archived summaries."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
 'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.4})
models=[('qwen3','Qwen3-1.7B-Base','#23628a'),('qwen25','Qwen2.5-1.5B','#b95d29')]
curves={k:np.load(ROOT/'data'/k/'local-curves.npz') for k,_,_ in models}
pred={k:np.load(ROOT/'data'/k/'predictions-curves.npz') for k,_,_ in models}
summ={k:json.loads((ROOT/'data'/k/'summary.json').read_text()) for k,_,_ in models}
def save(fig,name):
 fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
 fig.savefig(ROOT/'figures'/f'{name}.png',dpi=170,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(6.75,2.25),layout='constrained')
for ax,j,title in zip(axes,[0,1,4],['Local logit gain','Local probability gain','Attention to first token']):
 for key,label,color in models:
  a=curves[key];start=4 if j<4 else 1;q=np.arange(start,1024)
  ax.plot(q,a['mean'][j,start:],color=color,label=label)
  ax.fill_between(q,a['low'][j,start:],a['high'][j,start:],color=color,alpha=.17,linewidth=0)
 ax.set(xscale='log',xlabel='Query position',title=title);ax.axhline(0,color='#888888',lw=.6)
 ax.set_xticks([4,32,256,1023],['4','32','256','1023']);ax.grid(axis='y',alpha=.16)
axes[0].legend(loc='upper right',frameon=False);save(fig,'boundary_profiles')
fig,axes=plt.subplots(2,2,figsize=(6.75,3.45),layout='constrained')
styles=[(0,'Relative only','#666666','--'),(1,'+ first-key bias','#23628a','-'),(2,'+ current-key bias','#b95d29',':')]
for row,(key,label,_) in enumerate(models):
 p=pred[key];q=np.arange(4,256)
 for j,name,c,ls in styles:axes[row,0].plot(q,p['mean'][j,4:256],label=name,color=c,ls=ls)
 for j,name,c,ls in [(6,'Observed','#222222','-'),(7,'Relative only','#888888','--'),(8,'+ first-key bias','#23628a','-')]:
  axes[row,1].plot(q,p['mean'][j,4:256],label=name,color=c,ls=ls)
 axes[row,0].set(ylabel=label+'\nTV error',ylim=(0,1));axes[row,1].set(ylabel='First-token mass',ylim=(0,1))
 for ax in axes[row]:ax.set(xlabel='Query position',xlim=(4,255));ax.grid(axis='y',alpha=.15)
axes[0,0].legend(frameon=False,loc='upper right');axes[0,1].legend(frameon=False,loc='upper right')
save(fig,'predictive_comparison')
fig,axes=plt.subplots(2,3,figsize=(7,4.5),layout='constrained')
for row,(key,label,_) in enumerate(models):
 h=np.load(ROOT/'data'/key/'head-endpoints.npz')['mean']
 for col,(j,title) in enumerate([(0,'Local logit early-late gain'),(2,'First vs. relative KL gain'),(4,'Late first-token mass')]):
  ax=axes[row,col];im=ax.imshow(h[:,j,:],origin='lower',aspect='auto',cmap='viridis')
  ax.set(title=title if row==0 else '',ylabel=label+'\nLayer' if col==0 else 'Layer',xlabel='Head')
  fig.colorbar(im,ax=ax,fraction=.05,pad=.03)
save(fig,'head_structure')
rows=[]
labels=['Local logit gain: early minus late','Local probability gain: early minus late','First-key bias vs. relative: KL gain','First-key bias vs. current-key bias: KL gain']
for i,label in enumerate(labels):
 entries=[]
 for key,_,_ in models:
  e=summ[key]['endpoints'][i];digits=5 if i==1 else 4
  entries.append(f"{e['mean']:.{digits}f} [{e['familywise_ci'][0]:.{digits}f}, {e['familywise_ci'][1]:.{digits}f}]")
 rows.append(label+' & '+' & '.join(entries)+r' \\')
(ROOT/'endpoint_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for key,label,_ in models:
 s=summ[key]
 for metric,name,digits in [('local_logit_gain','Logit',4),('local_probability_gain','Probability',5)]:
  v=s['local_gains'][metric];a=v['early']['mean'];b=v['late']['mean']
  rows.append(f'{label} & {name} & {a:.{digits}f} & {b:.{digits}f} & {a/b:.1f}'+r' \\')
(ROOT/'attenuation_rows.tex').write_text('\n'.join(rows)+'\n')
print('Figures and tables regenerated from archived test summaries.')
