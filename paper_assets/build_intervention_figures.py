from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':6.6,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(7.1,4.1),constrained_layout=True)
q=np.arange(4,512)
for row,model in enumerate(['qwen25','qwen3']):
 z=np.load(ROOT/'data'/'interventions'/model/'curve-summary.npz')
 def line(ax,key,j,label,color,start=0,ls='-'):
  ax.plot(q[start:],z[key+'_mean'][j,start:],label=label,color=color,lw=1.1,ls=ls)
  ax.fill_between(q[start:],z[key+'_lo'][j,start:],z[key+'_hi'][j,start:],color=color,alpha=.13,linewidth=0)
 for j,color in zip([1,2,3,4],['#56a36c','#1976b5','#b54a3c','#8a65b3']):
  line(axs[row,0],'boundary_local_tv',j,['','16 tokens','64 tokens','256 tokens','64 shuffled'][j],color,ls='--' if j==4 else '-')
 line(axs[row,1],'boundary_first_mass',0,'No prefix: first key','#444444',ls='--')
 line(axs[row,1],'boundary_first_mass',2,'64 prefix: new first','#1976b5')
 line(axs[row,1],'boundary_suffix_start_mass',2,'64 prefix: old first','#b54a3c')
 line(axs[row,2],'replacement_local_tv',0,'Replace token 0','#1976b5',start=1)
 line(axs[row,2],'replacement_local_tv',1,'Replace token 1','#d17828',start=1)
 for col in range(3):
  ax=axs[row,col];ax.set_xscale('log');ax.set_xticks([4,16,64,256],['4','16','64','256'])
  ax.grid(alpha=.15);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
  if row==1:ax.set_xlabel('Query coordinate in fixed passage')
 axs[row,0].set_ylabel(('Qwen2.5' if row==0 else 'Qwen3')+'\nLocal TV')
 axs[row,1].set_ylabel('Attention mass');axs[row,2].set_ylabel('Local TV')
for ax,title in zip(axs[0],['Preceding context','First-key attention','Token replacement']):ax.set_title(title)
fig.savefig(ROOT/'figures'/'interventions.pdf');fig.savefig(ROOT/'figures'/'interventions.png',dpi=200);plt.close(fig)
rows=[]
summaries={m:json.loads((ROOT/'data'/'interventions'/m/'summary.json').read_text()) for m in ['qwen3','qwen25']}
for k,label in enumerate(['Prefix response: early minus later','Replacement response: token 0 minus token 1']):
 values=[]
 for m in ['qwen3','qwen25']:
  e=summaries[m]['primary_endpoints'][k];lo,hi=e['familywise_ci']
  values.append(f"{e['mean']:.4f} [{lo:.4f}, {hi:.4f}]")
 rows.append(label+' & '+' & '.join(values)+r'\\')
(ROOT/'intervention_rows.tex').write_text('\n'.join(rows)+'\n')
