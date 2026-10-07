import json
from pathlib import Path
import numpy as np
from scipy import stats
from design import band

ENDPOINTS = ['prefix64_local_tv_early_minus_late', 'replacement0_minus_replacement1_local_tv_queries5_63']

def holm(pvalues):
    p = np.asarray(pvalues,dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]).clip(0,1)
    result = np.empty(len(p));result[order] = adjusted
    return result

def summarize(x, family_size):
    x = np.asarray(x,dtype=float)
    n = len(x)
    if n < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite document measurements required')
    mean,sd = float(x.mean()),float(x.std(ddof=1))
    se = sd/np.sqrt(n)
    if se == 0:
        p,t = (1.,0.) if mean == 0 else (0.,np.copysign(np.inf,mean))
    else:
        t = mean/se;p = float(2*stats.t.sf(abs(t),n-1))
    ci = stats.t.ppf(1-.05/(2*family_size),n-1)*se
    return {'n_documents':n,'mean':mean,'sd':sd,'familywise_ci':[mean-ci,mean+ci],
            'p_two_sided':p, 't':float(t) if np.isfinite(t) else None,
            'note':'Degenerate sample variance; inference not calibrated' if se==0 else None}

def bootstrap_curves(values, replicates, seed):
    values = np.asarray(values,dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    # Multinomial counts retain whole documents and avoid a huge B x N x Q array.
    means = np.empty((replicates,)+values.shape[1:],dtype=np.float32)
    flat = values.reshape(n,-1)
    for start in range(0,replicates,50):
        count = min(50,replicates-start)
        weights = rng.multinomial(n,np.full(n,1/n),size=count)/n
        means[start:start+count] = (weights @ flat).reshape((count,)+values.shape[1:])
    return values.mean(axis=0),np.quantile(means,.025,axis=0),np.quantile(means,.975,axis=0)

def analyze(root,cfg,stage,stage_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model_stats, combined_p, entries = {},[],[]
    size = 2*len(cfg['models'])
    for model in cfg['models']:
        directory = stage_dir/model
        paths = sorted((directory/'documents').glob('*.npz'))
        if len(paths) < 2:
            print(f'{model}: only {len(paths)} completed documents; no inference yet',flush=True)
            continue
        manifest = json.loads((directory/'manifest.json').read_text())
        expected_ids = manifest['document_ids']
        data = {k:[] for k in ['endpoints','boundary_local_tv','boundary_first_mass','boundary_suffix_start_mass',
                             'replacement_local_tv','replacement_full_tv','replacement_first_mass_change',
                             'position_shift_local_tv','position_shift_full_tv','position_shift_boundary_endpoint']}
        ids = []
        for path in paths:
            idx = int(path.stem)
            with np.load(path,allow_pickle=False) as z:
                doc = str(z['document_id'].item())
                if idx >= len(expected_ids) or doc != expected_ids[idx] or str(z['run_identity'].item()) != manifest['identity']:
                    raise ValueError(f'Result identity mismatch: {path}')
                ids.append(doc)
                for k in data:
                    a = z[k]
                    data[k].append(a if k in {'endpoints','position_shift_boundary_endpoint'} else (a.mean(axis=0) if k.startswith('position_shift_') else a.mean(axis=1)))
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate document results')
        data = {k:np.stack(v) for k,v in data.items()}
        records = []
        for j,name in enumerate(ENDPOINTS):
            entry = {'endpoint':name,**summarize(data['endpoints'][:,j],size)}
            records.append(entry);entries.append(entry);combined_p.append(entry['p_two_sided'])
        expected = len(expected_ids)
        summary = {'model':model,'stage':stage,'status':cfg['status'], 'n_documents':len(ids),
                   'expected_documents':expected,'complete':len(ids)==expected,
                   'independent_unit':'source document; donors, heads and positions averaged within document',
                   'inference':'conditional on fixed donor pool and chosen interventions; exploratory',
                   'family_size':size,'primary_endpoints':records,
                   'warning':'Response TV is not a fraction of positional information or a causal source decomposition.'}
        control = data['position_shift_boundary_endpoint']
        observed = data['endpoints'][:,0]
        rms_control = float(np.sqrt(np.mean(control**2)))
        rms_observed = float(np.sqrt(np.mean(observed**2)))
        ratio = rms_control/max(rms_observed,1e-12)
        summary['position_shift_numerical_control'] = {
            'shift':64,'rms_boundary_endpoint_response':rms_control,
            'rms_observed_boundary_endpoint':rms_observed,'ratio':ratio,
            'review_required':bool(ratio>.01),
            'note':'A sensitivity diagnostic, not an error bound. If above 1%, run a matched float64 audit before interpreting the boundary endpoint; main collection is retained.'}
        model_stats[model] = summary
        q = np.arange(4,512)
        fig,axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        plots = [('boundary_local_tv','Local response to added prefix (TV)'),
                 ('boundary_first_mass','Attention to actual first key'),
                 ('boundary_suffix_start_mass','Attention to fixed suffix-start key'),
                 ('replacement_local_tv','Local response to replacement (TV)'),
                 ('replacement_full_tv','Full attention response to replacement (TV)'),
                 ('replacement_first_mass_change','Change in first-key mass')]
        arrays = {}
        for index,(key,title) in enumerate(plots):
            mean,lo,hi = bootstrap_curves(data[key],cfg['bootstrap_replicates'],cfg['seed']+index)
            arrays[key+'_mean']=mean;arrays[key+'_lo']=lo;arrays[key+'_hi']=hi
            labels = ['0 preceding','16 preceding','64 preceding','256 preceding','64 shuffled'] if index<3 else ['Replace token 0','Replace token 1']
            ax = axes.flat[index]
            for j,label in enumerate(labels):
                start = 0 if index<3 else 1 # exclude query 4 in both replacement curves
                ax.plot(q[start:],mean[j,start:],label=label,lw=1.3)
                ax.fill_between(q[start:],lo[j,start:],hi[j,start:],alpha=.12)
            if index in [0,3,4]:
                control_key='position_shift_full_tv' if index==4 else 'position_shift_local_tv'
                control_curve=data[control_key].mean(axis=0)
                ax.plot(q,control_curve,color='black',ls=':',lw=1,label='Position-ID shift control')
            ax.set_xscale('log');ax.set_title(title,fontsize=10)
            ax.set_xlabel('Query coordinate in unchanged suffix (zero-based)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f'{model}: exploratory paired interventions, {len(ids)} documents\nPointwise 95% document-bootstrap bands; not simultaneous',fontsize=12)
        fig.savefig(directory/'curves.pdf');fig.savefig(directory/'curves.png',dpi=160);plt.close(fig)
        np.savez_compressed(directory/'curve-summary.npz',**arrays)
        with (directory/'document-endpoints.csv').open('w') as f:
            f.write('source_id,'+','.join(ENDPOINTS)+'\n')
            for doc,row in zip(ids,data['endpoints']):
                f.write(doc+','+','.join(map(str,row))+'\n')
    # Preserve the planned family even if one model has not finished.
    corrected = holm(combined_p+[1.]*(size-len(combined_p)))[:len(combined_p)]
    for entry,p in zip(entries,corrected):
        entry['p_holm_planned_family'] = float(p)
    for model,summary in model_stats.items():
        (stage_dir/model/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    final = {'stage':stage,'status':cfg['status'], 'models':model_stats,
             'complete':len(model_stats)==len(cfg['models']) and all(s['complete'] for s in model_stats.values()),
             'pilot_inference':'descriptive implementation check only' if stage=='pilot' else None}
    (stage_dir/'summary.json').write_text(json.dumps(final,indent=2,allow_nan=False))
    print(json.dumps(final,indent=2),flush=True)
    return final
