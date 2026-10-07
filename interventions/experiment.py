"""A document is the paired analysis unit; donors and heads stay together."""
import numpy as np
from design import boundary_conditions, donor_tokens, tv, primary_head_endpoints

def document(engine,tokens,cfg,doc_id,train,specials,audit=False):
    conditions = boundary_conditions(tokens,cfg,doc_id)
    baseline_tokens = conditions[0][1]
    baseline = engine.collect(baseline_tokens,retain_full=True,audit=audit)
    boundary_tv, boundary_rms, first, suffix = [],[],[],[]
    for name,sequence,m in conditions:
        result = baseline if m == 0 else engine.collect(sequence,suffix_start=m,audit=audit)
        boundary_tv.append(tv(result['u'],baseline['u']))
        boundary_rms.append(np.sqrt(np.mean((result['z'].astype('float64')-baseline['z'])**2,axis=-1)))
        first.append(result['first'])
        suffix.append(result['suffix'])
        if result is not baseline:
            del result
    donors = donor_tokens(train,baseline_tokens,specials,cfg['replacement_donors'],cfg['seed'],doc_id)
    replace_tv, replace_full, replace_first, replace_rms = [],[],[],[]
    for donor in donors:
        local,full,mass,rms = [],[],[],[]
        for pos in cfg['replacement_positions']:
            replaced = baseline_tokens.copy()
            replaced[pos] = int(donor)
            assert np.sum(replaced != baseline_tokens) == 1
            result = engine.collect(replaced,reference=baseline['full'],audit=audit)
            local.append(tv(result['u'],baseline['u']))
            full.append(result['full_tv'])
            mass.append(result['first']-baseline['first'])
            rms.append(np.sqrt(np.mean((result['z'].astype('float64')-baseline['z'])**2,axis=-1)))
            del result
        replace_tv.append(local);replace_full.append(full);replace_first.append(mass);replace_rms.append(rms)
    boundary_tv = np.asarray(boundary_tv)
    replace_tv = np.asarray(replace_tv)  # donor, position, layer, head, query
    mean_replacement = replace_tv.mean(axis=0)
    heads = primary_head_endpoints(boundary_tv,mean_replacement)
    # A common position-ID shift changes no relative distances or accessible keys.
    # Under fixed-frequency RoPE its response is a numerical negative control,
    # retained even when the old strict entrywise shift tolerance would fail.
    shifted = engine.collect(baseline_tokens,reference=baseline['full'],position_shift=64,audit=audit)
    shift_tv = tv(shifted['u'],baseline['u'])
    shift_endpoint = primary_head_endpoints(np.stack([shift_tv]*5),np.zeros_like(mean_replacement))[0]
    output = {
        'document_id':np.asarray(str(doc_id)), 'donor_tokens':donors,
        'conditions':np.asarray([x[0] for x in conditions]),
        'query_coordinates':np.arange(4,512),
        'boundary_local_tv':boundary_tv.mean(axis=2),
        'boundary_local_logit_rms':np.asarray(boundary_rms).mean(axis=2),
        'boundary_first_mass':np.asarray(first).mean(axis=2),
        'boundary_suffix_start_mass':np.asarray(suffix).mean(axis=2),
        'replacement_local_tv':mean_replacement.mean(axis=2),
        'replacement_full_tv':np.asarray(replace_full).mean(axis=(0,3)),
        'replacement_first_mass_change':np.asarray(replace_first).mean(axis=(0,3)),
        'replacement_local_logit_rms':np.asarray(replace_rms).mean(axis=(0,3)),
        'replacement_donor_local_tv':replace_tv.mean(axis=3),
        'head_endpoints':heads,
        'endpoints':heads.mean(axis=(1,2)),
        'position_shift_local_tv':shift_tv.mean(axis=1),
        'position_shift_full_tv':shifted['full_tv'].mean(axis=1),
        'position_shift_boundary_endpoint':np.asarray(shift_endpoint.mean()),
    }
    for key,value in output.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise RuntimeError(f'Non-finite saved metric: {key}')
    return output
