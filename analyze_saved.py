"""Recompute reported endpoint means, intervals and Holm tests without a GPU."""
from pathlib import Path
import json
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent

def summarize(x,family):
    n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1));se=sd/np.sqrt(n)
    half=stats.t.ppf(1-.05/(2*family),n-1)*se
    p=float(2*stats.t.sf(abs(mean/se),n-1)) if se else (1. if mean==0 else 0.)
    return {'n_documents':n,'mean':mean,'familywise_ci':[mean-half,mean+half],'p_two_sided':p}

def holm(p):
    p=np.array(p);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));return out

def check(records,expected):
    for a,b in zip(records,expected):
        np.testing.assert_allclose(a['mean'],b['mean'],rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(a['familywise_ci'],b['familywise_ci'],rtol=1e-9,atol=1e-11)

def main():
    results={};snapshot=json.loads((ROOT/'data/input_snapshot/manifest.json').read_text())
    for model in ['qwen3','qwen25']:
        for split in ['validation','test']:
            base=ROOT/'saved_results/predictive'/model
            with np.load(base/f'{split}-documents.npz',allow_pickle=False) as z:
                assert z['document_id'].tolist()==[str(x['id']) for x in snapshot['splits'][split]]
                local=z['local'].astype(float);pred=z['predictions'].astype(float)
                xs=[local[:,j,4:8].mean(-1)-local[:,j,32:64].mean(-1) for j in [0,1]]
                xs += [pred[:,j,4:256].mean(-1) for j in [3,5]]
            expected=json.loads((base/f'{split}-summary.json').read_text())['endpoints']
            records=[{'endpoint':e['endpoint'],**summarize(x,4)} for x,e in zip(xs,expected)]
            check(records,expected)
            for row,p in zip(records,holm([r['p_two_sided'] for r in records])):row['p_holm']=float(p)
            results[f'predictive/{model}/{split}']=records
    intervention_records=[];donors=[]
    for model in ['qwen3','qwen25']:
        base=ROOT/'saved_results/interventions'/model
        with np.load(base/'documents.npz',allow_pickle=False) as z:
            assert z['document_id'].tolist()==[str(x['id']) for x in snapshot['splits']['test'][:128]]
            donors.append(z['donor_tokens'])
            b=z['boundary_local_tv'];r=z['replacement_local_tv']
            # Layer/head averaging was performed before packaging, preserving documents.
            xs=[b[:,2,:4].mean(-1)-b[:,2,28:60].mean(-1),
                (r[:,0,1:60]-r[:,1,1:60]).mean(-1)]
            np.testing.assert_allclose(np.stack(xs,axis=1),z['endpoints'],rtol=1e-9,atol=1e-11)
        expected=json.loads((base/'summary.json').read_text())['primary_endpoints']
        records=[{'endpoint':e['endpoint'],**summarize(x,4)} for x,e in zip(xs,expected)]
        check(records,expected);intervention_records.extend(records)
        results[f'interventions/{model}']=records
    np.testing.assert_array_equal(*donors)
    for row,p in zip(intervention_records,holm([r['p_two_sided'] for r in intervention_records])):row['p_holm']=float(p)
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    (out/'recomputed-endpoints.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print('All 20 endpoint estimates and familywise intervals match saved reports.')
    print('Document identities and cross-model replacement donors verified.')
    print('Output: generated/recomputed-endpoints.json')
if __name__=='__main__':main()
