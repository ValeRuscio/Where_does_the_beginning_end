"""Validation-developed revision: separate local offsets from key zero.

Reuses the old training reference ONLY on balanced, boundary-free distances.
New GPU fitting uses training prefixes only; validation/test never fit a bias.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .core import atomic_json, atomic_npz, digest, softmax, bootstrap_curves, paired_summary, holm
from .model import InstrumentedModel

VERSION = "revised-1"
LOCAL = ["local_logit_gain", "local_probability_gain", "local_logit_baseline_mse",
         "local_probability_baseline_mse", "first_key_mass", "self_mass"]
PRED = ["relative_tv", "first_bias_tv", "self_bias_tv", "first_bias_kl_gain",
        "self_bias_kl_gain", "first_over_self_kl_gain", "observed_first_mass",
        "relative_first_mass", "first_bias_first_mass", "self_bias_first_mass"]


def context(path):
    cfg = json.loads(Path(path).read_text())
    source, out = Path(cfg["source_run"]), Path(cfg["output"])
    if source.resolve() == out.resolve():
        raise ValueError("Revised outputs must not overwrite the original run")
    manifest = json.loads((source / "manifest.json").read_text())
    original = manifest["config"]
    if digest(original) != manifest["config_digest"]:
        raise ValueError("Source manifest/config mismatch")
    maximum, D, L = cfg["max_offset"], cfg["local_offsets"], original["length"]
    if not D <= maximum < original["fit_query_min"] < L:
        raise ValueError("Require local_offsets <= max_offset < original fit_query_min < length")
    if D != original["common_offsets"]:
        raise ValueError("Reused local templates require unchanged common_offsets")
    if not D <= cfg["early"][0] < cfg["early"][1] <= cfg["late"][0] < cfg["late"][1] <= L:
        raise ValueError("Early/late queries must exclude key zero from all local offsets")
    for name in ["bias_queries", "comparison_queries"]:
        a, b = cfg[name]
        if not D <= a < b <= maximum + 1:
            raise ValueError(f"Invalid {name}; no predictions at unobserved distances")
    with np.load(source / "reference.npz") as z:
        if str(z["manifest_digest"]) != digest(manifest):
            raise ValueError("Source reference/manifest mismatch")
        reference = {k: z[k].copy() for k in ["profile", "common_logits", "common_probabilities"]}
    fingerprint = digest({k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in reference.items()})
    identity = digest([VERSION, cfg, manifest, fingerprint])
    if (out / "revision.json").exists():
        prior = json.loads((out / "revision.json").read_text())
        if prior["identity"] != identity:
            raise ValueError("Changed revision settings/source reference: choose a new output directory")
    return cfg, source, out, manifest, reference, identity


def begin(path):
    cfg, source, out, manifest, reference, identity = context(path)
    atomic_json(out / "revision.json", {"version": VERSION, "config": cfg, "identity": identity,
        "source_manifest_digest": digest(manifest), "model_revision": manifest["model_revision"],
        "development": "Analysis revised after viewing original validation; conditional on old training reference"})
    return cfg, source, out, manifest, reference, identity


def log_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def log_odds(profile, queries, kind):
    """Unboosted log odds for first key (d=q) or current key (d=0)."""
    values = []
    for q in queries:
        row = profile[..., :q+1]
        selected = q if kind == "first" else 0
        others = np.delete(row, selected, axis=-1)
        values.append(row[..., selected] - np.logaddexp.reduce(others, axis=-1))
    return np.stack(values, axis=-1)


def fit_bias(profile, target_mass, queries, kind):
    """One bounded scalar/head minimizing training cross entropy, f held fixed.

    dCE/dbeta = mean predicted selected-key mass - mean observed selected-key mass.
    A mean mass is therefore sufficient; no validation attention is involved.
    """
    odds = log_odds(profile, queries, kind)
    lo, hi = np.full_like(target_mass, -40.0), np.full_like(target_mass, 40.0)
    for _ in range(70):
        middle = (lo + hi) / 2
        predicted = np.exp(-np.logaddexp(0, -(odds + middle[..., None]))).mean(axis=-1)
        lo = np.where(predicted < target_mass, middle, lo)
        hi = np.where(predicted >= target_mass, middle, hi)
    return (lo + hi) / 2


def fit(path):
    cfg, source, out, manifest, old, identity = begin(path)
    checkpoint = out / "bias-checkpoint.npz"
    completed, sums = 0, None
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            if str(z["identity"]) != identity:
                raise ValueError("Stale checkpoint")
            completed, sums = int(z["completed"]), z["sums"].copy()
    tokens = np.load(source / "tokens-train.npz")["tokens"]
    a,b = cfg["bias_queries"]
    if completed < len(tokens):
        model = InstrumentedModel(manifest["config"], manifest)
        atomic_json(out / "fit-environment.json", model.metadata)
        for index in range(completed, len(tokens)):
            layers = []
            def collect(layer, module, query, key, weights, scaling, mode):
                if index == completed:
                    model.check_weights(query,key,module,weights,scaling,mode)
                p = weights[0].float().cpu().numpy().astype(np.float64)
                p /= p.sum(axis=-1, keepdims=True)
                layers.append(np.stack([p[:,a:b,0].mean(-1),
                                        np.diagonal(p,axis1=-2,axis2=-1)[:,a:b].mean(-1)]))
            start=time.perf_counter()
            model.forward(tokens[index,:b], collect)
            value=np.stack(layers)
            sums = value if sums is None else sums+value
            atomic_npz(checkpoint, completed=np.array(index+1), sums=sums, identity=np.array(identity))
            print(f"Bias fit {index+1}/{len(tokens)} ({b} tokens): {time.perf_counter()-start:.1f}s",flush=True)
    if sums is None or len(tokens) != len(manifest["splits"]["train"]):
        raise ValueError("Invalid training split")
    profile=old["profile"][...,:cfg["max_offset"]+1]
    target=sums/len(tokens)
    beta_first=fit_bias(profile,target[:,0],range(a,b),"first")
    beta_self=fit_bias(profile,target[:,1],range(a,b),"self")
    stationary_probability=old["common_probabilities"][...,manifest["config"]["fit_query_min"]:,:].mean(axis=-2)
    atomic_npz(out/"reference.npz", profile=profile, local_logits=old["common_logits"],
               local_probabilities=old["common_probabilities"], stationary_probability=stationary_probability,
               beta_first=beta_first,beta_self=beta_self,identity=np.array(identity))
    atomic_json(out/"bias-fit.json", {"training_documents":len(tokens),"queries":[a,b-1],
        "beta_bound":40,"first_bias_heads_at_bound":int((np.abs(beta_first)>39.99).sum()),
        "self_bias_heads_at_bound":int((np.abs(beta_self)>39.99).sum()),
        "note":"One scalar per head in each alternative, trained on identical documents/query positions."})


def metrics(logits, actual, reference, layer, cfg):
    H,L,_=logits.shape
    D=cfg["local_offsets"]
    f=reference["profile"][layer]
    local=logits[...,:D].copy()
    local[:,:D]=0  # Invalid: d=0..D-1 contains key zero when q=D-1.
    local-=local.mean(-1,keepdims=True)
    p=softmax(local)
    baseline=f[:,:D]-f[:,:D].mean(-1,keepdims=True)
    lmse=((local-baseline[:,None])**2).mean(-1)
    pmse=((p-reference["stationary_probability"][layer,:,None,:])**2).mean(-1)
    production=actual.astype(np.float64)
    production/=production.sum(-1,keepdims=True)
    local_metrics=np.stack([
        lmse-((local-reference["local_logits"][layer])**2).mean(-1),
        pmse-((p-reference["local_probabilities"][layer])**2).mean(-1),
        lmse,pmse,production[...,0],np.diagonal(production,axis1=-2,axis2=-1)])
    local_metrics[:4,:,:D]=0
    maximum=cfg["max_offset"]
    pred=np.zeros((len(PRED),H,maximum+1))
    # Query zero's full attention is deterministically one; prediction gains are zero.
    pred[6:,:,0]=1
    for q in range(1,maximum+1):
        obs=softmax(logits[:,q,:q+1])
        raw=f[:,:q+1]
        first=raw.copy(); first[:,q]+=reference["beta_first"][layer]
        self_scores=raw.copy(); self_scores[:,0]+=reference["beta_self"][layer]
        log0,log1,log2=map(log_softmax,[raw,first,self_scores])
        p0,p1,p2=map(np.exp,[log0,log1,log2])
        g1=(obs*(log1-log0)).sum(-1);g2=(obs*(log2-log0)).sum(-1)
        pred[:,:,q]=np.stack([np.abs(obs-p0).sum(-1)/2,np.abs(obs-p1).sum(-1)/2,
            np.abs(obs-p2).sum(-1)/2,g1,g2,g1-g2,obs[:,q],p0[:,q],p1[:,q],p2[:,q]])
    if not np.isfinite(local_metrics).all() or not np.isfinite(pred).all():
        raise ValueError("Non-finite revised metrics")
    return local_metrics,pred


def evaluate(path,split="validation",allow_test=False):
    if split=="test" and not allow_test:
        raise ValueError("Test locked. Freeze the revised protocol first, then explicitly pass --allow-test")
    cfg,source,out,manifest,old,identity=context(path)
    with np.load(out/"reference.npz") as z:
        if str(z["identity"])!=identity:raise ValueError("Stale reference")
        ref={k:z[k].copy() for k in z.files if k!="identity"}
    ref_hash=digest({k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ref.items()})
    run_hash=digest([identity,ref_hash,split])
    tokens=np.load(source/f"tokens-{split}.npz")["tokens"]
    destination=out/"evaluations"/split
    todo=[]
    for i in range(len(tokens)):
        file=destination/f"{i:05d}.npz"
        if file.exists():
            with np.load(file) as z:
                if str(z["run_hash"])!=run_hash:raise ValueError("Stale evaluation")
        else:todo.append(i)
    if not todo:
        print("Revised split already evaluated",flush=True);return
    model=InstrumentedModel(manifest["config"],manifest)
    atomic_json(destination/"environment.json",model.metadata)
    for i in todo:
        layers=[];predictions=[];heads=[]
        def collect(layer,module,query,key,weights,scaling,mode):
            if i==todo[0]:model.check_weights(query,key,module,weights,scaling,mode)
            logits=model.distance_logits(module,query,key,scaling)
            loc,pre=metrics(logits,weights[0].float().cpu().numpy(),ref,layer,cfg)
            layers.append(loc.mean(1));predictions.append(pre.mean(1))
            e0,e1=cfg["early"];l0,l1=cfg["late"];a,b=cfg["comparison_queries"]
            heads.append(np.stack([loc[0,:,e0:e1].mean(-1)-loc[0,:,l0:l1].mean(-1),
                loc[1,:,e0:e1].mean(-1)-loc[1,:,l0:l1].mean(-1),
                pre[3,:,a:b].mean(-1),pre[5,:,a:b].mean(-1),loc[4,:,len(tokens[i])//2:].mean(-1)]))
        start=time.perf_counter();model.forward(tokens[i],collect)
        atomic_npz(destination/f"{i:05d}.npz",local=np.stack(layers).mean(0).astype(np.float32),
            predictions=np.stack(predictions).mean(0).astype(np.float32),head_endpoints=np.stack(heads).astype(np.float32),
            document_id=np.array(manifest["splits"][split][i]["id"]),run_hash=np.array(run_hash))
        print(f"Revised {split} {i+1}/{len(tokens)}: {time.perf_counter()-start:.1f}s",flush=True)


def load_revised(path,split):
    cfg,source,out,manifest,old,identity=context(path)
    local=[];pred=[];heads=[];hashes=set()
    for i,record in enumerate(manifest["splits"][split]):
        file=out/"evaluations"/split/f"{i:05d}.npz"
        if not file.exists():raise FileNotFoundError(f"Incomplete split: {file}")
        with np.load(file) as z:
            if str(z["document_id"])!=record["id"]:raise ValueError("Document mismatch")
            local.append(z["local"]);pred.append(z["predictions"]);heads.append(z["head_endpoints"])
            hashes.add(str(z["run_hash"]))
    if len(hashes)!=1:raise ValueError("Mixed references")
    return cfg,out,np.stack(local).astype(float),np.stack(pred).astype(float),np.stack(heads).astype(float)


def analyze(path,split="validation",allow_test=False):
    if split=="test" and not allow_test:raise ValueError("Test analysis locked; use --allow-test only after protocol freeze")
    cfg,out,local,pred,heads=load_revised(path,split)
    dst=out/"analysis"/split;dst.mkdir(parents=True,exist_ok=True)
    e0,e1=cfg["early"];l0,l1=cfg["late"];a,b=cfg["comparison_queries"]
    values=[local[:,j,e0:e1].mean(-1)-local[:,j,l0:l1].mean(-1) for j in [0,1]]
    values += [pred[:,j,a:b].mean(-1) for j in [3,5]]
    labels=["local_logit_early_minus_late_gain","local_probability_early_minus_late_gain",
            "first_bias_over_relative_kl_gain","first_bias_over_self_bias_kl_gain"]
    endpoints=[]
    for label,value in zip(labels,values):
        s=paired_summary(value,cfg["alpha"])
        s.update(endpoint=label,familywise_ci=paired_summary(value,cfg["alpha"]/4)["ci"])
        endpoints.append(s)
    for s,p in zip(endpoints,holm([s["p_two_sided"] for s in endpoints])):
        s["p_holm_four_endpoints"]=float(p)
    report={"status":"validation-developed analysis" if split=="validation" else "held-out revised test; confirmatory only if frozen before test access",
        "n_documents":len(local),"unit":"source document","endpoints":endpoints,
        "local_offsets":list(range(cfg["local_offsets"])),"early_queries":[e0,e1-1],"late_queries":[l0,l1-1],
        "prediction_queries":[a,b-1],"uncertainty":"Conditional on training reference and fitted biases; four-test Holm correction",
        "local_gains":{name:{"early":paired_summary(local[:,j,e0:e1].mean(-1)),"late":paired_summary(local[:,j,l0:l1].mean(-1))} for j,name in enumerate(LOCAL[:2])},
        "model_fit":{name:float(pred[:,j,a:b].mean()) for j,name in enumerate(PRED)},
        "late_first_key_mass":paired_summary(local[:,4,local.shape[-1]//2:].mean(-1)),
        "limits":"Positive model gain is predictive evidence, not proof of a causal mechanism; no full-distribution prediction beyond max_offset."}
    atomic_json(dst/"summary.json",report)
    for name,data,names in [("local",local,LOCAL),("predictions",pred,PRED)]:
        mean,lo,hi,slo,shi=bootstrap_curves(data,cfg["bootstrap"],cfg["seed"],cfg["alpha"]/2)
        atomic_npz(dst/f"{name}-curves.npz",metrics=np.array(names),mean=mean,low=lo,high=hi,simultaneous_low=slo,simultaneous_high=shi)
        with (dst/f"{name}-curves.csv").open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["metric","query","mean","low","high","simultaneous_low","simultaneous_high","valid"])
            for j,label in enumerate(names):
                for q in range(mean.shape[-1]):
                    valid=not (name=="local" and j<4 and q<cfg["local_offsets"])
                    w.writerow([label,q,mean[j,q],lo[j,q],hi[j,q],slo[j,q],shi[j,q],valid])
    # Descriptive layer/head summaries preserve documents as the replication unit.
    atomic_npz(dst/"head-endpoints.npz",mean=heads.mean(0),se=heads.std(0,ddof=1)/np.sqrt(len(heads)),
               names=np.array(labels+["late_first_key_mass"]))
    plot(dst,cfg)
    print(json.dumps(report,indent=2))


def plot(dst,cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a=np.load(dst/"local-curves.npz");p=np.load(dst/"predictions-curves.npz")
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,j,title in [(axes[0,0],0,"Local logit prediction gain"),(axes[0,1],1,"Local probability prediction gain"),(axes[0,2],4,"Observed first-key attention mass")]:
        start=cfg["local_offsets"] if j<4 else 0
        q=np.arange(start,a["mean"].shape[-1]);ax.plot(q,a["mean"][j,start:]);ax.fill_between(q,a["low"][j,start:],a["high"][j,start:],alpha=.2)
        ax.set_xscale("symlog",linthresh=16);ax.set_xlim(start,len(a["mean"][j])-1);ax.set_title(title)
    start,end=cfg["comparison_queries"];q=np.arange(start,end)
    for j,label in [(0,"Relative only"),(1,"+ first-key bias"),(2,"+ current-key bias")]:
        axes[1,0].plot(q,p["mean"][j,start:end],label=label)
    axes[1,0].set_title("Full-prefix TV error (lower is better)");axes[1,0].legend()
    for j,label in [(6,"Observed"),(7,"Relative only"),(8,"+ first-key bias")]:
        axes[1,1].plot(q,p["mean"][j,start:end],label=label)
    axes[1,1].set_title("First-key mass: observation vs prediction");axes[1,1].legend()
    for j,label in [(3,"First vs relative"),(5,"First vs current-key control")]:
        axes[1,2].plot(q,p["mean"][j,start:end],label=label)
        axes[1,2].fill_between(q,p["low"][j,start:end],p["high"][j,start:end],alpha=.15)
    axes[1,2].set_title("KL improvement (nats; higher is better)");axes[1,2].legend()
    for ax in axes.flat:ax.set_xlabel("Query position (zero-based)");ax.axhline(0,color="grey",lw=.5)
    fig.suptitle("Revised measurements — training-fitted references, held-out documents")
    fig.savefig(dst/"revised-curves.png",dpi=160);fig.savefig(dst/"revised-curves.pdf");plt.close(fig)
    h=np.load(dst/"head-endpoints.npz")["mean"]
    fig,axes=plt.subplots(1,3,figsize=(13,7),constrained_layout=True)
    for ax,j,label in zip(axes,[0,2,4],["Local logit early–late gain","First-bias KL improvement","Late first-key mass"]):
        view=h[:,j,:]
        im=ax.imshow(view,aspect="auto",origin="lower",cmap="viridis")
        ax.set_title(label);ax.set_xlabel("Head");ax.set_ylabel("Layer")
        fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle("Descriptive head means; heads are not independent observations")
    fig.savefig(dst/"head-overview.png",dpi=160);fig.savefig(dst/"head-overview.pdf");plt.close(fig)


def reanalyze(path):
    cfg,source,out,manifest,old,identity=begin(path)
    xs=[];hs=[];hashes=set()
    for i,record in enumerate(manifest["splits"]["validation"]):
        with np.load(source/"evaluations/validation/causal"/f"{i:05d}.npz") as z:
            if str(z["document_id"])!=record["id"]:raise ValueError("Document mismatch")
            xs.append(z["curve"]);hs.append(z["head_bands"]);hashes.add(str(z["run_hash"]))
    if len(hashes)!=1:raise ValueError("Mixed original runs")
    x=np.stack(xs).astype(float);heads=np.stack(hs).astype(float)
    report={"status":"exploratory reanalysis of already-viewed validation; no GPU used","n_documents":len(x),"results":{}}
    for j,name in enumerate(["logits","conditional_probabilities"]):
        early=x[:,j,4:8].mean(-1);late=x[:,j,32:64].mean(-1)
        report["results"][name]={"query3":paired_summary(x[:,j,3]),"queries4_to7":paired_summary(early),
            "queries32_to63":paired_summary(late),"contrast_excluding_query3":paired_summary(early-late),
            "query3_share_of_old_early_gain":float(x[:,j,3].mean()/x[:,j,3:8].mean(0).sum())}
    report["first_mass_by_band"]={f"{a}-{b-1}":paired_summary(x[:,5,a:b].mean(-1)) for a,b in [(4,8),(32,64),(512,x.shape[-1])] if b<=x.shape[-1] and a<b}
    report["head_warning"]="Original head summaries include query 3. Excluding it per head requires revised collection; do not relabel these old head summaries."
    dst=out/"original-validation-reanalysis"
    atomic_json(dst/"summary.json",report)
    atomic_npz(dst/"old-head-bands.npz",mean=heads.mean(0),se=heads.std(0,ddof=1)/np.sqrt(len(heads)))
    with (dst/"old-head-bands.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["layer","head","metric","original_early_3_to7","original_late_32_to63"])
        mean=heads.mean(0)
        for layer in range(mean.shape[0]):
            for head in range(mean.shape[2]):
                for j,label in [(0,"logit_gain"),(1,"conditional_probability_gain"),(5,"first_key_mass")]:
                    w.writerow([layer,head,label,*mean[layer,j,head]])
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["reanalyze","fit","evaluate","analyze"])
    p.add_argument("--config",default="configs/revised.json")
    p.add_argument("--split",choices=["validation","test"],default="validation")
    p.add_argument("--allow-test",action="store_true")
    a=p.parse_args()
    if a.command=="reanalyze":reanalyze(a.config)
    elif a.command=="fit":fit(a.config)
    elif a.command=="evaluate":evaluate(a.config,a.split,a.allow_test)
    else:analyze(a.config,a.split,a.allow_test)


if __name__=="__main__":main()
