"""Small retrospective mammalian NEURON panel: frozen Lite versus fitted LIF.

This is not a fly benchmark. LIF fitting and Lite threshold selection use only
training traces; both freeze before the validation split is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
import torch
from axosim.checkpoint import load_checkpoint
from .neural import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT

PARAMETER_NAMES=('tau_m_ms','tau_e_ms','tau_i_ms','log10_gain_e','log10_gain_i',
                 'rest_mv','threshold_mv','reset_offset_mv','refractory_ms')
BOUNDS=((2.,80.),(1.,20.),(1.,30.),(-4.,.7),(-4.,.7),(-85.,-55.),(-65.,-40.),(5.,35.),(1.,5.))
WARMUP=100


@njit(cache=True)
def lif_predict(exc,inh,parameters):
    """Fixed 1-ms current-based LIF, no adaptive currents or dendrites."""
    tau_m,tau_e,tau_i,log_ge,log_gi,rest,threshold,reset_offset,refractory=parameters
    de=np.exp(-1./tau_e);di=np.exp(-1./tau_i);dm=np.exp(-1./tau_m)
    ge=10.**log_ge;gi=10.**log_gi
    reset=threshold-reset_offset
    ref_steps=int(np.floor(refractory+.5))
    voltage=np.empty_like(exc)
    spikes=np.zeros(exc.shape,dtype=np.bool_)
    for n in range(exc.shape[0]):
        v=-76.;e=0.;i=0.;remaining=0
        for t in range(exc.shape[1]):
            e=de*e+exc[n,t];i=di*i+inh[n,t]
            if remaining>0:
                v=reset;remaining-=1
            else:
                v=rest+(v-rest)*dm+ge*e-gi*i
                if v>=threshold:
                    spikes[n,t]=True
                    voltage[n,t]=-55.
                    v=reset;remaining=ref_steps
                    continue
            voltage[n,t]=min(v,-55.)
    return voltage,spikes


@njit(cache=True)
def event_counts(predicted,expected,tolerance=5,warmup=WARMUP):
    """Greedy chronological one-to-one matching; counts pooled over traces."""
    tp=0;npred=0;ntrue=0
    for n in range(predicted.shape[0]):
        for t in range(warmup,predicted.shape[1]):
            npred+=int(predicted[n,t]);ntrue+=int(expected[n,t])
        next_prediction=warmup
        for t in range(warmup,expected.shape[1]):
            if not expected[n,t]:continue
            lower=max(warmup,t-tolerance);upper=min(expected.shape[1]-1,t+tolerance)
            next_prediction=max(next_prediction,lower)
            while next_prediction<=upper and not predicted[n,next_prediction]:
                next_prediction+=1
            if next_prediction<=upper:
                tp+=1;next_prediction+=1
    return tp,npred,ntrue


def f1(counts):
    tp,npred,ntrue=counts
    return 2*tp/(npred+ntrue) if npred+ntrue else 1.


def load_split(root,split):
    files=sorted(Path(root).glob(f'specimen_479770916-{split}-*-neuron.npz'))
    if not files:raise FileNotFoundError(f'No {split} traces under {root}')
    records=[]
    for path in files:
        with np.load(path) as data:
            records.append({k:data[k].copy() for k in ('inputs','targets','expected_soma_mv')})
    inputs=np.stack([r['inputs'] for r in records])
    targets=np.stack([r['targets'] for r in records])
    return {'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
            'inputs':inputs,'exc':inputs.clip(min=0).sum(-1).astype(np.float64),
            'inh':(-inputs.clip(max=0)).sum(-1).astype(np.float64),
            'voltage':targets[...,1].astype(np.float64)/.1-67.7,
            'spikes':targets[...,0]>.5,'raw_voltage':np.stack([r['expected_soma_mv'] for r in records])}


def train_lif(data,*,joint,seed,maxiter=120):
    def objective(params):
        if params[5]>=params[6]:return 1e6+(params[5]-params[6])**2
        v,s=lif_predict(data['exc'],data['inh'],np.asarray(params))
        mse=np.square(v[:,WARMUP:]-data['voltage'][:,WARMUP:]).mean()
        return mse/100.+(1.-f1(event_counts(s,data['spikes'])) if joint else 0.)
    result=differential_evolution(objective,BOUNDS,seed=seed,maxiter=maxiter,popsize=10,polish=False,workers=1)
    return {'parameters':dict(zip(PARAMETER_NAMES,map(float,result.x))),'vector':result.x.tolist(),
            'objective':float(result.fun),'evaluations':result.nfev,'iterations':result.nit,
            'converged':bool(result.success),'message':str(result.message),'seed':seed,
            'joint_objective':joint,'training_only':True}


def lite_predict(model,data):
    with torch.inference_mode():
        pred=model(torch.from_numpy(data['inputs']),morphology_indices=torch.zeros(len(data['inputs']),dtype=torch.long)).cpu().numpy()
    return pred[...,1].astype(np.float64)/.1-67.7,pred[...,0]


def logit_events(logits,threshold):
    spikes=np.zeros(logits.shape,dtype=bool)
    for n,row in enumerate(logits):
        peaks,_=find_peaks(row,distance=2)
        peaks=peaks[row[peaks]>=threshold]
        spikes[n,peaks]=True
    return spikes


def select_threshold(logits,expected):
    candidates=np.unique(np.r_[0,np.quantile(logits[:,WARMUP:],np.linspace(0,1,129))])
    rows=[(f1(event_counts(logit_events(logits,float(t)),expected)),float(t)) for t in candidates]
    # Highest threshold wins exact ties, avoiding extra false events.
    score,threshold=max(rows)
    return {'threshold':threshold,'training_f1_5ms':score,'candidates':len(rows),'training_only':True}


def metrics(voltage,spikes,data):
    def row(v,s,target_v,target_s,raw):
        return {'clipped_target_rmse_mv':float(np.sqrt(np.square(v[:,WARMUP:]-target_v[:,WARMUP:]).mean())),
                'raw_target_rmse_mv':float(np.sqrt(np.square(v[:,WARMUP:]-raw[:,WARMUP:]).mean())),
                'f1_exact':f1(event_counts(s,target_s,0)), 'f1_5ms':f1(event_counts(s,target_s,5)),
                'event_counts_5ms':list(map(int,event_counts(s,target_s,5)))}
    all_rows=[row(voltage[n:n+1],spikes[n:n+1],data['voltage'][n:n+1],data['spikes'][n:n+1],data['raw_voltage'][n:n+1]) for n in range(len(voltage))]
    return {'pooled':row(voltage,spikes,data['voltage'],data['spikes'],data['raw_voltage']),'per_trace':all_rows}


def run_panel(root,output,checkpoint=DEFAULT_CHECKPOINT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest!=CHECKPOINT_SHA256:raise ValueError('checkpoint hash mismatch')
    torch.set_num_threads(1)
    train=load_split(root,'train')
    model,_=load_checkpoint(checkpoint);model.eval()
    lite_voltage,lite_logits=lite_predict(model,train)
    lif_voltage=train_lif(train,joint=False,seed=17)
    lif_joint=train_lif(train,joint=True,seed=18)
    threshold=select_threshold(lite_logits,train['spikes'])
    selection={'checkpoint_sha256':digest,'train_files':train['files'],'lif_voltage':lif_voltage,'lif_joint':lif_joint,'lite_threshold':threshold,
               'protocol':'fixed protocol.md, warmup100ms, 1ms inputs, train-only selection; no validation files loaded yet'}
    # Persist the immutable selected settings before reading validation traces.
    (output/'train-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    results={'scope':'small retrospective mammalian validation panel; not fly fidelity or independent blind holdout',
             'selection':selection,'splits':{}}
    for split,data in [('train',train),('val',load_split(root,'val'))]:
        v,l=lite_predict(model,data)
        per_model={'axosim_lite':metrics(v,logit_events(l,threshold['threshold']),data),
                   'axosim_lite_threshold_zero':metrics(v,logit_events(l,0),data)}
        arrays={'target_clipped_mv':data['voltage'],'target_raw_mv':data['raw_voltage'],'target_spikes':data['spikes'],
                'axosim_lite_mv':v,'axosim_lite_logits':l,'axosim_lite_spikes':logit_events(l,threshold['threshold'])}
        for name,fit in [('lif_voltage',lif_voltage),('lif_joint',lif_joint)]:
            pv,ps=lif_predict(data['exc'],data['inh'],np.array(fit['vector']))
            per_model[name]=metrics(pv,ps,data);arrays[f'{name}_mv']=pv;arrays[f'{name}_spikes']=ps
        results['splits'][split]={'files':data['files'],'models':per_model}
        np.savez_compressed(output/f'{split}-predictions.npz',**arrays)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v['pooled'] for k,v in results['splits']['val']['models'].items()},indent=2))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    args=parser.parse_args();run_panel(args.data,args.output,args.checkpoint)


if __name__=='__main__':main()
