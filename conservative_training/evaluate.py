"""Independent derivative tests and failure-aware source/incomplete rollouts."""
import argparse
from functools import partial
import json
from pathlib import Path
import pickle
import os
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
os.environ.setdefault('MPLCONFIGDIR', str(Path(__import__('tempfile').gettempdir())/'hail_mary_lnn_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .engine import network, energy
from .train import predictor, source_validation, neural_action, record_json
from . import physics as p


def evaluate(checkpoint_path,smoke=False,label=''):
    path=Path(checkpoint_path);out=path.parent/(('smoke_evaluation' if smoke else 'final_evaluation')+label)
    out.mkdir(exist_ok=False)
    with path.open('rb') as f:checkpoint=pickle.load(f)
    config=checkpoint['config'];model=config['model'];baseline=config['baseline']
    if model=='full_effective':raise RuntimeError('Physical full-model evaluation is gated; use benchmark audit diagnostics')
    from jax.example_libraries import optimizers
    _,_,get_params=optimizers.adam(.001)
    parameters=get_params(checkpoint['optimizer_state'])
    _,forward=network(3 if model=='source' or baseline else 2,seed=config['seed'],dtype=jnp.float32 if model=='source' else jnp.float64)
    predict=jax.jit(predictor(model,forward,parameters,baseline))
    # Smoke evaluations NEVER access the immutable final test seed.
    seed=1001 if smoke else 2000
    data=source_validation(seed,10000) if model=='source' else p.sample_incomplete(jax.random.PRNGKey(seed),10000,True)
    q,v,g,target=data
    guesses=[]
    for k in range(0,len(q),32):guesses.append(np.asarray(jax.vmap(predict)(q[k:k+32],v[k:k+32],g[k:k+32])))
    guesses=np.concatenate(guesses)
    beta=np.abs(np.asarray(v)[:,0]);error=np.abs(guesses-np.asarray(target))[:,0]
    bounds=[0.,.7,.9,.99,.99999]
    bands=[]
    for low,high in zip(bounds[:-1],bounds[1:]):
        mask=(beta>=low)&(beta<high)
        bands.append(dict(low=low,high=high,count=int(mask.sum()),MAE=float(np.mean(error[mask])),
                          unscaled_MAE_m_s2=None if model=='source' else p.A*float(np.mean(error[mask])),
                          passed=bool(np.all(np.isfinite(error[mask])) and np.mean(error[mask])<.01)))
    rng=np.random.default_rng(seed)
    cases=[]
    # Predeclared 40 independent initial states with fixed speed-band stratification.
    for i in range(40):
        low,high=bounds[i%4:i%4+2]
        speed=rng.uniform(low,high)*rng.choice([-1.,1.])
        pos=rng.uniform(-10,10) if model=='source' else rng.uniform(0,p.Q_ROUTE)
        force=float(rng.normal(0,10))
        cases.append((f'heldout_{i:02d}',pos,speed,force,.1,1.))
    if model=='incomplete':
        horizon=float(np.sqrt((p.Q_ROUTE/2)*(p.Q_ROUTE/2+2)))
        cases.append(('mission',0.,0.,1.,horizon/10,horizon))
    else:
        key=jax.random.PRNGKey(4)
        original_q=float(jax.random.uniform(key,(1,),dtype=jnp.float32,minval=-10,maxval=10)[0])
        original_v=float(jax.random.uniform(key+1,(1,),dtype=jnp.float32,minval=-.99,maxval=.99)[0])
        original_g=float(10*jax.random.normal(key+2,(1,),dtype=jnp.float32)[0])
        cases.append(('notebook_seed4',original_q,original_v,original_g,.1,1.))
    def run(name,q0,beta0,force,short,long):
        initial=np.array([q0,beta0,0.])
        def reference(t,z):
            b=z[1]
            acc=force*(1-b*b)**(2.5 if model=='source' else 1.5)
            if model=='source':acc/=1+2*b*b
            return [b,acc,force*b]
        calls=0
        def learned(t,z):
            nonlocal calls
            calls+=1
            if calls>10000:raise FloatingPointError(f'solver evaluation budget exceeded at t={t}')
            if not np.isfinite(z).all() or abs(z[1])>=1:
                # Do not clip or continue an invalid physical state.
                raise FloatingPointError(f'invalid state at t={t}')
            a=float(predict(jnp.array([z[0]],dtype=v.dtype),jnp.array([z[1]],dtype=v.dtype),jnp.array([force],dtype=v.dtype))[0])
            if not np.isfinite(a):raise FloatingPointError(f'nonfinite acceleration at t={t}')
            return [z[1],a,force*z[1]]
        result=dict(name=name,initial_position=q0,initial_beta=beta0,force=force,short_horizon=short,long_horizon=long)
        times=np.unique(np.concatenate([np.linspace(0,short,51),np.linspace(0,long,201)]))
        truth=solve_ivp(reference,(0,long),initial,t_eval=times,method='DOP853',rtol=1e-11,atol=1e-12)
        try:
            prediction=solve_ivp(learned,(0,long),initial,t_eval=times,method='DOP853',
                rtol=1e-6 if model=='source' else 1e-9,atol=1e-7 if model=='source' else 1e-11)
            if not prediction.success or not np.isfinite(prediction.y).all():raise FloatingPointError(prediction.message)
        except (FloatingPointError,ValueError) as failure:
            result.update(failed=True,reason=str(failure));return result,None
        diff=prediction.y[:2]-truth.y[:2]
        result.update(failed=False,beta_MAE=float(np.mean(np.abs(diff[1]))),beta_max_error=float(np.max(np.abs(diff[1]))),
                      position_MAE_scaled=float(np.mean(np.abs(diff[0]))),
                      short_beta_MAE=float(np.mean(np.abs(diff[1,times<=short]))),
                      maximum_speed_c=float(np.max(np.abs(prediction.y[1]))))
        if model=='incomplete':
            true_energy=1/np.sqrt(1-prediction.y[1]**2)-1
            physical_residual=true_energy-true_energy[0]-prediction.y[2]
            # Fixed scale, independent of almost-zero initial energy.
            scale=max(1.,abs(force)*long+1/np.sqrt(1-beta0**2))
            result['true_work_max_normalized_residual']=float(np.max(np.abs(physical_residual))/scale)
            if not baseline:
                action=neural_action(model,forward,parameters)
                energies=jax.vmap(partial(energy,action))(jnp.array(prediction.y[0,:,None]),jnp.array(prediction.y[1,:,None]))
                learned_residual=np.asarray(energies)-float(energies[0])-prediction.y[2]
                result['learned_work_max_normalized_residual']=float(np.max(np.abs(learned_residual))/scale)
            else:learned_residual=np.full_like(times,np.nan)
        else:physical_residual=learned_residual=np.full_like(times,np.nan)
        result['passed']=result['beta_MAE']<.001 and result['beta_max_error']<.005 and result['maximum_speed_c']<1
        if model=='incomplete':result['passed']&=result['true_work_max_normalized_residual']<.001
        return result,dict(times=times,truth=truth.y,prediction=prediction.y,true_work=physical_residual,learned_work=learned_residual)
    results=[];traces={}
    for case in cases:
        result,trace=run(*case);results.append(result)
        if trace is not None and (case[0] in ['mission','notebook_seed4'] or len(traces)<3):traces[case[0]]=trace
    runtime_path=path.parent/'runtime.json'
    completed_budget=runtime_path.exists() and json.loads(runtime_path.read_text(encoding='utf-8')).get('scientific_result',False)
    metrics=dict(checkpoint_step=checkpoint['step'],model=model,seed=config['seed'],
                 scientific_result=not smoke and completed_budget,split_seed=seed,
                 final_test_used=not smoke,acceleration_bands=bands,rollouts=results,
                 solver_method='DOP853',reference_rtol=1e-11,reference_atol=1e-12,
                 learned_rtol=1e-6 if model=='source' else 1e-9,learned_atol=1e-7 if model=='source' else 1e-11,
                 rollout_solver_evaluation_limit=10000,
                 failed_rollouts=sum(z['failed'] for z in results),
                 accepted=all(z['passed'] for z in bands) and all(z.get('passed',False) for z in results))
    record_json(out/'metrics.json',metrics)
    np.savez(out/'derivatives.npz',q=np.asarray(q),v=np.asarray(v),g=np.asarray(g),target=np.asarray(target),prediction=guesses)
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(4),[x['MAE'] for x in bands]);ax.axhline(.01,color='red',ls='--');
    ax.set(xticks=range(4),xticklabels=['0–.7','.7–.9','.9–.99','.99–.99999'],xlabel='Speed / c',ylabel='Acceleration MAE / fixed scale')
    fig.tight_layout();fig.savefig(out/'acceleration_by_speed.png',dpi=150);plt.close(fig)
    for name,trace in traces.items():
        fig,axes=plt.subplots(2 if model=='source' else 3,1,figsize=(8,6 if model=='source' else 9))
        axes[0].plot(trace['times'],trace['truth'][1],label='reference');axes[0].plot(trace['times'],trace['prediction'][1],ls='--',label='direct acceleration NN' if baseline else 'LNN');axes[0].set_ylabel('Speed / c');axes[0].legend()
        axes[1].plot(trace['truth'][0],trace['truth'][1],label='reference');axes[1].plot(trace['prediction'][0],trace['prediction'][1],ls='--');axes[1].set(xlabel='Scaled position',ylabel='Speed / c')
        if model!='source':
            axes[2].plot(trace['times'],trace['true_work'],label='true energy − work');axes[2].plot(trace['times'],trace['learned_work'],label='learned energy − work');axes[2].set(xlabel='Scaled time',ylabel='Work residual');axes[2].legend()
        fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=150);plt.close(fig)
    history=checkpoint['history']
    if history:
        fig,ax=plt.subplots(figsize=(8,4));ax.semilogy([x['step'] for x in history],[x['training_loss'] for x in history],marker='o',markevery=max(1,len(history)//20),label='training')
        ax.semilogy([x['step'] for x in history],[x['validation_score'] for x in history],marker='o',markevery=max(1,len(history)//20),label='validation');ax.legend();ax.set(xlabel='Update',ylabel='Weighted absolute acceleration error')
        fig.tight_layout();fig.savefig(out/'loss.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k!='rollouts'}),flush=True)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--smoke',action='store_true');parser.add_argument('--label',default='')
    args=parser.parse_args();evaluate(args.checkpoint,args.smoke,args.label)
