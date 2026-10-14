"""One resumable driver for source control and conservative PHM networks.

Scientific defaults are fixed at 4x500, batch 512, and 300000 updates.
Smoke runs keep that schedule and are explicitly non-scientific.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
from .platform_utils import max_rss_bytes
import shutil
import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import optimizers
from .engine import network, acceleration, diagnostics, learning_rate
from . import physics as p

BUDGET=300000


def source_epoch(epoch, batch=512):
    """Exact source sampler array shapes and key expressions; fixed velocity key."""
    count=batch*3000
    key=jax.random.PRNGKey(epoch)
    q=jax.random.uniform(key,(count,),dtype=jnp.float32,minval=-10,maxval=10)
    v=.99999*jnp.tanh(2*jax.random.normal(jax.random.PRNGKey(1),(count,),dtype=jnp.float32))
    g=10*jax.random.normal(key+2,(count,),dtype=jnp.float32)
    target=g*(1-v**2)**2.5/(1+2*v**2)
    return tuple(x.reshape(3000,batch,1) for x in (q,v,g,target))


def source_validation(seed, count):
    kq,kv,kg=jax.random.split(jax.random.PRNGKey(seed),3)
    q=jax.random.uniform(kq,(count,1),dtype=jnp.float32,minval=-10,maxval=10)
    v=.99999*jnp.tanh(2*jax.random.normal(kv,(count,1),dtype=jnp.float32))
    g=10*jax.random.normal(kg,(count,1),dtype=jnp.float32)
    return q,v,g,g*(1-v**2)**2.5/(1+2*v**2)


def neural_action(model,forward,parameters,conditionals=None):
    def action(q,v):
        if model=='full_effective':
            q=q.at[:3].set(q[:3]/(p.DISTANCE/p.JOINT_SCALES[:3]))
        features=jnp.concatenate([q,v]) if conditionals is None else jnp.concatenate([q,v,conditionals/10])
        return jnp.squeeze(forward(parameters,features))
    return action


def predictor(model, forward, parameters, baseline=False):
    if baseline:
        return lambda q,v,g:forward(parameters,jnp.concatenate([q,v,g]))
    if model=='source':
        def predict(q,v,g):
            action=neural_action(model,forward,parameters,g)
            # Intentionally preserve the source's derivative-level position rescaling.
            return acceleration(action,q/10,v,jnp.zeros_like(q))
        return predict
    # Differentiate through fixed position features; all chain factors retained.
    action=neural_action(model,forward,parameters)
    return partial(acceleration,action)


def weights(model,v):
    beta=jnp.abs(v[:,0]) if v.shape[1]==1 else jnp.linalg.norm(v[:,:3],axis=1)
    return 1+1/jnp.sqrt(1-beta**2)


def record_json(path,value):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x):return str(x)
        return x
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False))


def manifest(config):
    folder=Path(__file__).parent
    files=list(folder.glob('*.py'))+list((folder/'provenance').glob('*'))
    hashes={str(f.relative_to(folder)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
    return dict(config=config,platform=platform.platform(),python=platform.python_version(),
                packages={x:importlib.metadata.version(x) for x in ['jax','jaxlib','numpy','scipy','matplotlib']},
                x64_enabled=bool(jax.config.jax_enable_x64),
                devices=[str(x) for x in jax.devices()],source_sha='3b994720c87ebcb4b32a9bae5b09994567a79c2a',
                source_hashes=hashes,jax_adam_source=__import__('inspect').getsource(optimizers.adam),
                deviations=['Modern jax.example_libraries imports replace removed jax.experimental imports',
                            'Modern JAX PRNG and autodiff implementation; not a historical environment reproduction',
                            'Auxiliary validation and checkpointing added; no checkpoint loaded from source notebook',
                            'Loss/gradient evaluated in chunks; same weighted batch objective and one Adam update',
                            'Portable export: standard temp paths and platform-specific memory reporting; stored coordinate scales retained on resume'],
                physical_full_claim=False)


def train(args):
    mathematical = getattr(args, 'mathematical_benchmark', False)
    if args.model == 'full_effective' and args.steps > 1000:
        if not mathematical:
            raise RuntimeError('Physical full training remains gated; explicitly select the mathematical benchmark')
        audit_path = getattr(args, 'physics_audit', None)
        if not audit_path:
            raise ValueError('The mathematical benchmark requires the completed numerical physics audit')
        audit = json.loads(Path(audit_path).read_text(encoding='utf-8'))
        if not audit['full_training_gates']['default_pinv_retains_nine_modes']:
            raise RuntimeError('The nine-mode numerical audit failed')
    # The source control must keep fractional Adam progress in historical float32.
    # With x64 globally enabled, that progress silently promotes all Adam leaves.
    jax.config.update('jax_enable_x64',args.model!='source')
    out=Path(args.out)
    if args.resume:
        with Path(args.resume).open('rb') as f: checkpoint=pickle.load(f)
        config=checkpoint['config']
        expected_dtype=np.dtype(config['dtype'])
        if any(np.asarray(x).dtype!=expected_dtype for x in jax.tree.leaves(checkpoint['optimizer_state'])):
            raise ValueError('Checkpoint actual precision differs from its declared configuration')
        if args.model=='full_effective' and not np.allclose(np.asarray(config.get('coordinate_scales_SI',[])),np.asarray(p.JOINT_SCALES),rtol=1e-14,atol=0.):
            raise ValueError('Coupled-action coordinate scales changed; use a new experiment')
        # Keep the stored transform exactly; a different backend may round its construction by one ULP.
        if args.model=='full_effective':
            p.JOINT_SCALES=jnp.asarray(config['coordinate_scales_SI'],dtype=jnp.float64)
            p.JOINT_TIME=float(config['time_scale_seconds'])
        # Preserve optimizer/RNG state; conflicting settings are not silently applied.
        for name in ['model','seed','microbatch','optimizer','baseline','low_speed']:
            if getattr(args,name)!=config[name]: raise ValueError(f'Resume mismatch: {name}')
        start=checkpoint['step']
        config['requested_updates']=args.steps
        config['mode']='profile' if args.profile else 'smoke' if args.steps<BUDGET else 'full_budget'
        if start>=args.steps:raise ValueError('Resume target must exceed saved step')
        if not out.exists():
            out.mkdir(parents=True)
            original=Path(args.resume).parent
            for filename in ['validation.npz','best.pkl','split_manifest.json']:
                if (original/filename).exists():shutil.copy2(original/filename,out/filename)
            snapshot=out/'code_snapshot';snapshot.mkdir()
            for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,snapshot/file.name)
            shutil.copytree(Path(__file__).parent/'provenance',snapshot/'provenance')
            config['resumed_from']=str(Path(args.resume).resolve())
            record_json(out/'manifest.json',manifest(config))
    else:
        if out.exists(): raise FileExistsError(f'Use a fresh run directory: {out}')
        out.mkdir(parents=True)
        start=0
        config=dict(model=args.model,seed=args.seed,width=500,layers=4,batch_size=512,
                    microbatch=args.microbatch,budget=BUDGET,optimizer=args.optimizer,
                    baseline=args.baseline,dtype='float32' if args.model=='source' else 'float64',
                    validation_seed=1000,final_test_seed=2000,validation_count=10000,test_count=10000,
                    checkpoint_selection='minimum fixed validation gamma-weighted normalized MAE',
                    mode='profile' if args.profile else 'smoke' if args.steps<BUDGET else 'full_budget',
                    requested_updates=args.steps,speed_domain='conditioned below 0.7c' if args.low_speed else '0–0.99999c',
                    low_speed=args.low_speed,training_rng='fold_in(PRNGKey(seed), integer_step); source preserves epoch sampler',
                    time_convention='source' if args.model=='source' else 'mission scaled time' if args.model=='incomplete' else 'effective common coordinate time',
                    position_feature_map='translation divided by mission distance inside the differentiated action' if args.model=='full_effective' else 'identity',
                    coordinate_scales_SI=np.asarray(p.JOINT_SCALES).tolist() if args.model=='full_effective' else [p.INC_LENGTH] if args.model=='incomplete' else None,
                    time_scale_seconds=p.JOINT_TIME if args.model=='full_effective' else p.INC_TIME if args.model=='incomplete' else 1.,
                    action_scale_J=p.ENERGY_SCALE if args.model!='source' else None,
                    acceptance={'band_component_scaled_MAE':.01,'speed_MAE_c':.001,'speed_max_error_c':.005,
                                'work_residual':.001,'gravity_error_g':.01,'attitude_MAE_deg':1.,'extension_MAE_m':.01})
        record_json(out/'config.json',config);record_json(out/'manifest.json',manifest(config))
        snapshot=out/'code_snapshot';snapshot.mkdir()
        for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,snapshot/file.name)
        shutil.copytree(Path(__file__).parent/'provenance',snapshot/'provenance')
    config['interpretation'] = 'coupled mathematical benchmark; physical high-speed claims gated' if args.model == 'full_effective' else 'source reproduction' if args.model == 'source' else 'single relativistic kinetic action with known external force'
    config['mathematical_benchmark'] = mathematical
    config['physical_full_claim'] = False
    if mathematical and args.model == 'full_effective':
        config['numerical_audit_sha256'] = hashlib.sha256(Path(args.physics_audit).read_bytes()).hexdigest()
    if 512%args.microbatch: raise ValueError('microbatch must divide 512')
    dimensions=9 if args.model=='full_effective' else 1
    inputs=2*dimensions+(dimensions if args.baseline else 1 if args.model=='source' else 0)
    dtype=jnp.float32 if args.model=='source' else jnp.float64
    parameters,forward=network(inputs,dimensions if args.baseline else 1,args.seed,dtype=dtype)
    if args.model=='source' and args.optimizer=='faithful':
        opt_init,opt_update,get_params=optimizers.adam(learning_rate)
    else:
        opt_init,opt_update,get_params=optimizers.adam(lambda i:learning_rate(i/BUDGET))
    state=opt_init(parameters)
    best_score=float('inf');best_step=None;history=[]
    if args.resume:
        state=jax.tree.map(jnp.asarray,checkpoint['optimizer_state'])
        best_score=checkpoint['best_score'];best_step=checkpoint['best_step'];history=checkpoint['history']
    sample=partial(p.sample_incomplete,count=512,low_speed=args.low_speed)
    if args.model=='full_effective':
        eq=p.equilibrium()
        sample=partial(p.sample_joint,count=512,eq=eq,low_speed=args.low_speed)
    sample=jax.jit(sample)
    scales=jnp.ones(dimensions,dtype=dtype)
    if args.model=='full_effective':
        # Freeze from an independent reference pilot before any neural training.
        pilot=sample(jax.random.PRNGKey(999))
        scales=jnp.maximum(jnp.mean(jnp.abs(pilot[3]),axis=0),1e-15)
    if args.resume:
        scales=jnp.array(checkpoint['acceleration_scales'],dtype=dtype)
    config['acceleration_scales']=np.asarray(scales).tolist()
    record_json(out/'config.json',config)
    record_json(out/f'invocation_{start:06d}.json',manifest(config))
    count=512//args.microbatch
    def objective(params,batch,denominator):
        q,v,g,target=batch
        prediction=jax.vmap(predictor(args.model,forward,params,args.baseline))(q,v,g)
        error=jnp.mean(jnp.abs(prediction-target)/scales,axis=1)
        return jnp.sum(error*weights(args.model,v))/denominator
    @jax.jit
    def update(step,optstate,batch):
        params=get_params(optstate)
        denominator=jnp.sum(weights(args.model,batch[1]))
        chunks=tuple(z.reshape((count,args.microbatch,dimensions)) for z in batch)
        zero=jax.tree.map(jnp.zeros_like,params)
        def accumulate(carry,chunk):
            value,grad=jax.value_and_grad(objective)(params,chunk,denominator)
            prior_loss,prior_grad=carry
            return (prior_loss+value,jax.tree.map(jnp.add,prior_grad,grad)),None
        (loss,gradient),_=jax.lax.scan(accumulate,(jnp.array(0.,dtype=dtype),zero),chunks)
        finite=jnp.logical_and(jnp.isfinite(loss),jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradient)])))
        index=step/BUDGET if args.model=='source' and args.optimizer=='faithful' else step
        return opt_update(index,gradient,optstate),loss,finite
    @jax.jit
    def errors(params,batch):
        q,v,g,target=batch
        pred=jax.vmap(predictor(args.model,forward,params,args.baseline))(q,v,g)
        return pred,weights(args.model,v)
    def validation(params,batch):
        sums=np.zeros(dimensions);numerator=denominator=0.
        for k in range(0,len(batch[0]),args.microbatch):
            chunk=tuple(z[k:k+args.microbatch] for z in batch)
            prediction,w=errors(params,chunk)
            err=np.abs(np.asarray(prediction)-np.asarray(chunk[3]))/np.asarray(scales)
            sums+=err.sum(axis=0)
            numerator+=float(np.sum(err.mean(axis=1)*np.asarray(w)));denominator+=float(jnp.sum(w))
        score=numerator/denominator
        if not np.isfinite(score): raise FloatingPointError('Nonfinite validation score')
        return score,(sums/len(batch[0])).tolist()
    validation_data=None
    if not args.profile:
        if args.resume and (out/'validation.npz').exists():
            with np.load(out/'validation.npz') as data: validation_data=tuple(jnp.array(data[f'arr_{k}']) for k in range(4))
        else:
            if args.model=='source':validation_data=source_validation(1000,10000)
            elif args.model=='incomplete':validation_data=p.sample_incomplete(jax.random.PRNGKey(1000),10000,True,args.low_speed)
            else:validation_data=p.sample_joint(jax.random.PRNGKey(1000),10000,p.equilibrium(),True,args.low_speed)
            np.savez(out/'validation.npz',*[np.asarray(x) for x in validation_data])
        validation_hash=hashlib.sha256((out/'validation.npz').read_bytes()).hexdigest()
        record_json(out/'split_manifest.json',{'validation_sha256':validation_hash,'final_test_seed':2000,'final_test_used':False})
    source_data=None;epoch_loaded=None;elapsed_steps=[];started=time.perf_counter();completed_step=start
    committed=(state,start)
    def save(name,step):
        step=committed[1]
        content=dict(step=step,config=config,optimizer_state=jax.device_get(committed[0]),
                     training_rng_key=np.asarray(jax.random.PRNGKey(args.seed)),next_integer_step=step,
                     source_epoch=step//3000,source_minibatch=step%3000,
                     acceleration_scales=np.asarray(scales),best_score=best_score,best_step=best_step,
                     history=history,seconds=time.perf_counter()-started)
        temporary=out/(name+'.tmp')
        with temporary.open('wb') as f:pickle.dump(content,f)
        temporary.replace(out/name)
    try:
        if validation_data is not None and start==0:
            initial_score,initial_components=validation(get_params(state),validation_data)
            row=dict(step=0,training_loss=None,validation_score=initial_score,validation_components=initial_components,elapsed_s=0.)
            history.append(row)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            # Checkpoint selection includes the initial state, as predeclared.
            best_score,best_step=initial_score,0;save('initial.pkl',0);save('best.pkl',0)
        for step in range(start,args.steps):
            if args.model=='source':
                epoch=step//3000
                if epoch_loaded!=epoch:
                    source_data=source_epoch(epoch);epoch_loaded=epoch
                batch=tuple(z[step%3000] for z in source_data)
            else:
                batch=sample(jax.random.fold_in(jax.random.PRNGKey(args.seed),step))
            t=time.perf_counter()
            candidate,loss,finite=update(step,state,batch)
            jax.block_until_ready((candidate,loss,finite))
            duration=time.perf_counter()-t
            if step>start: elapsed_steps.append(duration)
            if not bool(finite):
                save('failed.pkl',completed_step)
                raise FloatingPointError(f'Nonfinite training loss/gradient at update {step+1}')
            state=candidate
            completed_step=step+1
            # Atomic Python tuple assignment defines the last resumable state.
            committed=(state,completed_step)
            if step==start:
                params=get_params(state)
                if args.baseline:diag={}
                else:
                    if args.model=='source':
                        action=neural_action(args.model,forward,params,batch[2][0])
                        diag=diagnostics(action,batch[0][0]/10,batch[1][0])
                    else:
                        action=neural_action(args.model,forward,params)
                        diag=diagnostics(action,batch[0][0],batch[1][0])
                    diag={k:np.asarray(v).tolist() for k,v in diag.items()}
                record_json(out/'after_first_update_hessian.json',diag)
                record_json(out/'actual_parameter_dtypes.json',sorted(set(str(x.dtype) for x in jax.tree.leaves(params))))
                print(f'{args.model} update={step+1} loss={float(loss):.6g} compile_and_update_s={duration:.3f}',flush=True)
            if (step+1)%1000==0 or step+1==args.steps:
                score,components=(None,None) if args.profile else validation(get_params(state),validation_data)
                row=dict(step=step+1,training_loss=float(loss),validation_score=score,validation_components=components,
                         elapsed_s=time.perf_counter()-started)
                if not args.baseline:
                    params=get_params(state)
                    probe=tuple(z[:16] for z in (validation_data if validation_data is not None else batch))
                    if args.model=='source':
                        inspect_hessian=lambda q,v,g:diagnostics(neural_action(args.model,forward,params,g),q/10,v)
                    else:
                        inspect_hessian=lambda q,v,g:diagnostics(neural_action(args.model,forward,params),q,v)
                    stats=jax.jit(jax.vmap(inspect_hessian))(*probe[:3])
                    row['hessian']={'min_singular':float(jnp.min(stats['min_singular'])),
                                    'max_condition':float(jnp.max(stats['condition'])) if bool(jnp.all(jnp.isfinite(stats['condition']))) else 'inf',
                                    'minimum_retained_rank':int(jnp.min(stats['retained_rank'])),
                                    'positive_definite_fraction':float(jnp.mean(jnp.all(stats['eigenvalues']>0,axis=1)))}
                history.append(row)
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if score is not None and score<best_score:
                    best_score,best_step=score,step+1;save('best.pkl',step+1)
                print(json.dumps(row),flush=True)
            if (step+1)%10000==0:save(f'checkpoint_{step+1:06d}.pkl',step+1)
            if (step+1)%100==0:
                record_json(out/'progress.json',dict(step=step+1,total=args.steps,elapsed_s=time.perf_counter()-started,
                                                    recent_update_seconds=duration,last_training_loss=float(loss)))
        save('final.pkl',args.steps)
        profile=dict(status='completed',updates=args.steps-start,median_update_seconds=float(np.median(elapsed_steps)) if elapsed_steps else None,
                     estimated_seconds_300000=float(np.median(elapsed_steps))*BUDGET if elapsed_steps else None,
                     max_rss_bytes=max_rss_bytes(),
                     validation_selected_step=best_step,scientific_result=args.steps==BUDGET and not args.profile)
        record_json(out/'runtime.json',profile);print(json.dumps(profile),flush=True)
        return profile
    except (Exception,KeyboardInterrupt) as error:
        if not (out/'failed.pkl').exists():save('interrupted.pkl',completed_step)
        record_json(out/'failure.json',dict(type=type(error).__name__,message=str(error),step=completed_step,
                                          scientific_result=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['source','incomplete','full_effective'],required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--steps',type=int,default=BUDGET)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--microbatch',type=int,default=32)
    parser.add_argument('--optimizer',choices=['faithful','corrected'],default='corrected')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--low-speed',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--mathematical-benchmark',action='store_true')
    parser.add_argument('--physics-audit')
    args=parser.parse_args()
    if not 0<args.steps<=BUDGET:parser.error('steps must lie in 1..300000')
    if args.model!='source' and args.optimizer=='faithful':parser.error('PHM Adam uses integer updates')
    train(args)


if __name__=='__main__':main()
