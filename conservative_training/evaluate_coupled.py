"""Held-out evaluations of the coupled mathematical benchmark, with failure counts.

Short integrations resolve internal motion directly. The mission-duration case
has a fixed computation budget and is reported failed if direct resolution is
infeasible; it is never silently replaced with averaged or frozen rotation.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(__import__('tempfile').gettempdir())/'hail_mary_lnn_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .engine import network, energy, diagnostics
from .train import predictor, neural_action, record_json
from .rollout_utils import integrate
from . import physics as p

NAMES = ['x','y','z','roll','pitch','yaw','habitat extension','counter extension','rotor angle']


def evaluate(checkpoint_path, smoke=False, label='', verification=False):
    path = Path(checkpoint_path)
    with path.open('rb') as stream: ck = pickle.load(stream)
    cfg = ck['config']
    if cfg['model'] != 'full_effective': raise ValueError('Requires a coupled checkpoint')
    out = path.parent/(('verification_evaluation' if verification else 'smoke_evaluation' if smoke else 'final_evaluation')+label)
    out.mkdir(exist_ok=False)
    _, _, get_params = optimizers.adam(.001)
    params = get_params(ck['optimizer_state'])
    _, forward = network(27 if cfg['baseline'] else 18, 9 if cfg['baseline'] else 1, seed=cfg['seed'], dtype=jnp.float64)
    predict = jax.jit(predictor('full_effective', forward, params, cfg['baseline']))
    seed = 1002 if verification else 1001 if smoke else 2000
    sample = jax.jit(partial(p.sample_joint, count=16 if verification else 10000, eq=p.equilibrium(), stratified=True))
    q,v,f,target = sample(jax.random.PRNGKey(seed))
    guesses = np.concatenate([np.asarray(jax.vmap(predict)(q[i:i+16],v[i:i+16],f[i:i+16])) for i in range(0,len(q),16)])
    scales = np.asarray(ck['acceleration_scales'])
    err = np.abs(guesses-np.asarray(target))/scales
    speed = np.linalg.norm(np.asarray(v)[:,:3],axis=1)
    bounds = [0.,.7,.9,.99,.99999]; bands=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        mask=(speed>=lo)&(speed<hi); component=err[mask].mean(axis=0)
        bands.append(dict(low=lo,high=hi,count=int(mask.sum()),normalized_component_MAE=component.tolist(),
                          component_MAE_SI=(np.abs(guesses-np.asarray(target))[mask].mean(axis=0)*np.asarray(p.JOINT_SCALES)/p.JOINT_TIME**2).tolist(),
                          passed=bool(np.isfinite(component).all() and np.all(component<.01))))
    reference = jax.jit(p.joint_reference)
    true_energy = jax.jit(partial(energy,p.joint_action))
    learned_action = neural_action('full_effective',forward,params)
    learned_energy = None if cfg['baseline'] else jax.jit(partial(energy,learned_action))
    def measurements(q,v,force,accel):
        a=accel(q,v,force);si=q*p.JOINT_SCALES;vs=v*p.JOINT_SCALES/p.JOINT_TIME
        asi=a*p.JOINT_SCALES/p.JOINT_TIME**2
        points=p.internal_point_accelerations_si(si,vs,asi)
        internal=p.internal_kinetic(si,vs)+.5*jnp.dot(p.STIFFNESS,si[6:8]**2)
        return jnp.array([jnp.linalg.norm(points[1])/p.G0,internal])
    true_measure=jax.jit(partial(measurements,accel=reference))
    learned_measure=jax.jit(partial(measurements,accel=predict))
    exact_stats = jax.vmap(partial(diagnostics,p.joint_action))(q[:16],v[:16])
    learned_stats = None if cfg['baseline'] else jax.vmap(partial(diagnostics,learned_action))(q[:16],v[:16])
    def valid(q0,v0):
        si=q0*np.asarray(p.JOINT_SCALES)
        if np.linalg.norm(v0[:3])>=1: raise FloatingPointError('Superluminal COM trial state')
        if abs(si[4])>=np.pi/2-.01: raise FloatingPointError('Euler coordinate chart singularity')
        if np.any(np.asarray(p.RADII)+si[6:8]<=0): raise FloatingPointError('Nonpositive rotor radius')
        if np.min(np.linalg.norm(np.asarray(p.STARS)-si[:3],axis=1))<p.AU:
            raise FloatingPointError('Trial state inside the one-AU exclusion')
    def rollout(q0,v0,force,horizon_seconds,name):
        times=np.unique(np.r_[np.linspace(0,min(1.,horizon_seconds)/p.JOINT_TIME,51),np.linspace(0,horizon_seconds/p.JOINT_TIME,201)])
        initial=np.r_[q0,v0,0.]
        def field(accel):
            def rhs(t,z):
                valid(z[:9],z[9:18])
                a=np.asarray(accel(jnp.asarray(z[:9]),jnp.asarray(z[9:18]),force))
                return np.r_[z[9:18],a,np.dot(np.asarray(force),z[9:18])]
            return rhs
        truth,ts=integrate(field(reference),initial,times,rtol=1e-10,atol=1e-12,max_calls=10000)
        prediction,ls=integrate(field(predict),initial,times,rtol=1e-9,atol=1e-11,max_calls=10000)
        record=dict(name=name,horizon_seconds=horizon_seconds,short_horizon_seconds=min(1.,horizon_seconds),
                    initial_q=np.asarray(q0).tolist(),initial_v=np.asarray(v0).tolist(),force=np.asarray(force).tolist(),
                    reference_solver=ts,learned_solver=ls,failed=ts['failed'] or ls['failed'])
        measured={}
        for kind,z,measure in [('reference',truth,true_measure),('learned',prediction,learned_measure)]:
            count=ts['completed_samples'] if kind=='reference' else ls['completed_samples']
            measured[kind]=np.full((len(times),2),np.nan)
            measured[kind][:count]=np.asarray(jax.vmap(lambda q,v:measure(q,v,force))(jnp.asarray(z[:9,:count].T),jnp.asarray(z[9:18,:count].T)))
        energy_traces={}
        for kind,z,status in [('reference',truth,ts),('learned',prediction,ls)]:
            count=status['completed_samples']; values=np.full(len(times),np.nan)
            values[:count]=np.asarray(jax.vmap(true_energy)(jnp.asarray(z[:9,:count].T),jnp.asarray(z[9:18,:count].T)))
            energy_traces[kind]=values-values[0]-z[18]
        if learned_energy is not None:
            count=ls['completed_samples'];values=np.full(len(times),np.nan)
            values[:count]=np.asarray(jax.vmap(learned_energy)(jnp.asarray(prediction[:9,:count].T),jnp.asarray(prediction[9:18,:count].T)))
            energy_traces['network']=values-values[0]-prediction[18]
        if not record['failed']:
            dq=(prediction[:9]-truth[:9])*np.asarray(p.JOINT_SCALES)[:,None]
            dv=(prediction[9:18]-truth[9:18])*np.asarray(p.JOINT_SCALES)[:,None]/p.JOINT_TIME
            speed_error=np.linalg.norm(dv[:3],axis=0)/p.C
            te=np.asarray(jax.vmap(true_energy)(jnp.asarray(prediction[:9].T),jnp.asarray(prediction[9:18].T)))
            residual=te-te[0]-prediction[18]
            record.update(position_MAE_m=np.abs(dq[:3]).mean(axis=1).tolist(),
                          velocity_vector_MAE_c=float(speed_error.mean()),velocity_vector_max_error_c=float(speed_error.max()),
                          attitude_MAE_deg=(np.abs(dq[3:6]).mean(axis=1)*180/np.pi).tolist(),
                          extension_MAE_m=np.abs(dq[6:8]).mean(axis=1).tolist(),
                          rotor_rate_MAE_rad_s=float(np.abs(dv[8]).mean()),
                          habitat_acceleration_MAE_g=float(np.abs(measured['learned'][:,0]-measured['reference'][:,0]).mean()),
                          true_energy_work_max_residual_mc2=float(np.abs(residual).max()))
            if learned_energy is not None:
                le=np.asarray(jax.vmap(learned_energy)(jnp.asarray(prediction[:9].T),jnp.asarray(prediction[9:18].T)))
                record['learned_energy_work_max_residual_mc2']=float(np.abs(le-le[0]-prediction[18]).max())
            record['passed_evaluated_thresholds']=bool(speed_error.mean()<.001 and speed_error.max()<.005 and np.all(np.abs(dq[3:6]).mean(axis=1)<np.pi/180) and np.all(np.abs(dq[6:8]).mean(axis=1)<.01) and np.abs(residual).max()<.001 and record['habitat_acceleration_MAE_g']<.01)
        np.savez(out/(name+'.npz'),times_seconds=times*p.JOINT_TIME,truth=truth,prediction=prediction,reference_measurements=measured['reference'],learned_measurements=measured['learned'])
        if name in ['heldout_00','mission_short','mission_long']:
            fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
            for z,kind,style in [(truth,'reference','-'),(prediction,'LNN','--')]:
                axes[0].plot(times*p.JOINT_TIME,np.linalg.norm(z[9:12],axis=0),style,label=kind)
                axes[1].plot(times*p.JOINT_TIME,z[3]*float(p.JOINT_SCALES[3])*180/np.pi,style,label=kind)
                axes[2].plot(times*p.JOINT_TIME,z[6]*float(p.JOINT_SCALES[6]),style,label=kind)
            axes[0].set_ylabel('COM speed / c');axes[1].set_ylabel('Roll / degrees');axes[2].set_ylabel('Habitat extension / m');axes[2].set_xlabel('Coordinate time / seconds')
            for ax,status in [(a,s) for a in axes for s in [ts,ls]]:
                if status['failed']:ax.axvline(status['last_accepted_time']*p.JOINT_TIME,color='red',ls=':',alpha=.5)
            axes[0].legend();fig.suptitle(name+(' — FAILED; blank tails are not predictions' if record['failed'] else '')+'\nCoupled mathematical benchmark')
            fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
            fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
            for z,kind,style in [(truth,'reference','-'),(prediction,'learned','--')]:
                axes[0].plot(times*p.JOINT_TIME,z[17]*float(p.JOINT_SCALES[8])/p.JOINT_TIME,style,label=kind)
                axes[1].plot(times*p.JOINT_TIME,measured[kind][:,0],style,label=kind)
                axes[2].plot(times*p.JOINT_TIME,measured[kind][:,1],style,label=kind)
            axes[0].set_ylabel('Rotor angle rate / rad s⁻¹');axes[1].set_ylabel('Habitat acceleration / g');axes[2].set_ylabel('Internal kinetic and elastic energy / J');axes[2].set_xlabel('Coordinate time / seconds');axes[0].legend()
            fig.suptitle(name+(' — FAILED; blank tails are not predictions' if record['failed'] else '')+'\nInternal motion in the coupled mathematical benchmark')
            fig.tight_layout();fig.savefig(out/(name+'_internal.png'),dpi=150);plt.close(fig)
            fig,ax=plt.subplots(figsize=(9,4))
            for kind,residual in energy_traces.items():ax.plot(times*p.JOINT_TIME,residual,label=kind+' energy minus work')
            ax.set(xlabel='Coordinate time / seconds',ylabel='Energy work residual / mc²',title=name+(' — FAILED; blank tails are not predictions' if record['failed'] else ''));ax.legend();fig.tight_layout();fig.savefig(out/(name+'_energy.png'),dpi=150);plt.close(fig)
        return record
    # Independent 40 cases, ten per speed band. Directly resolve internal periods.
    results=[rollout(np.asarray(q[i]),np.asarray(v[i]),f[i],.1 if verification else 10.,f'heldout_{i:02d}') for i in range(4 if verification else 40)]
    eq=p.equilibrium();q0=np.r_[np.zeros(6),eq[:2],0.]/np.asarray(p.JOINT_SCALES)
    v0=np.r_[np.zeros(8),eq[2]]*p.JOINT_TIME/np.asarray(p.JOINT_SCALES)
    force=p.force_joint(jnp.array([1.,0.,0.]))
    results.append(rollout(q0,v0,force,.1 if verification else 10.,'mission_short'))
    mission_seconds=float(np.sqrt((p.Q_ROUTE/2)*(p.Q_ROUTE/2+2))*p.INC_TIME)
    if not verification:results.append(rollout(q0,v0,force,mission_seconds,'mission_long'))
    runtime=path.parent/'runtime.json';completed=runtime.exists() and json.loads(runtime.read_text(encoding='utf-8')).get('scientific_result',False)
    metrics=dict(model='full_effective',interpretation='mathematical benchmark; physical high-speed claims remain gated',
                 physical_full_claim=False,checkpoint_step=ck['step'],seed=cfg['seed'],scientific_result=completed and not smoke and not verification,
                 final_test_used=not smoke and not verification,verification_only=verification,test_seed=seed,acceleration_bands=bands,rollouts=results,
                 failed_rollouts=sum(z['failed'] for z in results),
                 reference_minimum_rank=int(np.min(exact_stats['retained_rank'])),
                 learned_minimum_rank=None if learned_stats is None else int(np.min(learned_stats['retained_rank'])),
                 benchmark_evaluated_thresholds_passed=all(b['passed'] for b in bands) and all(r.get('passed_evaluated_thresholds',False) for r in results),
                 complete_plan_acceptance=False,
                 remaining_acceptance='Five-seed aggregation, internal energy transfer tests, mission stiffness resolution, interaction ablation ladder and physical audit remain required')
    record_json(out/'metrics.json',metrics)
    np.savez(out/'derivatives.npz',q=np.asarray(q),v=np.asarray(v),force=np.asarray(f),target=np.asarray(target),prediction=guesses,scales=scales)
    fig,ax=plt.subplots(figsize=(11,4));image=ax.imshow(np.log10(np.maximum(np.array([b['normalized_component_MAE'] for b in bands]),1e-16)),aspect='auto')
    ax.set(xticks=range(9),xticklabels=NAMES,yticks=range(4),yticklabels=['0–.7c','.7–.9c','.9–.99c','.99–.99999c'])
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right');fig.colorbar(image,ax=ax,label='log10(MAE / fixed reference scale)');fig.tight_layout();fig.savefig(out/'component_errors.png',dpi=150);plt.close(fig)
    rows=[x for x in ck['history'] if x['training_loss'] is not None]
    if rows:
        fig,ax=plt.subplots(figsize=(8,4));ax.semilogy([r['step'] for r in rows],[r['training_loss'] for r in rows],label='training');ax.semilogy([r['step'] for r in rows],[r['validation_score'] for r in rows],label='validation');ax.legend();ax.set(xlabel='Update',ylabel='Weighted component-scaled MAE');fig.tight_layout();fig.savefig(out/'loss.png',dpi=150);plt.close(fig)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--smoke',action='store_true');parser.add_argument('--label',default='');parser.add_argument('--verification',action='store_true')
    a=parser.parse_args();evaluate(a.checkpoint,a.smoke,a.label,a.verification)
