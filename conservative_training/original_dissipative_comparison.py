"""Original derivation comparison using its archived deterministic parameter set.

The archived model is read-only. Its translation, rotor and attitude partitions
are shown separately; no fabricated continuous multiyear internal history is
used. Saved LNNs have different hardware/physics, so this is a model-scope
comparison, not a controlled estimate of learned-law error.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np
from scipy.spatial.transform import Rotation
os.environ.setdefault('MPLCONFIGDIR', str(Path(__import__('tempfile').gettempdir())/'hail_mary_lnn_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import physics as p
from .engine import network
from .train import predictor,record_json
from .rollout_utils import integrate

DEFAULT_SOURCE=Path(__file__).resolve().parent.parent/'references/original_system'


def load_original(source=DEFAULT_SOURCE):
    file=Path(source)/'src/hail_mary_model.py'
    spec=importlib.util.spec_from_file_location('archived_original_hail_mary_model',file)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def load_network(checkpoint_path,expected):
    with Path(checkpoint_path).open('rb') as stream:ck=pickle.load(stream)
    cfg=ck['config']
    if cfg['model']!=expected or cfg['baseline']:raise ValueError('Incorrect model checkpoint')
    _,_,get=optimizers.adam(.001)
    _,forward=network(2 if expected=='incomplete' else 18,seed=cfg['seed'],dtype=jnp.float64)
    return ck,jax.jit(predictor(expected,forward,get(ck['optimizer_state'])))


def original_force(given,pars,time,x,beta,include_environment,source):
    mass=float(np.interp(time,given['t'],given['mass']))
    switch=float(np.interp(given['switch_tau'],given['tau'],given['t']))
    # Original mass and proper-thrust schedules are supplied to all models.
    # This isolates translational response; neither LNN learns fuel dynamics.
    thrust=(1. if time<switch else -1.)*mass*pars.proper_accel
    if not include_environment:return thrust/(mass*p.A)
    if pars.source_offset+x<=0 or pars.target_offset+pars.distance-x<=0:
        raise FloatingPointError('Predicted path crossed a star; original inverse-distance field is invalid')
    fs=source.stellar_forces(x,mass,pars)
    drag=float(source.drag_force(np.arctanh(beta),pars))
    return (thrust+sum(fs)+drag)/(mass*p.A)


def compare(incomplete_checkpoint,coupled_checkpoint,out,source=DEFAULT_SOURCE):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    src=load_original(source);pars=src.Parameters()
    translation=src.simulate_translation(pars,scenario='nominal',n_eval=1600)
    cent=src.simulate_centrifuge(pars);att=src.simulate_attitude(pars)
    ck,inc=load_network(incomplete_checkpoint,'incomplete')
    cj,joint=load_network(coupled_checkpoint,'full_effective')
    if not translation['arrived']:raise RuntimeError('Original mission reference did not arrive')
    times=translation['t']*p.A/p.C
    switch=float(np.interp(translation['switch_tau'],translation['tau'],translation['t']))*p.A/p.C
    curves={};statuses={}
    for name,environment in [('One-term LNN',False),('One-term LNN + original external forces',True)]:
        def rhs(s,z):
            if abs(z[1])>=1:raise FloatingPointError('Superluminal trial state')
            g=original_force(translation,pars,s*p.C/p.A,z[0]*p.INC_LENGTH,z[1],environment,src)
            return [z[1],float(inc(jnp.array([z[0]]),jnp.array([z[1]]),jnp.array([g]))[0])]
        # Solve the scheduled acceleration and braking phases separately.
        first_grid=np.unique(np.r_[times[times<switch],switch])
        first,status=integrate(rhs,[0.,0.],first_grid,max_calls=10000)
        values=np.full((2,len(times)),np.nan)
        mask=times<switch;values[:,mask]=first[:,:len(first_grid)-1]
        if not status['failed']:
            second_grid=np.r_[switch,times[times>=switch]]
            second,second_status=integrate(rhs,first[:,-1],second_grid,max_calls=10000)
            values[:,~mask]=second[:,1:]
            status['acceleration_phase']=dict(status)
            status['braking_phase']=second_status
            status['failed']=second_status['failed']
            status['last_accepted_time']=second_status['last_accepted_time']
        curves[name]=values;statuses[name]=status
    def exact_rhs(s,z):
        if abs(z[1])>=1:raise FloatingPointError('Superluminal exact-reference trial state')
        g=original_force(translation,pars,s*p.C/p.A,z[0]*p.INC_LENGTH,z[1],True,src)
        return [z[1],g*(1-z[1]**2)**1.5]
    exact,exact_status=integrate(exact_rhs,[0.,0.],times,rtol=1e-10,atol=1e-12,max_calls=10000)
    if exact_status['failed']:raise RuntimeError('Independent coordinate-time reference failed: '+str(exact_status))
    exact_beta_error=float(np.max(np.abs(exact[1]-translation['beta'])))
    exact_position_error=float(np.max(np.abs(exact[0]*p.INC_LENGTH-translation['x']))/pars.distance)
    if exact_beta_error>1e-5 or exact_position_error>1e-5:
        raise RuntimeError(f'Original proper/coordinate-time references disagree: {exact_beta_error}, {exact_position_error}')
    # Original separate-frame internal reference starts at rest in COM translation.
    # Coupled checkpoint uses the same physical coordinates/rates at t=tau=0.
    # Original hardware is different; this is an explicitly out-of-domain scope comparison.
    physical_q=np.r_[np.zeros(6),cent['xh'][0],cent['xc'][0],0.]
    physical_v=np.r_[np.zeros(8),pars.spin_rate]
    initial=np.r_[physical_q/np.asarray(p.JOINT_SCALES),physical_v*p.JOINT_TIME/np.asarray(p.JOINT_SCALES)]
    local_seconds=np.unique(np.r_[np.linspace(0,.01,101),np.linspace(0,600,1201)])
    force=p.force_joint(jnp.zeros(3))
    def joint_rhs(s,z):
        si=z[:9]*np.asarray(p.JOINT_SCALES)
        if np.linalg.norm(z[9:12])>=1:raise FloatingPointError('Superluminal coupled COM trial state')
        if np.any(np.asarray(p.RADII)+si[6:8]<=0):raise FloatingPointError('Nonpositive coupled rotor radius')
        if abs(si[4])>=np.pi/2-.01:raise FloatingPointError('Coupled Euler chart singularity')
        return np.r_[z[9:],np.asarray(joint(jnp.asarray(z[:9]),jnp.asarray(z[9:]),force))]
    coupled,coupled_status=integrate(joint_rhs,initial,local_seconds/p.JOINT_TIME,max_calls=10000)
    coupled_q=coupled[:9]*np.asarray(p.JOINT_SCALES)[:,None]
    coupled_v=coupled[9:]*np.asarray(p.JOINT_SCALES)[:,None]/p.JOINT_TIME
    styles=[('#7844a4','--'),('#14836f',':')]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    ref_time=translation['t']/pars.year
    axes[0,0].plot(ref_time,translation['beta'],color='#d06e1b',label='Original dissipative reference')
    axes[0,1].plot(ref_time,translation['x']/pars.distance,color='#d06e1b',label='Original dissipative reference')
    axes[0,0].plot(ref_time,exact[1],color='#263847',ls=':',lw=1.,label='Independent exact-law check')
    axes[0,1].plot(ref_time,exact[0]*p.INC_LENGTH/pars.distance,color='#263847',ls=':',lw=1.)
    for (name,z),(color,style) in zip(curves.items(),styles):
        axes[0,0].plot(ref_time,z[1],color=color,ls=style,label=name)
        axes[0,1].plot(ref_time,z[0]*p.INC_LENGTH/pars.distance,color=color,ls=style,label=name)
        if statuses[name]['failed']:
            stop=statuses[name]['last_accepted_time']*p.C/p.A/pars.year
            for ax in axes[0]:ax.axvline(stop,color=color,ls=':',alpha=.6)
            axes[0,0].text(.04,.18,'External-force LNN failed near start',transform=axes[0,0].transAxes,color=color,fontsize=8)
    axes[0,0].set(xlabel='Earth time / years',ylabel='Speed / c');axes[0,1].set(xlabel='Earth time / years',ylabel='Position / mission distance')
    axes[1,0].semilogy(ref_time,translation['mass'],color='#d06e1b');axes[1,0].set(xlabel='Earth time / years',ylabel='Supplied spacecraft mass / kg')
    axes[1,1].plot(ref_time,translation['drag_energy'],color='#d06e1b');axes[1,1].set(xlabel='Earth time / years',ylabel='Cumulative gas and dust energy loss / J')
    axes[0,0].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    phase='After full-budget training' if ck['config']['mode']=='full_budget' else 'Provisional trial'
    fig.suptitle(f'Original derivation system versus the one-term LNN\n{phase} checkpoint at {ck["step"]:,} updates',fontsize=14)
    fig.text(.08,.025,'Original density and fuel parameters replace the earlier arbitrary drag strength 0.02.\nMass and thrust schedules are supplied; they are not learned. Vertical dotted lines mark failure; missing tails are not predictions.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.93));fig.savefig(out/'original_mission_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    axes[0,0].plot(cent['tau'],cent['xh'],color='#d06e1b',label='Original dissipative reference')
    axes[0,0].plot(local_seconds,coupled_q[6],color='#7844a4',ls='--',label=f'Coupled LNN ({cj["step"]:,} updates)')
    axes[0,1].plot(cent['tau'],cent['omega'],color='#d06e1b');axes[0,1].plot(local_seconds,coupled_v[8],color='#7844a4',ls='--')
    # Compare body angular speed; Euler angle rates alone are not body omega.
    w=np.full((3,len(local_seconds)),np.nan);count=coupled_status['completed_samples']
    w[:,:count]=np.asarray(jax.vmap(p.body_omega)(jnp.asarray(coupled_q[3:6,:count].T),jnp.asarray(coupled_v[3:6,:count].T))).T
    axes[1,0].plot(att['tau'],np.linalg.norm(att['omega'],axis=0),color='#d06e1b');axes[1,0].plot(local_seconds,np.linalg.norm(w,axis=0),color='#7844a4',ls='--')
    mh,rh,kh,ch=src.cable_properties(pars,'h');mc,rc,kc,cc=src.cable_properties(pars,'c')
    full_loss=ch*cent['vh']**2+cc*cent['vc']**2+pars.radial_coulomb*(cent['vh']*np.tanh(cent['vh']/1e-4)+cent['vc']*np.tanh(cent['vc']/1e-4))+pars.bearing_damping*cent['omega']**2+pars.bearing_coulomb*cent['omega']*np.tanh(cent['omega']/1e-4)
    axes[1,1].plot(cent['tau'],full_loss,color='#d06e1b')
    titles=['Habitat extension / m','Rotor rate / rad s⁻¹','Hub angular speed / rad s⁻¹','Cable and bearing dissipation / W']
    for ax,title in zip(axes.flat,titles):ax.set(xlabel='Local proper time / seconds',ylabel=title);ax.grid(alpha=.2)
    axes[0,0].set_xlim(0,300);axes[0,1].set_xlim(0,300);axes[1,0].set_xlim(0,600);axes[1,1].set_xlim(0,300)
    axes[0,0].legend(fontsize=8)
    if coupled_status['failed']:
        stop=coupled_status['last_accepted_time']*p.JOINT_TIME
        for ax in [axes[0,0],axes[0,1],axes[1,0]]:
            ax.axvline(stop,color='#7844a4',ls=':');ax.annotate('Coupled rollout failed near start',xy=(.04,.90),xycoords='axes fraction',fontsize=8,color='#7844a4')
    fig.suptitle('Original cable bearing and attitude losses with prescribed impacts\nCoupled LNN shown only where its rollout remains valid',fontsize=14)
    fig.text(.07,.02,'Original hardware: 73 m and 24.33 m arms; LNN hardware: 30 m and 20 m. Clocks also differ.\nThis is a scope comparison, not a controlled learning-error score. Original implementation regularizes dry friction with tanh(v/10⁻⁴).',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.93));fig.savefig(out/'original_internal_comparison.png',dpi=160);plt.close(fig)
    source_files=[Path(source)/'src/hail_mary_model.py',Path(source)/'docs/model_specification.md',Path(source).parent/'PHM_Derivations.pdf']
    audit=dict(source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},original_parameters=asdict(pars),
               original_derived={'habitat_stiffness_N_m':kh,'counter_stiffness_N_m':kc,'habitat_damping_N_s_m':ch,'counter_damping_N_s_m':cc},
               incomplete_checkpoint=str(Path(incomplete_checkpoint).resolve()),incomplete_updates=ck['step'],
               coupled_checkpoint=str(Path(coupled_checkpoint).resolve()),coupled_updates=cj['step'],
               original_reference_arrived=bool(translation['arrived']),earth_years=float(ref_time[-1]),crew_years=float(translation['tau'][-1]/pars.year),
               original_mass_initial_kg=float(translation['mass'][0]),original_mass_final_kg=float(translation['mass'][-1]),
               peak_speed_c=float(np.max(translation['beta'])),drag_energy_J=float(translation['drag_energy'][-1]),
               mission_rollouts=statuses,coupled_rollout=coupled_status,coupled_time_scale_seconds=p.JOINT_TIME,
               independent_reference_check={'max_beta_error_c':exact_beta_error,'max_position_error_route_fraction':exact_position_error,'passed':True,'solver':exact_status},
               scientific_result=False,physical_full_claim=False,controlled_coupled_accuracy_score=False,
               nominal_EM_and_translation_impacts='zero in archived nominal reference',
               radial_impact={'time_seconds':120.,'habitat_impulse_N_s':5000.,'counter_impulse_N_s':-5000.},
               attitude_impact={'time_seconds':100.,'angular_impulse_N_m_s':[0.,2e6,-1e6]},
               caveats=['Archived deterministic reference has separate translation/rotor/attitude partitions rather than a fully coupled variational spacecraft',
                        'Original geometry, mass and inertias differ from current coupled LNN training target',
                        'Original internal clock is proper time; coupled mathematical LNN uses coordinate time',
                        'Original code smooths dry friction with tanh; literal PDF sign law is nonsmooth',
                        'Mass and thrust are externally supplied; conservative networks do not learn the propulsion subsystem'])
    record_json(out/'metrics.json',audit)
    np.savez(out/'reference_and_predictions.npz',earth_time=translation['t'],crew_time=translation['tau'],reference_beta=translation['beta'],reference_x=translation['x'],reference_mass=translation['mass'],reference_drag_energy=translation['drag_energy'],
             one_term=curves['One-term LNN'],one_term_external=curves['One-term LNN + original external forces'],local_time=local_seconds,coupled_q=coupled_q,coupled_v=coupled_v,
             original_rotor_time=cent['tau'],original_xh=cent['xh'],original_xc=cent['xc'],original_spin=cent['omega'],original_internal_dissipation=full_loss,original_attitude_time=att['tau'],original_attitude_q=att['q'],original_attitude_omega=att['omega'])
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--incomplete',required=True);parser.add_argument('--coupled',required=True);parser.add_argument('--out',required=True);parser.add_argument('--source',default=str(DEFAULT_SOURCE))
    a=parser.parse_args();compare(a.incomplete,a.coupled,a.out,a.source)
