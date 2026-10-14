"""Executable reference tests and full-model audit gates."""
import json
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from .engine import acceleration, energy, diagnostics
from . import physics as p


def scalar(value):
    if isinstance(value,dict): return {k:scalar(v) for k,v in value.items()}
    return np.asarray(value).tolist()


def run_audit(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result={"dtype":"float64", "full_model_status":"effective mathematical benchmark only",
            "time_convention":"single SI coordinate time for the benchmark; no high-speed physical claim",
            "full_physical_training_authorized_by_gate":False}
    q,v,g,labels=p.sample_incomplete(jax.random.PRNGKey(142),1000,True)
    automatic=jax.jit(jax.vmap(partial(acceleration,p.incomplete_action)))(q,v,g)
    error=float(jnp.max(jnp.abs(automatic-labels)))
    assert error<1e-10,error
    result['incomplete_explicit_autodiff_max_error']=error
    errors=[];work_errors=[]
    for force in [-10.,-1.,0.,1.,10.]:
        for beta0 in [-.9,0.,.9]:
            times=np.linspace(0,1,201)
            sol=solve_ivp(lambda t,z:[z[1],force*(1-z[1]**2)**1.5],(0,1),[0,beta0],
                          method='DOP853',t_eval=times,rtol=1e-12,atol=1e-13)
            exact=np.asarray(p.constant_force_solution(jnp.array(times),force,beta0=beta0))
            errors.append(float(np.max(np.abs(sol.y.T-exact))))
            H=1/np.sqrt(1-sol.y[1]**2)-1
            work_errors.append(float(np.max(np.abs(H-H[0]-force*sol.y[0]))))
    assert max(errors)<1e-8
    assert max(work_errors)<1e-7
    result['incomplete_solver_exact_max_error']=max(errors)
    result['incomplete_work_max_residual']=max(work_errors)
    eq=p.equilibrium();result['corrected_equilibrium']={'extension_m':eq[:2].tolist(),'spin_rad_s':eq[2],
                                                     'spin_rpm':eq[2]*60/(2*np.pi)}
    # Explicit PDF limit: constrained/anchored hub, no recentering, no external fields.
    siq=jnp.array([0.,0.,0.,.1,-.2,.15,eq[0]+.01,eq[1]-.01,.4])
    siv=jnp.array([0.,0.,0.,0.,0.,0.,.03,-.02,eq[2]])
    # Independent vector relativistic Lorentz + stellar-force limit, tide disabled.
    vector_v=siv.at[:3].set(p.C*jnp.array([.6,.2,-.1]))
    vector_g=jnp.array([1.,-2.,.5])
    vector_auto=p.joint_reference(siq/p.JOINT_SCALES,vector_v*p.JOINT_TIME/p.JOINT_SCALES,
                                  p.force_joint(vector_g),tidal=False)
    dr=p.STARS-siq[:3];dist=jnp.linalg.norm(dr,axis=1)
    F=p.MASS*p.A*vector_g+jnp.sum((p.MASS*p.GM-p.K)[:,None]*dr/dist[:,None]**3,axis=0)
    F+=p.CHARGE*(p.E_FIELD+jnp.cross(vector_v[:3],p.B_FIELD))
    beta=vector_v[:3]/p.C
    expected=(F-beta*jnp.dot(beta,F))*jnp.sqrt(1-jnp.dot(beta,beta))/p.MASS
    recovered=vector_auto[:3]*p.JOINT_SCALES[:3]/p.JOINT_TIME**2
    error=float(jnp.max(jnp.abs(recovered-expected)))
    assert error<1e-8,error
    result['vector_translation_Lorentz_stellar_limit_error_m_s2']=error
    cent=lambda e,w: p.internal_kinetic(siq.at[6:9].set(e),siv.at[6:9].set(w),False)-.5*jnp.dot(p.STIFFNESS,e[:2]**2)
    cent_acc=acceleration(cent,siq[6:9],siv[6:9],jnp.zeros(3))
    radii=p.RADII+siq[6:8]
    explicit=jnp.concatenate([radii*siv[8]**2-p.STIFFNESS/p.MASSES[1:]*siq[6:8],
                 jnp.array([-2*jnp.sum(p.MASSES[1:]*radii*siv[6:8])*siv[8]/jnp.sum(p.MASSES[1:]*radii**2)])])
    assert float(jnp.max(jnp.abs(cent_acc-explicit)))<1e-9
    result['anchored_hub_pdf_limit_max_error']=float(jnp.max(jnp.abs(cent_acc-explicit)))
    # COM correction T_centered = T_raw - M/2 |V_center_raw|², including moving hub.
    raw=lambda z: jnp.stack([jnp.zeros(3),(p.RADII[0]+z[0])*jnp.array([0.,jnp.cos(z[2]),jnp.sin(z[2])]),
                            -(p.RADII[1]+z[1])*jnp.array([0.,jnp.cos(z[2]),jnp.sin(z[2])])])
    loc,rate=jax.jvp(raw,(siq[6:9],),(siv[6:9],))
    Vbar=jnp.sum(p.MASSES[:,None]*rate,axis=0)/p.MASS
    raw_T=p.internal_kinetic(siq,siv,False)
    centered_T=p.internal_kinetic(siq,siv,True)
    recenter_error=float(jnp.abs(centered_T-(raw_T-.5*p.MASS*jnp.dot(Vbar,Vbar))))
    assert recenter_error<1e-8
    result['COM_kinetic_identity_abs_J']=recenter_error
    # Five-coordinate axial/roll prototype; constrained limit of the same action.
    # Rotor motion is in the transverse plane. This is reduced, not the vector target.
    indices=jnp.array([0,3,6,7,8])
    reduced_q=siq[indices]
    reduced_v=siv[indices]
    embed=lambda z:jnp.zeros(9).at[indices].set(z)
    reduced=lambda z,w:p.internal_kinetic(embed(z),embed(w))-.5*jnp.dot(p.STIFFNESS,z[2:4]**2)
    red_auto=acceleration(reduced,reduced_q,reduced_v,jnp.zeros(5))
    mh,mc=p.MASSES[1:];rh,rc=p.RADII+reduced_q[2:4]
    offset=(mh*rh-mc*rc)/p.MASS
    mass_matrix=jnp.array([[mh-mh**2/p.MASS,mh*mc/p.MASS],[mh*mc/p.MASS,mc-mc**2/p.MASS]])
    inertia=mh*rh**2+mc*rc**2-(mh*rh-mc*rc)**2/p.MASS
    grad_inertia=2*jnp.array([mh*(rh-offset),mc*(rc+offset)])
    total_spin=reduced_v[1]+reduced_v[4]
    radial_a=jnp.linalg.solve(mass_matrix,.5*grad_inertia*total_spin**2-p.STIFFNESS*reduced_q[2:4])
    spin_a=-jnp.dot(grad_inertia,reduced_v[2:4])/inertia*total_spin
    red_explicit=jnp.array([0.,0.,radial_a[0],radial_a[1],spin_a])
    result['reduced_axial_roll_prototype_max_error']=float(jnp.max(jnp.abs(red_auto-red_explicit)))
    assert result['reduced_axial_roll_prototype_max_error']<1e-8
    # Differentiate only internal action to avoid cancellation against mc².
    isolated=lambda z,w:p.internal_kinetic(z,w)-.5*jnp.dot(p.STIFFNESS,z[6:8]**2)
    iq=siq.at[3:6].set(jnp.array([.1,-.2,.15]));iv=siv.at[3:6].set(jnp.array([.01,-.015,.02]))
    direction=jnp.array([0.,0.,0.,.3,-.2,.1,.4,-.3,.2])
    derivative=jnp.dot(jax.grad(isolated,0)(iq,iv),direction)
    fd_errors=[]
    for step in [1e-4,1e-5,1e-6]:
        fd=(isolated(iq+step*direction,iv)-isolated(iq-step*direction,iv))/(2*step)
        fd_errors.append(float(jnp.abs(fd-derivative)/jnp.maximum(1.,jnp.abs(derivative))))
    assert min(fd_errors)<1e-6
    result['internal_directional_fd_relative_errors']=fd_errors
    # Cross terms exchange hub and rotor momentum; no artificial independent blocks.
    hess=jax.hessian(isolated,1)(iq,iv)
    result['hub_rotor_hessian_cross_SI']=float(hess[3,8])
    assert abs(float(hess[3,8]))>1e6
    # Mixed position-attitude derivatives must be reciprocal and nonzero.
    tidal=lambda z: p.joint_action_si(z,jnp.zeros(9),gravity=False,radiation=False,em=False,tidal=True)-p.joint_action_si(z,jnp.zeros(9),gravity=False,radiation=False,em=False,tidal=False)
    cross=jax.hessian(tidal)(siq)
    result['tidal_position_attitude_cross_max']=float(jnp.max(jnp.abs(cross[:3,3:6])))
    assert result['tidal_position_attitude_cross_max']>0
    result['tidal_mixed_derivative_reciprocity']=float(jnp.max(jnp.abs(cross[:3,3:6]-cross[3:6,:3].T)))
    distances=jnp.linalg.norm(p.STARS-siq[:3],axis=1)
    directions=((p.STARS-siq[:3])/distances[:,None])@p.rotation(siq[3:6])
    I=p.inertia_total(siq)
    printed_torque=jnp.sum(3*p.GM[:,None]/distances[:,None]**3*jnp.cross(directions,directions@I),axis=0)
    mapping=jax.jacfwd(partial(p.body_omega,siq[3:6]))(jnp.zeros(3))
    # tidal is -U, so its coordinate derivative is the generalized torque.
    automatic_torque=jnp.linalg.solve(mapping.T,jax.grad(tidal)(siq)[3:6])
    torque_error=float(jnp.max(jnp.abs(printed_torque-automatic_torque)))
    assert torque_error<1e-16
    result['tidal_potential_printed_torque_max_error_Nm']=torque_error
    # Single-action energy identity and isolated total angular momentum, internal limit.
    a=acceleration(lambda z,w:isolated(jnp.concatenate([iq[:3],z]),jnp.concatenate([jnp.zeros(3),w])),iq[3:],iv[3:],jnp.zeros(6))
    e=lambda z:energy(lambda x,w:isolated(jnp.concatenate([iq[:3],x]),jnp.concatenate([jnp.zeros(3),w])),z[:6],z[6:])
    zdot=jnp.concatenate([iv[3:],a]);z=jnp.concatenate([iq[3:],iv[3:]])
    result['isolated_energy_derivative_W']=float(jax.grad(e)(z)@zdot)
    assert abs(result['isolated_energy_derivative_W'])<1e-7
    def total_angular(z):
        x=jnp.concatenate([iq[:3],z[:6]]);w=jnp.concatenate([jnp.zeros(3),z[6:]])
        rho,rho_dot=jax.jvp(p.centered_locations,(x[6:8],x[8]),(w[6:8],w[8]))
        omega=p.body_omega(x[3:6],w[3:6])
        H=p.HUB_INERTIA@omega+jnp.sum(p.MASSES[:,None]*jnp.cross(rho,jnp.cross(omega,rho)+rho_dot),axis=0)
        return p.rotation(x[3:6])@H
    derivativeH=jax.jvp(total_angular,(z,),(zdot,))[1]
    result['isolated_total_angular_momentum_derivative']=scalar(derivativeH)
    assert float(jnp.max(jnp.abs(derivativeH)))<1e-6
    def hub_angular(z):
        return p.rotation(z[:3])@p.HUB_INERTIA@p.body_omega(z[:3],z[6:9])
    hub_exchange=jax.jvp(hub_angular,(z,),(zdot,))[1]
    assert float(jnp.linalg.norm(hub_exchange))>1.
    result['isolated_hub_momentum_exchange_Nm']=scalar(hub_exchange)
    result['isolated_rotor_momentum_exchange_Nm']=scalar(derivativeH-hub_exchange)
    # All nine accelerations must survive inversion. Source pinv fails this test.
    jq,jv,jf,jlabels=jax.jit(partial(p.sample_joint,count=16,eq=eq,stratified=True))(jax.random.PRNGKey(1000))
    diag=diagnostics(p.joint_action,jq[0],jv[0]);result['joint_exact_hessian']=scalar(diag)
    generic_inverse=jax.jit(jax.vmap(partial(acceleration,p.joint_action)))(jq,jv,jf)
    result['source_pinv_internal_max_disagreement']=float(jnp.max(jnp.abs(generic_inverse[:,3:]-jlabels[:,3:])))
    normalized_disagreement=float(jnp.max(jnp.abs(generic_inverse-jlabels)/jnp.maximum(jnp.abs(jlabels),1e-15)))
    assert normalized_disagreement<1e-4,normalized_disagreement
    result['balanced_joint_pinv_scaled_max_disagreement']=normalized_disagreement
    old_q=jq*p.JOINT_SCALES/p.UNBALANCED_JOINT_SCALES
    old_v=jv*p.JOINT_SCALES/p.UNBALANCED_JOINT_SCALES
    old_f=jf*p.UNBALANCED_JOINT_SCALES/p.JOINT_SCALES
    result['unbalanced_exact_hessian']=scalar(diagnostics(p.unbalanced_joint_action,old_q[0],old_v[0]))
    result['coordinate_scaling_correction']='nominal mass-metric coordinate scales applied to the whole action, preserving coupling strengths'
    result['coordinate_scales_SI']=scalar(p.JOINT_SCALES)
    assert np.all(np.isfinite(jlabels))
    result['full_training_gates']={
        'proper_time_and_field_action':'unresolved: PDF does not define a covariant flexible-body action',
        'finite_hub_vs_PDF_limit':'anchored-hub limit passes; suppressing angular velocity alone is insufficient for PDF equations',
        'gravity_gradient_torque_sign':'passed: PDF potential and printed torque agree under virtual work',
        'default_pinv_retains_nine_modes':int(diag['retained_rank'])==9,
        'attitude_chart':'bounded ZYX chart only, no wide-rotation claims',
        'material_elastic_bound':'Young modulus alone does not establish cable yield/allowable stress'}
    result['published_benchmark_domain']={
        'speed_max_c':.99999,'attitude_each_rad':[-np.pi/6,np.pi/6],
        'hub_angle_rates_rad_s':[-.02,.02],'extension_offsets_m':[-.03,.03],
        'extension_rates_m_s':[-.05,.05],'spin_factor':[.9,1.1],
        'route_m':[0,p.DISTANCE],'transverse_position_m':[-p.AU,p.AU],
        'star_exclusion_m':p.AU,'position_mixture':[.5,.25,.25],
        'force_mixture':{'synthetic_vector_normal_sigma10':.5,'zero_external_force':.25,'mission_g1_axial':.25},
        'near_star_distances_m':[p.AU,p.DISTANCE/2],
        'interpretation':'provisional effective benchmark, not certified hardware envelope'}
    result['incomplete_reference_gate_passed']=True
    (out/'physics_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='conservative_training/audit')
    run_audit(parser.parse_args().out)
