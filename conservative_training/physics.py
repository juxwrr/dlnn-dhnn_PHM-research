"""Analytic labels and genuinely joint effective action (not a covariant ship).

The effective action uses SI coordinate time for all terms, transformed as a
whole. It is only a mathematical benchmark at relativistic COM speeds.
"""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root

from .engine import acceleration

C = 299792458.0
AU = 149597870700.0
G0 = 9.80665
GM = jnp.array([1.3271244e20, 0.783 * 1.3271244e20])
MASS = 100000.0
MASSES = jnp.array([50000., 20000., 30000.])
RADII = jnp.array([30., 20.])
HUB_INERTIA = jnp.diag(jnp.array([100000., 9254166.666666666, 9254166.666666666]))
STIFFNESS = 4 * 120e9 * (np.pi * .01**2 / 4) / RADII
DISTANCE = 11.9 * C * 365.25 * 86400
STARS = jnp.array([[-AU, 0., 0.], [DISTANCE + AU, 0., 0.]])
K = jnp.array([3.828e26, .52 * 3.828e26]) * (np.pi * 2**2) / (4 * np.pi * C)
CHARGE = 4 * np.pi * 8.8541878188e-12 * 10 * 20
E_FIELD = jnp.array([1e-5, 0., 0.])
B_FIELD = 5e-10 * jnp.array([np.cos(np.pi/6), np.sin(np.pi/6), 0.])
DIPOLE_BODY = jnp.array([100., 0., 0.])

# Mission scales for incomplete model; force g=1 retains the workbook's 15 m/s².
A = 15.
INC_TIME, INC_LENGTH = C / A, C**2 / A
Q_ROUTE = DISTANCE / INC_LENGTH
# One common coordinate time, all energies divided by MASS*C², no per-term factors.
JOINT_TIME = float(jnp.sqrt(MASSES[1] / STIFFNESS[0]))
UNBALANCED_JOINT_SCALES = jnp.array([C * JOINT_TIME]*3 + [1.]*3 + [.2, .2, 1.])
# Balance coordinate scales by the nominal mass metric. This transforms the
# entire action; no subsystem energy gets an independent multiplier.
NOMINAL_ROTOR_INERTIA = 20000*30**2 + 30000*20**2
NOMINAL_INERTIA = jnp.array([100000+NOMINAL_ROTOR_INERTIA,9254166.666666666,
                            9254166.666666666+NOMINAL_ROTOR_INERTIA])
JOINT_SCALES = jnp.concatenate([jnp.full(3,C*JOINT_TIME),
    C*JOINT_TIME*jnp.sqrt(MASS/NOMINAL_INERTIA),
    C*JOINT_TIME*jnp.sqrt(MASS/MASSES[1:]),
    jnp.array([C*JOINT_TIME*jnp.sqrt(MASS/NOMINAL_ROTOR_INERTIA)])])
ENERGY_SCALE = MASS * C**2


def incomplete_action(q, velocity):
    return 1 - jnp.sqrt(1 - velocity[0]**2)


def incomplete_labels(q, velocity, force):
    return force * (1 - velocity**2)**1.5


def constant_force_solution(t, force=1., q0=0., beta0=0.):
    momentum = beta0 / jnp.sqrt(1 - beta0**2)
    p = momentum + force * t
    beta = p / jnp.sqrt(1 + p**2)
    # Rationalization removes cancellation and handles g=0 continuously.
    position = q0 + t * (p + momentum) / (jnp.sqrt(1 + p**2) + jnp.sqrt(1 + momentum**2))
    return jnp.stack([position, beta], axis=-1)


def rotation(angles):
    """R=Rz(yaw) Ry(pitch) Rx(roll), chart limited to |angles| <= pi/6."""
    roll, pitch, yaw = angles
    cr, sr, cp, sp, cy, sy = jnp.cos(roll), jnp.sin(roll), jnp.cos(pitch), jnp.sin(pitch), jnp.cos(yaw), jnp.sin(yaw)
    rx = jnp.array([[1.,0.,0.],[0.,cr,-sr],[0.,sr,cr]])
    ry = jnp.array([[cp,0.,sp],[0.,1.,0.],[-sp,0.,cp]])
    rz = jnp.array([[cy,-sy,0.],[sy,cy,0.],[0.,0.,1.]])
    return rz @ ry @ rx


def body_omega(angles, angle_rates):
    roll, pitch, _ = angles
    mapping = jnp.array([[1.,0.,-jnp.sin(pitch)],
                         [0.,jnp.cos(roll),jnp.sin(roll)*jnp.cos(pitch)],
                         [0.,-jnp.sin(roll),jnp.cos(roll)*jnp.cos(pitch)]])
    return mapping @ angle_rates


def centered_locations(extension, theta):
    # Rotor plane is transverse to the cylinder axis (body x).
    direction = jnp.array([0., jnp.cos(theta), jnp.sin(theta)])
    raw = jnp.stack([jnp.zeros(3), (RADII[0]+extension[0])*direction,
                     -(RADII[1]+extension[1])*direction])
    center = jnp.sum(MASSES[:,None]*raw, axis=0)/MASS
    return raw - center


def internal_kinetic(q, velocity, recenter=True):
    angles, extension, theta = q[3:6], q[6:8], q[8]
    omega = body_omega(angles, velocity[3:6])
    internal = jnp.concatenate([extension, theta[None]])
    rates = velocity[6:9]
    def locations(z):
        if recenter:
            return centered_locations(z[:2], z[2])
        direction = jnp.array([0., jnp.cos(z[2]), jnp.sin(z[2])])
        return jnp.stack([jnp.zeros(3), (RADII[0]+z[0])*direction, -(RADII[1]+z[1])*direction])
    rho, rho_dot = jax.jvp(locations, (internal,), (rates,))
    point_velocity = jnp.cross(omega, rho) + rho_dot
    return .5 * omega @ HUB_INERTIA @ omega + .5*jnp.sum(MASSES[:,None]*point_velocity**2)


def inertia_total(q):
    rho = centered_locations(q[6:8], q[8])
    return HUB_INERTIA + jnp.sum(MASSES[:,None,None] *
        (jnp.sum(rho**2,axis=1)[:,None,None]*jnp.eye(3) - rho[:,:,None]*rho[:,None,:]), axis=0)


def internal_point_accelerations_si(q, velocity, acceleration):
    """Point accelerations relative to COM, including hub motion and Coriolis terms."""
    omega, omega_dot = jax.jvp(body_omega, (q[3:6], velocity[3:6]),
                              (velocity[3:6], acceleration[3:6]))
    z, rates, accelerations = q[6:9], velocity[6:9], acceleration[6:9]
    locations = lambda z: centered_locations(z[:2], z[2])
    rho, rho_dot = jax.jvp(locations, (z,), (rates,))
    # Time derivative of J(z)*rates includes both changing geometry and acceleration.
    tangent = lambda z, rates: jax.jvp(locations, (z,), (rates,))[1]
    _, rho_ddot = jax.jvp(tangent, (z, rates), (rates, accelerations))
    return (rho_ddot + 2*jnp.cross(omega, rho_dot) + jnp.cross(omega_dot, rho)
            + jnp.cross(omega, jnp.cross(omega, rho)))


def joint_action_si(q, velocity, gravity=True, radiation=True, em=True, tidal=True, recenter=True):
    r, v = q[:3], velocity[:3]
    # Stable rest-offset kinetic expression, equivalent up to an additive constant.
    b2 = jnp.dot(v,v)/C**2
    translation = MASS*C**2*b2/(1+jnp.sqrt(1-b2))
    displacements = STARS-r
    distances = jnp.linalg.norm(displacements,axis=1)
    action = translation + internal_kinetic(q,velocity,recenter) - .5*jnp.dot(STIFFNESS,q[6:8]**2)
    if gravity: action += MASS*jnp.sum(GM/distances)
    if radiation: action -= jnp.sum(K/distances)
    R = rotation(q[3:6])
    if em:
        vector_potential = .5*jnp.cross(B_FIELD,r)
        action += CHARGE*(jnp.dot(vector_potential,v)+jnp.dot(E_FIELD,r)) + jnp.dot(R@DIPOLE_BODY,B_FIELD)
    if tidal:
        I = inertia_total(q)
        body_direction = (displacements/distances[:,None]) @ R
        # PDF sign retained: virtual work gives +3*GM/r³ n_body x (I n_body).
        U = -.5*jnp.sum(GM/distances**3 * (jnp.trace(I)-3*jnp.einsum('ni,ij,nj->n',body_direction,I,body_direction)))
        action -= U
    return action


def joint_action(q, velocity, **kwargs):
    return joint_action_si(q*JOINT_SCALES, velocity*JOINT_SCALES/JOINT_TIME, **kwargs)/ENERGY_SCALE


def unbalanced_joint_action(q,velocity):
    return joint_action_si(q*UNBALANCED_JOINT_SCALES,velocity*UNBALANCED_JOINT_SCALES/JOINT_TIME)/ENERGY_SCALE


def joint_reference(q, velocity, force, **kwargs):
    """Preconditioned exact solve; generic neural model keeps the source pinv.

    SI mass balancing here is an algebraic solve of the same action, not a
    change of physical energy weights. Default source pinv loses internal modes.
    """
    action = partial(joint_action, **kwargs)
    hessian = jax.hessian(action,1)(q,velocity)
    rhs = jax.grad(action,0)(q,velocity)+force-jax.jacfwd(jax.grad(action,1),0)(q,velocity)@velocity
    d = jnp.sqrt(jnp.abs(jnp.diag(hessian)))
    return jnp.linalg.solve(hessian/(d[:,None]*d[None,:]),rhs/d)/d


def force_joint(g):
    force_si = jnp.concatenate([MASS*A*g,jnp.zeros(6)])
    return force_si*JOINT_SCALES/ENERGY_SCALE


def equilibrium():
    """Recentered equilibrium with hub held at zero angular rate, 1g at habitat."""
    def equations(z):
        h,c,spin=z
        radii=np.array(RADII)+[h,c]
        offset=(20000*radii[0]-30000*radii[1])/MASS
        centered=np.array([radii[0]-offset, -radii[1]-offset])
        return np.array([float(STIFFNESS[0])*h-20000*centered[0]*spin**2,
                         float(STIFFNESS[1])*c+30000*centered[1]*spin**2,
                         centered[0]*spin**2-G0]) / np.array([200000.,200000.,G0])
    answer=root(equations,[.156,.104,.57026])
    if not answer.success: raise RuntimeError(answer.message)
    return answer.x


def sample_incomplete(key, count, stratified=False, low_speed=False):
    kp,kv,kf = jax.random.split(key,3)
    q = jax.random.uniform(kp,(count,1),minval=0,maxval=Q_ROUTE)
    velocity=.99999*jnp.tanh(2*jax.random.normal(kv,(count,1)))
    if stratified:
        bounds=jnp.array([0.,.7,.9,.99,.99999])
        bands=jnp.arange(count)%4
        u=jax.random.uniform(kv,(count,1))
        speed=bounds[bands,None]+u*(bounds[bands+1,None]-bounds[bands,None])
        signs=jnp.where(jax.random.bernoulli(jax.random.fold_in(kv,1),shape=(count,1)),1.,-1.)
        velocity=speed*signs
    if low_speed:
        # Exact rejection sampling of the declared distribution conditioned on |beta|<.7.
        def replace(carry):
            iteration,v=carry
            z=.99999*jnp.tanh(2*jax.random.normal(jax.random.fold_in(kv,iteration+2),(count,1)))
            return iteration+1,jnp.where(jnp.abs(v)>=.7,z,v)
        _,velocity=jax.lax.while_loop(lambda carry:jnp.any(jnp.abs(carry[1])>=.7),replace,(0,velocity))
    g=10*jax.random.normal(kf,(count,1))
    return q,velocity,g,incomplete_labels(q,velocity,g)


def sample_joint(key,count,eq,stratified=False,low_speed=False):
    keys=jax.random.split(key,10)
    x,w,g,_=sample_incomplete(keys[0],count,stratified,low_speed)
    choose=jax.random.uniform(keys[1],(count,1))
    near=jnp.exp(jax.random.uniform(keys[2],(count,1),minval=jnp.log(AU),maxval=jnp.log(DISTANCE/2)))
    route=x[:,0:1]*INC_LENGTH
    route=jnp.where(choose<.25,near-AU,jnp.where(choose<.5,DISTANCE+AU-near,route))
    r=jnp.concatenate([route,jax.random.uniform(keys[3],(count,2),minval=-AU,maxval=AU)],axis=1)
    direction=jax.random.normal(keys[4],(count,3));direction/=jnp.linalg.norm(direction,axis=1,keepdims=True)
    v=jnp.abs(w)*C*direction
    angles=jax.random.uniform(keys[5],(count,3),minval=-np.pi/6,maxval=np.pi/6)
    angle_rates=jax.random.uniform(keys[6],(count,3),minval=-.02,maxval=.02)
    ext=jnp.array(eq[:2])+jax.random.uniform(keys[7],(count,2),minval=-.03,maxval=.03)
    radial=jax.random.uniform(keys[8],(count,2),minval=-.05,maxval=.05)
    spin=jax.random.uniform(keys[9],(count,1),minval=.9*eq[2],maxval=1.1*eq[2])
    theta=jax.random.uniform(jax.random.fold_in(keys[9],1),(count,1),minval=-np.pi,maxval=np.pi)
    q=jnp.concatenate([r,angles,ext,theta],axis=1)/JOINT_SCALES
    velocity=jnp.concatenate([v,angle_rates,radial,spin],axis=1)*JOINT_TIME/JOINT_SCALES
    # Independent force and velocity streams. Quiet/mission strata expose internal signals.
    kg,ks=jax.random.split(jax.random.fold_in(key,23))
    vector_g=10*jax.random.normal(kg,(count,3))
    stratum=jax.random.uniform(ks,(count,1))
    vector_g=jnp.where(stratum<.25,0.,jnp.where(stratum<.5,jnp.array([1.,0.,0.]),vector_g))
    forces=jax.vmap(force_joint)(vector_g)
    labels=jax.vmap(joint_reference)(q,velocity,forces)
    return q,velocity,forces,labels
