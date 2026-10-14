"""Deterministic Project Hail Mary physics model.

The mission translation state is integrated in crew proper time tau.  Internal
centrifuge and attitude dynamics are also evaluated in the instantaneous
spacecraft rest frame.  External stellar fields are evaluated in the inertial
source/target frame and mapped to the 1-D proper-force equation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Callable, Dict, Tuple
import math
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Parameters:
    c: float = 299_792_458.0
    G: float = 6.67430e-11
    ly: float = 9.4607304725808e15
    year: float = 365.25 * 86400.0
    au: float = 149_597_870_700.0
    distance_ly: float = 11.9
    dry_mass: float = 100_000.0
    proper_accel: float = 15.0
    propulsion_efficiency: float = 0.8031
    collimation_efficiency: float = 0.50
    astrophage_density: float = 5078.0
    area: float = math.pi * 2.0**2
    source_mass: float = 1.989e30
    target_mass: float = 1.56e30
    source_luminosity: float = 3.86e26
    target_luminosity: float = 2.0e26
    radiation_coefficient: float = 1.5
    source_offset: float = 149_597_870_700.0
    target_offset: float = 149_597_870_700.0
    gas_density: float = 1.67e-21
    dust_density: float = 1.0e-23
    gas_drag_coefficient: float = 1.0
    dust_drag_coefficient: float = 1.0
    cable_diameter: float = 0.010
    cable_count_h: int = 6
    cable_count_c: int = 6
    cable_modulus: float = 270e9
    radius_h: float = 73.0
    radius_c: float = 73.0 / 3.0
    mass_h: float = 25_000.0
    mass_c: float = 75_000.0
    spin_rate: float = math.sqrt(9.8 / 73.0)
    tether_zeta: float = 0.05
    radial_coulomb: float = 100.0
    bearing_damping: float = 2.0e4
    bearing_coulomb: float = 500.0
    attitude_inertia_x: float = 8.0e7
    attitude_inertia_y: float = 1.2e8
    attitude_inertia_z: float = 1.4e8
    attitude_kp: float = 5.0e5
    attitude_kd: float = 2.0e6

    @property
    def distance(self) -> float:
        return self.distance_ly * self.ly

    @property
    def effective_propulsion(self) -> float:
        return self.propulsion_efficiency * self.collimation_efficiency

    @property
    def eta_half(self) -> float:
        return math.acosh(1.0 + self.proper_accel * self.distance / (2.0 * self.c**2))

    @property
    def required_mass_ratio(self) -> float:
        return math.exp(2.0 * self.eta_half / self.effective_propulsion)

    @property
    def nominal_initial_mass(self) -> float:
        return self.dry_mass * self.required_mass_ratio


def parameter_dict(p: Parameters) -> Dict[str, float]:
    d = asdict(p)
    d.update({
        "distance": p.distance,
        "effective_propulsion": p.effective_propulsion,
        "eta_half": p.eta_half,
        "required_mass_ratio": p.required_mass_ratio,
        "nominal_initial_mass": p.nominal_initial_mass,
    })
    return d


def beta_from_eta(eta):
    return np.tanh(eta)


def gamma_from_eta(eta):
    return np.cosh(eta)


def constant_proper_accel(tau, alpha, c):
    eta = alpha * tau / c
    return {
        "eta": eta,
        "gamma": np.cosh(eta),
        "beta": np.tanh(eta),
        "v": c * np.tanh(eta),
        "t": (c / alpha) * np.sinh(eta),
        "x": (c**2 / alpha) * (np.cosh(eta) - 1.0),
    }


def stellar_forces(x, m, p: Parameters):
    rs = np.maximum(p.source_offset + x, 1.0)
    rt = np.maximum(p.target_offset + p.distance - x, 1.0)
    fg_source = -p.G * p.source_mass * m / rs**2
    fg_target = p.G * p.target_mass * m / rt**2
    ks = p.radiation_coefficient * p.area * p.source_luminosity / (4.0 * math.pi * p.c)
    kt = p.radiation_coefficient * p.area * p.target_luminosity / (4.0 * math.pi * p.c)
    fr_source = ks / rs**2
    fr_target = -kt / rt**2
    return fg_source, fg_target, fr_source, fr_target


def drag_force(eta, p: Parameters, gas_scale=1.0, dust_scale=1.0):
    beta = np.tanh(eta)
    gamma = np.cosh(eta)
    v = p.c * beta
    coeff = (
        p.gas_drag_coefficient * p.gas_density * gas_scale
        + p.dust_drag_coefficient * p.dust_density * dust_scale
    ) * p.area
    return -coeff * gamma**2 * v * np.abs(v)


def drag_rayleigh(v, p: Parameters, gas_scale=1.0, dust_scale=1.0):
    beta = np.clip(np.abs(v) / p.c, 0.0, 1.0 - 1e-14)
    coeff = (
        p.gas_drag_coefficient * p.gas_density * gas_scale
        + p.dust_drag_coefficient * p.dust_density * dust_scale
    ) * p.area
    return coeff * p.c**3 * (np.arctanh(beta) - beta)


def mission_mass_ratio(distance, alpha, chi_epsilon, c):
    eta_h = np.arccosh(1.0 + alpha * distance / (2.0 * c**2))
    return np.exp(2.0 * eta_h / chi_epsilon)


def _thrust_policy(tau, x, eta, m, p: Parameters, scenario: str, initial_mass: float):
    if m <= p.dry_mass * (1.0 + 1e-10):
        return 0.0
    alpha = p.proper_accel
    if scenario == "lower_efficiency":
        alpha = p.proper_accel
    if scenario in {"interrupted", "worst_case"} and 0.30 * p.year <= tau <= 0.80 * p.year:
        return 0.0
    if scenario == "engine_power_limited":
        beam_power = 1.0e21
        tmag = min(m * alpha, p.collimation_efficiency * beam_power / p.c)
    elif scenario == "fixed_thrust":
        tmag = p.dry_mass * alpha
    else:
        tmag = m * alpha
    # Calculated braking condition: compare remaining distance with the local
    # constant-proper-acceleration stopping distance.
    local_alpha = max(tmag / m, 1e-30)
    stop_distance = p.c**2 / local_alpha * max(math.cosh(max(eta, 0.0)) - 1.0, 0.0)
    remaining = p.distance - x
    if eta < 0.0:
        return tmag
    return -tmag if remaining <= stop_distance else tmag


def simulate_translation(
    p: Parameters,
    scenario: str = "ideal",
    initial_mass: float | None = None,
    density_scale: float = 1.0,
    n_eval: int = 2400,
    rtol: float = 2e-9,
    max_step: float | None = None,
):
    p_eff = p
    if scenario in {"lower_efficiency", "worst_case"}:
        p_eff = replace(p, propulsion_efficiency=0.55)
    if initial_mass is None:
        initial_mass = p_eff.nominal_initial_mass
    if scenario == "low_fuel":
        initial_mass = p.dry_mass * math.sqrt(p.required_mass_ratio)
    max_tau = 40.0 * p.year
    if max_step is None:
        max_step = 10_000.0

    def thrust_magnitude(tau, m):
        if m <= p_eff.dry_mass * (1.0 + 1e-10):
            return 0.0
        if scenario in {"interrupted", "worst_case"} and 0.30 * p.year <= tau <= 0.80 * p.year:
            return 0.0
        if scenario == "engine_power_limited":
            return min(m * p_eff.proper_accel, p_eff.collimation_efficiency * 1.0e21 / p_eff.c)
        if scenario == "fixed_thrust":
            return initial_mass * p_eff.proper_accel
        return m * p_eff.proper_accel

    def rhs_for_phase(phase):
      def rhs(tau, y):
        t, x, eta, m, e_drag = y
        gamma = math.cosh(eta)
        beta = math.tanh(eta)
        v = p.c * beta
        thrust = phase * thrust_magnitude(tau, m)
        fg_s, fg_t, fr_s, fr_t = stellar_forces(x, m, p_eff)
        isolated = scenario in {"ideal", "constant_mass", "low_fuel", "fixed_thrust", "engine_power_limited", "lower_efficiency", "interrupted"}
        f_stellar = 0.0 if isolated else fg_s + fg_t + fr_s + fr_t
        f_drag = 0.0 if isolated or scenario == "gravity_radiation" else drag_force(eta, p_eff, density_scale, density_scale)
        dm = 0.0 if scenario == "constant_mass" else -abs(thrust) / (p_eff.effective_propulsion * p.c)
        if m <= p_eff.dry_mass and dm < 0.0:
            dm = 0.0
            thrust = 0.0
        deta = (thrust + f_stellar + f_drag) / (max(m, p_eff.dry_mass) * p.c)
        # dE_loss/dtau = |F v| dt/dtau.
        return [gamma, p.c * math.sinh(eta), deta, dm, abs(f_drag * v) * gamma]
      return rhs

    rhs_accel = rhs_for_phase(+1.0)
    rhs_brake = rhs_for_phase(-1.0)

    def switch_event(tau, y):
        _, x, eta, m, _ = y
        tmag = thrust_magnitude(tau, m)
        if tmag <= 0.0:
            return p.distance - x
        local_alpha = tmag / max(m, p_eff.dry_mass)
        stop_distance = p.c**2 / local_alpha * max(math.cosh(max(eta, 0.0)) - 1.0, 0.0)
        return (p.distance - x) - stop_distance
    switch_event.terminal = True
    switch_event.direction = -1

    def arrival(tau, y):
        return y[1] - p.distance
    arrival.terminal = True
    arrival.direction = 1

    def stopped(tau, y):
        return y[2]
    stopped.terminal = True
    stopped.direction = -1

    sol1 = solve_ivp(
        rhs_accel,
        (0.0, max_tau),
        [0.0, 0.0, 0.0, initial_mass, 0.0],
        events=(switch_event, arrival),
        rtol=rtol,
        atol=[1e-5, 10.0, 1e-11, 1e-3, 1e4],
        max_step=max_step,
        dense_output=True,
    )
    segments = [sol1]
    switch_tau = sol1.t_events[0][0] if len(sol1.t_events[0]) else None
    arrived_in_accel = bool(len(sol1.t_events[1]))
    if switch_tau is not None and not arrived_in_accel:
        y_switch = sol1.y_events[0][0]
        sol2 = solve_ivp(
            rhs_brake,
            (switch_tau, max_tau),
            y_switch,
            events=(arrival, stopped),
            rtol=rtol,
            atol=[1e-5, 10.0, 1e-11, 1e-3, 1e4],
            max_step=max_step,
            dense_output=True,
        )
        segments.append(sol2)
    stopped_near_target = False
    if len(segments) == 2 and len(segments[1].t_events[1]):
        stop_state = segments[1].y_events[1][0]
        stopped_near_target = abs(stop_state[1] - p.distance) <= 1e-6 * p.distance
    arrived = arrived_in_accel or (len(segments) == 2 and bool(len(segments[1].t_events[0]))) or stopped_near_target
    if arrived_in_accel:
        tau_end = sol1.t_events[1][0]
    elif len(segments) == 2 and len(segments[1].t_events[0]):
        tau_end = segments[1].t_events[0][0]
    elif stopped_near_target:
        tau_end = segments[1].t_events[1][0]
    else:
        tau_end = segments[-1].t[-1]
    tau = np.linspace(0.0, tau_end, n_eval)
    if len(segments) == 1:
        y = sol1.sol(tau)
    else:
        mask = tau <= switch_tau
        y = np.empty((5, len(tau)))
        y[:, mask] = sol1.sol(tau[mask])
        if np.any(~mask):
            y[:, ~mask] = segments[1].sol(tau[~mask])
    t, x, eta, mass, e_drag = y
    gamma = np.cosh(eta)
    beta = np.tanh(eta)
    v = p.c * beta
    forces = {k: np.empty_like(tau) for k in ["thrust", "gravity_source", "gravity_target", "radiation_source", "radiation_target", "drag"]}
    for i in range(len(tau)):
        phase = 1.0 if switch_tau is None or tau[i] <= switch_tau else -1.0
        thrust = phase * thrust_magnitude(tau[i], mass[i])
        fg_s, fg_t, fr_s, fr_t = stellar_forces(x[i], mass[i], p_eff)
        fd = drag_force(eta[i], p_eff, density_scale, density_scale)
        isolated = scenario in {"ideal", "constant_mass", "low_fuel", "fixed_thrust", "engine_power_limited", "lower_efficiency", "interrupted"}
        if isolated:
            fg_s = fg_t = fr_s = fr_t = 0.0
        if isolated or scenario == "gravity_radiation":
            fd = 0.0
        forces["thrust"][i] = thrust
        forces["gravity_source"][i] = fg_s
        forces["gravity_target"][i] = fg_t
        forces["radiation_source"][i] = fr_s
        forces["radiation_target"][i] = fr_t
        forces["drag"][i] = fd
    return {
        "tau": tau, "t": t, "x": x, "eta": eta, "mass": mass,
        "gamma": gamma, "beta": beta, "v": v, "drag_energy": e_drag,
        "forces": forces, "arrived": arrived, "scenario": scenario,
        "initial_mass": initial_mass, "parameters": p_eff,
        "solver_nfev": sum(s.nfev for s in segments), "switch_tau": switch_tau,
    }


def cable_properties(p: Parameters, side: str):
    if side == "h":
        mass, radius, count = p.mass_h, p.radius_h, p.cable_count_h
    else:
        mass, radius, count = p.mass_c, p.radius_c, p.cable_count_c
    area_one = math.pi * p.cable_diameter**2 / 4.0
    k = count * p.cable_modulus * area_one / radius
    damping = 2.0 * p.tether_zeta * math.sqrt(k * mass)
    return mass, radius, k, damping


def equilibrium_extension(mass, radius, k, omega):
    denom = k - mass * omega**2
    return np.nan if denom <= 0 else mass * radius * omega**2 / denom


def simulate_centrifuge(p: Parameters, duration=300.0, impact_time=120.0, radial_impulse=5000.0):
    mh, rh, kh, ch = cable_properties(p, "h")
    mc, rc, kc, cc = cable_properties(p, "c")
    xh0 = equilibrium_extension(mh, rh, kh, p.spin_rate)
    xc0 = equilibrium_extension(mc, rc, kc, p.spin_rate)
    y0 = np.array([xh0, 0.0, xc0, 0.0, 0.0, p.spin_rate])

    def rhs(tau, y):
        xh, vh, xc, vc, theta, omega = y
        def sgn_smooth(v): return math.tanh(v / 1e-4)
        ah = ((mh * (rh + xh) * omega**2) - ch * vh - kh * xh - p.radial_coulomb * sgn_smooth(vh)) / mh
        ac = ((mc * (rc + xc) * omega**2) - cc * vc - kc * xc - p.radial_coulomb * sgn_smooth(vc)) / mc
        inertia = mh * (rh + xh)**2 + mc * (rc + xc)**2
        idot = 2 * mh * (rh + xh) * vh + 2 * mc * (rc + xc) * vc
        motor = 2.0e6 * (p.spin_rate - omega)
        domega = (motor - p.bearing_damping * omega - p.bearing_coulomb * sgn_smooth(omega) - idot * omega) / inertia
        return [vh, ah, vc, ac, omega, domega]

    segments = [(0.0, impact_time), (impact_time, duration)]
    ts, ys = [], []
    state = y0.copy()
    for i, (a, b) in enumerate(segments):
        ev = solve_ivp(rhs, (a, b), state, rtol=1e-9, atol=1e-11, max_step=0.03, dense_output=True)
        tt = np.linspace(a, b, int((b-a)*10)+1)
        yy = ev.sol(tt)
        if i:
            tt, yy = tt[1:], yy[:, 1:]
        ts.append(tt); ys.append(yy)
        state = ev.y[:, -1]
        if i == 0:
            state[1] += radial_impulse / mh
            state[3] -= radial_impulse / mc
    tau = np.concatenate(ts); y = np.concatenate(ys, axis=1)
    xh, vh, xc, vc, theta, omega = y
    tension_h = kh * xh
    tension_c = kc * xc
    gravity_h = omega**2 * (rh + xh)
    gravity_c = omega**2 * (rc + xc)
    damping_loss = ch * vh**2 + cc * vc**2 + p.bearing_damping * omega**2
    motor_torque = 2.0e6 * (p.spin_rate - omega)
    return locals()


def quat_rhs(q, omega):
    q0, q1, q2, q3 = q
    wx, wy, wz = omega
    return 0.5 * np.array([
        -q1*wx-q2*wy-q3*wz,
        q0*wx+q2*wz-q3*wy,
        q0*wy-q1*wz+q3*wx,
        q0*wz+q1*wy-q2*wx,
    ])


def simulate_attitude(p: Parameters, duration=600.0, impact_time=100.0):
    inertia = np.diag([p.attitude_inertia_x, p.attitude_inertia_y, p.attitude_inertia_z])
    inv_inertia = np.linalg.inv(inertia)
    mu = p.G * p.source_mass
    r = p.au
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def torque_parts(y):
        q = y[:4] / np.linalg.norm(y[:4]); w = y[4:]
        rhat = np.array([1.0, 0.2, 0.1])
        rhat /= np.linalg.norm(rhat)
        tgg = 3.0 * mu / r**3 * np.cross(rhat, inertia @ rhat)
        tctrl = -p.attitude_kp * q[1:] - p.attitude_kd * w
        tem = np.zeros(3)
        return tctrl, tgg, tem

    def rhs(tau, y):
        q = y[:4] / np.linalg.norm(y[:4]); w = y[4:]
        tctrl, tgg, tem = torque_parts(y)
        wdot = inv_inertia @ (tctrl + tgg + tem - np.cross(w, inertia @ w))
        return np.r_[quat_rhs(q, w), wdot]

    segments = [(0.0, impact_time), (impact_time, duration)]
    ts, ys = [], []
    state = y0.copy()
    for i, (a, b) in enumerate(segments):
        sol = solve_ivp(rhs, (a, b), state, rtol=1e-10, atol=1e-12, max_step=0.2, dense_output=True)
        tt = np.linspace(a, b, int((b-a)*5)+1); yy = sol.sol(tt)
        norms = np.linalg.norm(yy[:4], axis=0); yy[:4] /= norms
        if i:
            tt, yy = tt[1:], yy[:, 1:]
        ts.append(tt); ys.append(yy)
        state = sol.y[:, -1]; state[:4] /= np.linalg.norm(state[:4])
        if i == 0:
            angular_impulse = np.array([0.0, 2.0e6, -1.0e6])
            state[4:] += inv_inertia @ angular_impulse
    tau = np.concatenate(ts); y = np.concatenate(ys, axis=1)
    q = y[:4]; omega = y[4:]
    error_angle = 2.0 * np.arccos(np.clip(np.abs(q[0]), 0.0, 1.0))
    torques = np.array([np.r_[torque_parts(y[:, i])] for i in range(y.shape[1])]).reshape(-1, 3, 3)
    return {"tau": tau, "q": q, "omega": omega, "error_angle": error_angle, "torques": torques, "inertia": inertia}


def corrected_beetle(p: Parameters):
    dry, fuel, thrust = 5.0, 180.0, 930_000.0
    m0, mf = dry + fuel, dry
    eta_budget = p.effective_propulsion * math.log(m0 / mf)
    eta_peak = eta_budget / 2.0
    beta_peak = math.tanh(eta_peak)
    gamma_peak = math.cosh(eta_peak)
    m_switch = math.sqrt(m0 * mf)
    burn_rate = thrust / (p.effective_propulsion * p.c)
    tau_accel = (m0 - m_switch) / burn_rate
    tau_decel = (m_switch - mf) / burn_rate

    def leg_distance(ma, mb, sign=1.0):
        # Numeric quadrature in mass using eta=chi*epsilon*ln(ma/m).
        ms = np.geomspace(ma, mb, 4000)
        eta = p.effective_propulsion * np.log(ma / ms)
        dtaudm = -1.0 / burn_rate
        return abs(np.trapezoid(p.c * np.sinh(eta) * dtaudm, ms))
    d_acc = leg_distance(m0, m_switch)
    ms_acc = np.geomspace(m0, m_switch, 4000)
    eta_acc = p.effective_propulsion * np.log(m0 / ms_acc)
    t_acc = abs(np.trapezoid(np.cosh(eta_acc) * (-1.0 / burn_rate), ms_acc))
    # Deceleration starts at eta_peak; eta(m)=eta_peak-chi*epsilon*ln(m_switch/m).
    ms = np.geomspace(m_switch, mf, 4000)
    eta_dec = eta_peak - p.effective_propulsion * np.log(m_switch / ms)
    d_dec = abs(np.trapezoid(p.c * np.sinh(eta_dec) * (-1.0/burn_rate), ms))
    t_dec = abs(np.trapezoid(np.cosh(eta_dec) * (-1.0/burn_rate), ms))
    coast_distance = max(p.distance - d_acc - d_dec, 0.0)
    coast_earth = coast_distance / (p.c * beta_peak)
    coast_proper = coast_earth / gamma_peak
    return {
        "dry_mass": dry, "fuel_mass": fuel, "start_mass": m0, "thrust": thrust,
        "mass_switch": m_switch, "eta_budget": eta_budget, "eta_peak": eta_peak,
        "beta_peak": beta_peak, "gamma_peak": gamma_peak, "burn_rate": burn_rate,
        "tau_accel": tau_accel, "tau_decel": tau_decel, "distance_accel": d_acc,
        "distance_decel": d_dec, "coast_distance": coast_distance,
        "coast_earth": coast_earth, "coast_proper": coast_proper,
        "coordinate_accel": t_acc, "coordinate_decel": t_dec,
        "trip_earth": coast_earth + t_acc + t_dec,
        "trip_proper": coast_proper + tau_accel + tau_decel,
    }


def corrected_navigation(p: Parameters):
    v0 = 11_872_000.0
    distance = 4.5e12
    beta0 = v0 / p.c
    gamma0 = 1.0 / math.sqrt(1.0 - beta0**2)
    eta0 = math.atanh(beta0)
    alpha_required = (gamma0 - 1.0) * p.c**2 / distance
    t_stop = gamma0 * v0 / alpha_required
    tau_stop = eta0 * p.c / alpha_required
    elapsed = 4.3 * 86400.0
    sinh_eta_now = max(math.sinh(eta0) - alpha_required * elapsed / p.c, 0.0)
    eta_now = math.asinh(sinh_eta_now)
    t_now = elapsed
    x_closed = p.c**2 / alpha_required * (math.cosh(eta0) - math.cosh(eta_now))
    return {
        "initial_velocity": v0, "distance": distance, "required_proper_decel": alpha_required,
        "earth_stop_time": t_stop, "proper_stop_time": tau_stop, "elapsed_coordinate_time": elapsed,
        "current_velocity": p.c * math.tanh(eta_now), "distance_remaining": distance - x_closed,
        "earth_time_remaining": max(t_stop - t_now, 0.0), "eta_initial": eta0,
    }


def simulate_full(p: Parameters, scenario="nominal", density_scale=1.0, n_eval=2400):
    """Partitioned multirate propagation of the complete 17-component state.

    Translation evolves across mission years.  The stiff tether and attitude
    equations are solved on their natural sub-second/second grids and embedded
    as a discrete mid-mission disturbance window.  This retains the declared
    state topology without forcing a numerically wasteful monolithic step size.
    """
    tr = simulate_translation(p, scenario, density_scale=density_scale, n_eval=n_eval)
    cent = simulate_centrifuge(p)
    att = simulate_attitude(p)
    tau = tr["tau"]
    n = len(tau)
    state = np.zeros((n, 17))
    state[:, 0] = tr["t"]; state[:, 1] = tr["x"]; state[:, 2] = tr["eta"]; state[:, 3] = tr["mass"]
    mh, rh, kh, _ = cable_properties(p, "h"); mc, rc, kc, _ = cable_properties(p, "c")
    state[:, 4] = equilibrium_extension(mh, rh, kh, p.spin_rate)
    state[:, 6] = equilibrium_extension(mc, rc, kc, p.spin_rate)
    state[:, 8] = np.mod(p.spin_rate * tau, 2.0 * math.pi); state[:, 9] = p.spin_rate
    state[:, 10] = 1.0
    center = 0.5 * tau[-1]
    local = tau - (center - 100.0)
    active_c = (local >= cent["tau"][0]) & (local <= cent["tau"][-1])
    for col, key in [(4,"xh"),(5,"vh"),(6,"xc"),(7,"vc"),(8,"theta"),(9,"omega")]:
        state[active_c, col] = np.interp(local[active_c], cent["tau"], cent[key])
    active_a = (local >= att["tau"][0]) & (local <= att["tau"][-1])
    for j in range(4): state[active_a,10+j] = np.interp(local[active_a],att["tau"],att["q"][j])
    for j in range(3): state[active_a,14+j] = np.interp(local[active_a],att["tau"],att["omega"][j])
    qnorm = np.linalg.norm(state[:,10:14],axis=1); state[:,10:14] /= qnorm[:,None]
    return {"tau":tau,"state":state,"translation":tr,"centrifuge":cent,"attitude":att,
            "columns":["t","x","eta","m","xi_h","xi_h_dot","xi_c","xi_c_dot","theta","theta_dot","q0","q1","q2","q3","omega_x","omega_y","omega_z"]}
