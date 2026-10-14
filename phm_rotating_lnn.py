"""Conservative attitude and 1g centrifuge LNN experiments.

The blocks use physical time scales appropriate to each subsystem. Stellar
distances are frozen for the attitude benchmark; the centrifuge is a local
comoving experiment. This is a reduced, separable model, not a coupled
relativistic flexible-body mission simulator. Kinetic geometry is known;
two smooth scalar neural potentials are learned from derivative labels.
"""

import argparse
import json
import pickle
from functools import partial
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax.example_libraries import optimizers, stax
from jax.experimental.ode import odeint

import phm_lnn as axial

jax.config.update("jax_enable_x64", True)
G0 = 9.80665
HABITAT_MASS, COUNTERWEIGHT_MASS = 20_000.0, 30_000.0
HABITAT_RADIUS, COUNTERWEIGHT_RADIUS = 30.0, 20.0
CABLES, CABLE_DIAMETER, YOUNG_MODULUS = 4, 0.01, 120e9
CABLE_AREA = np.pi * CABLE_DIAMETER**2 / 4
MASSES = jnp.array([HABITAT_MASS, COUNTERWEIGHT_MASS])
RADII = jnp.array([HABITAT_RADIUS, COUNTERWEIGHT_RADIUS])
STIFFNESS = CABLES * YOUNG_MODULUS * CABLE_AREA / RADII
HABITAT_EXTENSION = HABITAT_MASS * G0 / float(STIFFNESS[0])
SPIN = np.sqrt(G0 / (HABITAT_RADIUS + HABITAT_EXTENSION))
EQUILIBRIUM = MASSES * RADII * SPIN**2 / (STIFFNESS - MASSES * SPIN**2)

# Residual dipole and uniform field are explicit assumptions, not PHM data.
MAGNETIC_FIELD = 0.5e-9
MAGNETIC_MOMENT = 100.0
FIELD_ANGLE = np.deg2rad(30.0)
HUB_MASS, HUB_LENGTH, HUB_RADIUS = 50_000.0, 47.0, 2.0
INERTIA_PARALLEL = 0.5 * HUB_MASS * HUB_RADIUS**2
INERTIA_TRANSVERSE = HUB_MASS * (3 * HUB_RADIUS**2 + HUB_LENGTH**2) / 12
INERTIA = jnp.array([INERTIA_PARALLEL, INERTIA_TRANSVERSE, INERTIA_TRANSVERSE])
TIDAL_COEFFICIENT = axial.GM_SUN / axial.AU**3 + axial.GM_TAU_CETI / (
    axial.TRIP_DISTANCE + axial.AU
)**3
ATTITUDE_ENERGY = 1.5 * TIDAL_COEFFICIENT * (INERTIA_TRANSVERSE - INERTIA_PARALLEL)
ATTITUDE_TIME = np.sqrt(INERTIA_TRANSVERSE / ATTITUDE_ENERGY)
EXTENSION_SCALE = 0.20
CENTRIFUGE_ENERGY = float(STIFFNESS[0]) * EXTENSION_SCALE**2
CENTRIFUGE_TIME = np.sqrt(HABITAT_MASS / float(STIFFNESS[0]))
CENTRIFUGE_INERTIA = float(jnp.sum(MASSES * (RADII + EQUILIBRIUM)**2))
ANGLE_SCALE = EXTENSION_SCALE * np.sqrt(HABITAT_MASS / CENTRIFUGE_INERTIA)


def gravity_gradient_energy(angle, source_distance=axial.AU,
                           target_distance=axial.TRIP_DISTANCE + axial.AU):
    """PDF U_gg with body-frame direction to each stationary star."""
    n_body = jnp.array([jnp.cos(angle), -jnp.sin(angle), 0.0])
    mu_over_r3 = axial.GM_SUN / source_distance**3 + axial.GM_TAU_CETI / target_distance**3
    return -0.5 * mu_over_r3 * (jnp.sum(INERTIA) - 3 * jnp.dot(INERTIA * n_body, n_body))


def magnetic_energy(angle):
    return -MAGNETIC_MOMENT * MAGNETIC_FIELD * jnp.cos(angle - FIELD_ANGLE)


def combined_lagrangian_si(q, velocity):
    """PDF reduced SI L for [x, attitude, extension_h, extension_c, spin_angle].

    Internal velocities are interpreted in the local slow-motion limit.
    The tests below use scaled blocks, avoiding a poorly conditioned SI Hessian.
    In 1D a gauge with A_x=0 can be used. Transverse Lorentz deflection is
    projected out by the axial constraint; a three-dimensional model is needed
    to predict that deflection.
    """
    x, angle, h, c, theta = q
    v, rate, vh, vc, spin_rate = velocity
    r_s, r_t = axial.SOURCE_OFFSET + x, axial.TRIP_DISTANCE + axial.TARGET_OFFSET - x
    translation = -axial.MASS * axial.C**2 * jnp.sqrt(1 - (v / axial.C)**2)
    gravity = axial.MASS * (axial.GM_SUN / r_s + axial.GM_TAU_CETI / r_t)
    radiation = -axial.K_SUN / r_s - axial.K_TAU_CETI / r_t
    electric = axial.SPACECRAFT_CHARGE * axial.AXIAL_ELECTRIC_FIELD * x
    attitude = 0.5 * INERTIA_TRANSVERSE * rate**2 - gravity_gradient_energy(angle, r_s, r_t) - magnetic_energy(angle)
    extension = jnp.array([h, c])
    radial_velocity = jnp.array([vh, vc])
    centrifuge = jnp.sum(0.5 * MASSES * (radial_velocity**2 + (RADII + extension)**2 * spin_rate**2) - 0.5 * STIFFNESS * extension**2)
    return translation + gravity + radiation + electric + attitude + centrifuge


def kinetic(kind, q, velocity):
    if kind == "attitude":
        return 0.5 * velocity[0]**2
    radius = RADII + EXTENSION_SCALE * q[:2]
    radial = 0.5 * jnp.sum(MASSES / HABITAT_MASS * velocity[:2]**2)
    moment = jnp.sum(MASSES * radius**2)
    spin = 0.5 * moment * ANGLE_SCALE**2 * velocity[2]**2 / (
        CENTRIFUGE_ENERGY * CENTRIFUGE_TIME**2
    )
    return radial + spin


def potential(kind, q):
    if kind == "attitude":
        # Remove only the attitude-independent constant for numerical clarity.
        return (gravity_gradient_energy(q[0]) - gravity_gradient_energy(0.0) + magnetic_energy(q[0])) / ATTITUDE_ENERGY
    return 0.5 * jnp.sum(STIFFNESS * EXTENSION_SCALE**2 * q[:2]**2) / CENTRIFUGE_ENERGY


def exact_lagrangian(kind, q, velocity):
    return kinetic(kind, q, velocity) - potential(kind, q)


def analytical_dynamics(kind, state):
    """Independent explicit equations from PDF Sections 9 and 10."""
    q, w = jnp.split(state, 2)
    if kind == "attitude":
        psi = q[0]
        acceleration = -jnp.sin(2 * psi) - MAGNETIC_MOMENT * MAGNETIC_FIELD / ATTITUDE_ENERGY * jnp.sin(psi - FIELD_ANGLE)
        return jnp.concatenate([w, jnp.array([acceleration])])
    radius = RADII + EXTENSION_SCALE * q[:2]
    physical_spin = ANGLE_SCALE / CENTRIFUGE_TIME * w[2]
    radial_acc = radius * physical_spin**2 - STIFFNESS / MASSES * EXTENSION_SCALE * q[:2]
    moment = jnp.sum(MASSES * radius**2)
    moment_prime = 2 * jnp.sum(MASSES * radius * EXTENSION_SCALE * w[:2])
    spin_acc = -moment_prime / moment * w[2]
    return jnp.concatenate([w, radial_acc * CENTRIFUGE_TIME**2 / EXTENSION_SCALE, jnp.array([spin_acc])])


def model(kind, seed=0):
    inputs = 2 if kind == "attitude" else 2
    init, forward = stax.serial(stax.Dense(32), stax.Softplus, stax.Dense(32), stax.Softplus, stax.Dense(1))
    _, params = init(jax.random.PRNGKey(seed), (-1, inputs))
    return params, forward


def learned_lagrangian(kind, params, forward):
    def lagrangian(q, velocity):
        # Periodic embedding prevents an artificial discontinuity at 2 pi.
        features = jnp.array([jnp.sin(q[0]), jnp.cos(q[0])]) if kind == "attitude" else q[:2]
        learned_v = jnp.squeeze(forward(params, features))
        return kinetic(kind, q, velocity) - learned_v
    return lagrangian


def make_data(kind, seed, count, test=False):
    rng = np.random.default_rng(seed)
    if kind == "attitude":
        q = rng.uniform(-np.pi, np.pi, (count, 1))
        w = rng.uniform(-0.6, 0.6, (count, 1))
    else:
        width = 0.30 if test else 0.15
        q = np.column_stack([np.asarray(EQUILIBRIUM / EXTENSION_SCALE) + rng.uniform(-width, width, (count, 2)), rng.uniform(-np.pi, np.pi, count)])
        w = rng.uniform(-0.3, 0.3, (count, 3))
        w[:, 2] = SPIN * CENTRIFUGE_TIME / ANGLE_SCALE * rng.uniform(0.9, 1.1, count)
    state = jnp.array(np.concatenate([q, w], axis=1))
    return state, jax.vmap(partial(analytical_dynamics, kind))(state)


def train(kind, steps, count, out):
    params, forward = model(kind)
    states, targets = make_data(kind, 0, count)
    test_states, test_targets = make_data(kind, 1, count, True)
    def loss(weights, x, y):
        learned = learned_lagrangian(kind, weights, forward)
        predictions = jax.vmap(partial(axial.equation_of_motion, learned, generalized_force=None))(x)
        # Coordinate derivatives are supplied exactly by the LNN structure.
        return jnp.mean((predictions[:, x.shape[1] // 2:] - y[:, x.shape[1] // 2:])**2)
    evaluate_loss = jax.jit(loss)
    opt_init, opt_update, get_params = optimizers.adam(lambda i: jnp.where(i < steps * 0.7, 1e-3, 2e-4))
    optimizer = opt_init(params)
    @jax.jit
    def update(i, state):
        weights = get_params(state)
        return opt_update(i, jax.grad(loss)(weights, states, targets), state)
    history = []
    best, best_loss = params, np.inf
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % 250 == 0 or step == steps:
            weights = get_params(optimizer)
            training = float(evaluate_loss(weights, states, targets))
            testing = float(evaluate_loss(weights, test_states, test_targets))
            if not np.isfinite(training + testing):
                raise RuntimeError(f"Nonfinite loss for {kind} at {step}")
            history.append([step, training, testing])
            if training < best_loss:
                best, best_loss = weights, training
            print(f"{kind} step={step} train={training:.6g} test={testing:.6g}", flush=True)
        if step < steps:
            optimizer = update(step, optimizer)
    with (out / f"{kind}.pkl").open("wb") as handle:
        pickle.dump({"params": jax.device_get(best), "history": history, "steps": steps, "samples": count, "model": "structured potential LNN", "seconds": time.perf_counter() - started}, handle)
    return best, forward, history


def evaluate(kind, params, forward, history, out):
    learned = learned_lagrangian(kind, params, forward)
    exact = partial(exact_lagrangian, kind)
    if kind == "attitude":
        initial = jnp.array([np.deg2rad(20), 0.0])
        times = jnp.linspace(0, 8.0, 800)
        physical_times = np.asarray(times) * ATTITUDE_TIME / 86_400
    else:
        initial = jnp.concatenate([EQUILIBRIUM / EXTENSION_SCALE + jnp.array([0.05, -0.04]), jnp.array([0.0, 0.0, 0.0, SPIN * CENTRIFUGE_TIME / ANGLE_SCALE])])
        times = jnp.linspace(0, 20 / CENTRIFUGE_TIME, 1600)
        physical_times = np.asarray(times) * CENTRIFUGE_TIME
    solve = lambda lag: axial.solve_lagrangian(lag, initial, times, generalized_force=None)
    truth, prediction = solve(exact), solve(learned)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Nonfinite trajectory for {kind}")
    h_true = jax.vmap(partial(axial.learned_energy, exact))(truth)
    h_on_learned = jax.vmap(partial(axial.learned_energy, exact))(prediction)
    h_learned = jax.vmap(partial(axial.learned_energy, learned))(prediction)
    h_scale = max(float(jnp.abs(h_true[0])), 1e-12)
    metrics = {"train_acceleration_mse": float(history[-1][1]), "test_acceleration_mse": float(history[-1][2]), "trajectory_state_rmse_scaled": float(jnp.sqrt(jnp.mean((truth - prediction)**2))), "true_energy_relative_defect_on_learned_path": float(jnp.max(jnp.abs(h_on_learned - h_on_learned[0])) / h_scale), "learned_energy_relative_drift": float(jnp.max(jnp.abs(h_learned - h_learned[0])) / h_scale), "reference_energy_relative_drift": float(jnp.max(jnp.abs(h_true - h_true[0])) / h_scale)}
    if kind == "attitude":
        metrics["attitude_rmse_degrees"] = float(jnp.sqrt(jnp.mean((truth[:, 0] - prediction[:, 0])**2)) * 180 / np.pi)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(physical_times, np.rad2deg(truth[:, 0]), label="physical model")
        axes[0].plot(physical_times, np.rad2deg(prediction[:, 0]), "--", label="trained LNN")
        axes[0].set_ylabel("Attitude angle (degrees)")
        axes[0].set_title("Attitude in fixed stellar and magnetic fields at 1 AU")
        axes[1].plot(physical_times, np.rad2deg(prediction[:, 0] - truth[:, 0]))
        axes[1].set_ylabel("Angle error (degrees)")
        axes[1].set_xlabel("Local benchmark time (days)")
    else:
        h_radius = HABITAT_RADIUS + EXTENSION_SCALE * truth[:, 0]
        ph_radius = HABITAT_RADIUS + EXTENSION_SCALE * prediction[:, 0]
        spin = truth[:, 5] * ANGLE_SCALE / CENTRIFUGE_TIME
        learned_spin = prediction[:, 5] * ANGLE_SCALE / CENTRIFUGE_TIME
        physical_g = h_radius * spin**2 / G0
        learned_g = ph_radius * learned_spin**2 / G0
        metrics["habitat_gravity_rmse_g"] = float(jnp.sqrt(jnp.mean((physical_g - learned_g)**2)))
        metrics["habitat_extension_rmse_m"] = float(EXTENSION_SCALE * jnp.sqrt(jnp.mean((truth[:, 0] - prediction[:, 0])**2)))
        metrics["spin_rmse_rpm"] = float(jnp.sqrt(jnp.mean((spin - learned_spin)**2)) * 60 / (2 * np.pi))
        angular_momentum = jax.vmap(lambda z: jnp.sum(MASSES * (RADII + EXTENSION_SCALE * z[:2])**2) * z[5] * ANGLE_SCALE / CENTRIFUGE_TIME)(prediction)
        metrics["spin_angular_momentum_relative_drift"] = float(jnp.max(jnp.abs(angular_momentum / angular_momentum[0] - 1)))
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(physical_times, EXTENSION_SCALE * truth[:, 0], label="physical model")
        axes[0].plot(physical_times, EXTENSION_SCALE * prediction[:, 0], "--", label="trained LNN")
        axes[0].set_ylabel("Habitat extension (m)")
        axes[0].set_title("Conservative centrifuge with a 1g equilibrium")
        axes[1].plot(physical_times, spin * 60 / (2*np.pi))
        axes[1].plot(physical_times, learned_spin * 60 / (2*np.pi), "--")
        axes[1].set_ylabel("Spin (rpm)")
        axes[2].plot(physical_times, physical_g)
        axes[2].plot(physical_times, learned_g, "--")
        axes[2].axhline(1, color="gray", linestyle=":")
        axes[2].set_ylabel("Habitat gravity (g)")
        axes[2].set_xlabel("Local comoving time (seconds)")
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / f"{kind}_trajectory.png", dpi=180)
    plt.close(fig)
    history = np.asarray(history)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.semilogy(history[:, 0], history[:, 1], label="training")
    axis.semilogy(history[:, 0], history[:, 2], label="held out" + (" wider extensions" if kind == "centrifuge" else " states"))
    axis.set(xlabel="Optimizer step", ylabel="Scaled acceleration MSE", title=f"Structured {kind} LNN")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / f"{kind}_loss.png", dpi=180)
    plt.close(fig)
    return metrics


def assumptions():
    return {"status": "assumed engineering scenario", "habitat_mass_kg": HABITAT_MASS, "counterweight_mass_kg": COUNTERWEIGHT_MASS, "hub_mass_kg": HUB_MASS, "radii_m": np.asarray(RADII).tolist(), "cable_count_per_arm": CABLES, "cable_diameter_m": CABLE_DIAMETER, "young_modulus_pa": YOUNG_MODULUS, "stiffness_N_per_m": np.asarray(STIFFNESS).tolist(), "equilibrium_extensions_m": np.asarray(EQUILIBRIUM).tolist(), "spin_rad_per_s": SPIN, "spin_rpm": SPIN * 60 / (2*np.pi), "habitat_gravity_m_per_s2": float((RADII[0] + EQUILIBRIUM[0]) * SPIN**2), "counterweight_gravity_g": float((RADII[1] + EQUILIBRIUM[1]) * SPIN**2 / G0), "inertia_kg_m2": np.asarray(INERTIA).tolist(), "magnetic_field_T": MAGNETIC_FIELD, "magnetic_moment_A_m2": MAGNETIC_MOMENT, "magnetic_field_angle_degrees": 30, "spacecraft_charge_C": axial.SPACECRAFT_CHARGE, "electric_field_V_per_m": axial.AXIAL_ELECTRIC_FIELD, "attitude_time_scale_days": ATTITUDE_TIME / 86400, "centrifuge_time_scale_s": CENTRIFUGE_TIME, "sources": ["https://ntrs.nasa.gov/citations/20205004959", "https://assets.science.nasa.gov/content/dam/science/hpd/heat/sten-guides/sten-guides-pdfs/MagnetismIntroduction_FINAL.pdf", "https://cdaweb.gsfc.nasa.gov/pub/documents/archived_websites/pwg.gsfc.nasa.gov/Education/FAQs6.html"], "approximations": ["fixed mass", "one-dimensional travel", "planar hub attitude", "frozen stellar distances for the attitude benchmark", "separate local comoving centrifuge clock", "no hub-centifuge angular momentum exchange", "no fully relativistic field transformations", "linear elastic tethers", "no damping or impacts"]}


def check():
    for kind in ("attitude", "centrifuge"):
        states, targets = make_data(kind, 14, 32)
        automatic = jax.vmap(partial(axial.equation_of_motion, partial(exact_lagrangian, kind), generalized_force=None))(states)
        error = float(jnp.max(jnp.abs(automatic - targets)))
        assert error < 1e-9, (kind, error)
        print(f"{kind} explicit/autodiff maximum error {error:.3g}", flush=True)
    assert np.isclose(float((RADII[0] + EQUILIBRIUM[0]) * SPIN**2), G0)
    stress = HABITAT_MASS * G0 / (CABLES * CABLE_AREA)
    print(f"1g equilibrium passed; spin={SPIN*60/(2*np.pi):.6f} rpm; cable stress={stress/1e6:.1f} MPa", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("results_extended"))
    args = parser.parse_args()
    if args.check:
        check()
    if args.train or args.evaluate:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "assumptions.json").write_text(json.dumps(assumptions(), indent=2))
        metrics = {}
        for kind in ("attitude", "centrifuge"):
            if args.train:
                params, forward, history = train(kind, args.steps, args.samples, args.output_dir)
            else:
                with (args.output_dir / f"{kind}.pkl").open("rb") as handle:
                    saved = pickle.load(handle)
                params, history = saved["params"], saved["history"]
                _, forward = model(kind)
            metrics[kind] = evaluate(kind, params, forward, history, args.output_dir)
            (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            print(json.dumps(metrics[kind], indent=2), flush=True)
    if not (args.check or args.train or args.evaluate):
        parser.print_help()


if __name__ == "__main__":
    main()
