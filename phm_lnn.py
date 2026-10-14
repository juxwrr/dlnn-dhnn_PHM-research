import argparse
import json
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax.example_libraries import optimizers, stax
from jax.experimental.ode import odeint

jax.config.update("jax_enable_x64", True)

# Internet-sourced physical/astronomical constants. Source links and modeling
# assumptions are listed in README.md.
C = 299_792_458.0
AU = 149_597_870_700.0
JULIAN_YEAR = 365.25 * 86_400.0
LIGHT_YEAR = C * JULIAN_YEAR
GM_SUN = 1.327_124_4e20
LUMINOSITY_SUN = 3.828e26
TAU_CETI_MASS_IN_SUNS = 0.783
TAU_CETI_LUMINOSITY_IN_SUNS = 0.52
GM_TAU_CETI = TAU_CETI_MASS_IN_SUNS * GM_SUN
LUMINOSITY_TAU_CETI = TAU_CETI_LUMINOSITY_IN_SUNS * LUMINOSITY_SUN

# Mission values from astrophage physics.xlsx, Hail Mary!A1:B8.
PROPER_ACCELERATION = 15.0
MASS = 100_000.0
TRIP_DISTANCE = 11.9 * LIGHT_YEAR

# The PDF leaves these spacecraft/boundary values unspecified. They are stated
# assumptions, not claimed measurements: a 4 m frontal diameter, perfectly
# absorbing surface, and 1 AU endpoint offsets to avoid point-mass singularities.
SHIP_DIAMETER = 4.0
CROSS_SECTIONAL_AREA = np.pi * (SHIP_DIAMETER / 2) ** 2
RADIATION_COEFFICIENT = 1.0
SOURCE_OFFSET = AU
TARGET_OFFSET = AU

K_SUN = (
    RADIATION_COEFFICIENT
    * CROSS_SECTIONAL_AREA
    * LUMINOSITY_SUN
    / (4 * np.pi * C)
)
K_TAU_CETI = (
    RADIATION_COEFFICIENT
    * CROSS_SECTIONAL_AREA
    * LUMINOSITY_TAU_CETI
    / (4 * np.pi * C)
)
THRUST = MASS * PROPER_ACCELERATION

# Assumed fixed spacecraft charge and stationary axial electric field.
# 20 V on a 10 m equivalent-radius conductor: Q = 4 pi epsilon_0 R V.
VACUUM_PERMITTIVITY = 8.854_187_8188e-12
SPACECRAFT_CHARGE = 4 * np.pi * VACUUM_PERMITTIVITY * 10.0 * 20.0
AXIAL_ELECTRIC_FIELD = 1e-5

# q = ax/c^2, s = at/c, beta = v/c. S_HALF and BETA_HALF are the
# propulsion-only reference horizon; the full gravity/radiation trajectory is
# integrated numerically over the same interval.
LENGTH_SCALE = C**2 / PROPER_ACCELERATION
Q_TOTAL = TRIP_DISTANCE / LENGTH_SCALE
Q_HALF = Q_TOTAL / 2
S_HALF = np.sqrt(Q_HALF * (Q_HALF + 2))
BETA_HALF = S_HALF / np.sqrt(1 + S_HALF**2)
BETA_TRAIN = 0.7

BLUE = "#2F6BFF"
ORANGE = "#F28E2B"
GREEN = "#2A9D6F"
RED = "#D1495B"
DARK = "#263238"


@jax.jit
def kinetic_lagrangian(q, q_dot):
    """Velocity-dependent part of L/(m c^2), with rest offset added."""
    beta = q_dot[0]
    return 1 - jnp.sqrt(1 - beta**2)


@jax.jit
def stellar_distances(q):
    """Distances from the Sun and Tau Ceti for axial coordinate q."""
    x = q[0] * LENGTH_SCALE
    return SOURCE_OFFSET + x, TRIP_DISTANCE + TARGET_OFFSET - x


@jax.jit
def gravitational_potential_energy(q, q_dot):
    """Dimensionless U_gravity/(m c^2) from both stars."""
    del q_dot
    r_sun, r_tau = stellar_distances(q)
    return -GM_SUN / (C**2 * r_sun) - GM_TAU_CETI / (C**2 * r_tau)


@jax.jit
def radiation_potential_energy(q, q_dot):
    """Dimensionless U_radiation/(m c^2) from both stars."""
    del q_dot
    r_sun, r_tau = stellar_distances(q)
    return K_SUN / (MASS * C**2 * r_sun) + K_TAU_CETI / (
        MASS * C**2 * r_tau
    )


@jax.jit
def potential_energy(q, q_dot):
    """All conservative axial potential terms supplied by the PDF."""
    electric = -SPACECRAFT_CHARGE * AXIAL_ELECTRIC_FIELD * q[0] * LENGTH_SCALE / (MASS * C**2)
    return (
        gravitational_potential_energy(q, q_dot)
        + radiation_potential_energy(q, q_dot)
        + electric
    )


@jax.jit
def lagrangian(q, q_dot):
    """Exact dimensionless conservative axial Lagrangian, L/(m c^2)."""
    return kinetic_lagrangian(q, q_dot) - potential_energy(q, q_dot)


@jax.jit
def kinetic_energy(q, q_dot):
    beta = q_dot[0]
    gamma = 1 / jnp.sqrt(1 - beta**2)
    return gamma - 1


@jax.jit
def hamiltonian(q, q_dot):
    """Exact dimensionless stored energy, excluding engine work."""
    return kinetic_energy(q, q_dot) + potential_energy(q, q_dot)


@jax.jit
def conservative_accelerations(q):
    """Physical acceleration contributions in m/s^2; signs follow the PDF."""
    r_sun, r_tau = stellar_distances(q)
    gravity_sun = -GM_SUN / r_sun**2
    gravity_tau = GM_TAU_CETI / r_tau**2
    radiation_sun = K_SUN / (MASS * r_sun**2)
    radiation_tau = -K_TAU_CETI / (MASS * r_tau**2)
    return jnp.stack([gravity_sun, gravity_tau, radiation_sun, radiation_tau])


@jax.jit
def external_thrust(q, q_dot, t=0):
    """Dimensionless generalized engine force Q/(m a); fixed mass for now."""
    del q, q_dot, t
    return jnp.array([THRUST / (MASS * PROPER_ACCELERATION)])


@jax.jit
def analytical_dynamics(state, t=0):
    """Section 14 axial equation with drag, impacts, and mass loss disabled."""
    del t
    q, beta = state
    environmental_acceleration = jnp.sum(
        conservative_accelerations(jnp.array([q]))
    ) + SPACECRAFT_CHARGE * AXIAL_ELECTRIC_FIELD / MASS
    coordinate_acceleration = (
        PROPER_ACCELERATION + environmental_acceleration
    ) * (1 - beta**2) ** 1.5
    q_prime = beta
    beta_prime = coordinate_acceleration / PROPER_ACCELERATION
    return jnp.stack([q_prime, beta_prime])


# Copied from Cranmer's tutorial and extended by the +Q term from the supplied
# Lagrange-d'Alembert equation.
def equation_of_motion(
    lagrangian, state, t=None, generalized_force=external_thrust
):
    q, q_t = jnp.split(state, 2)
    force = (
        jnp.zeros_like(q)
        if generalized_force is None
        else generalized_force(q, q_t, t)
    )
    q_tt = jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t)) @ (
        jax.grad(lagrangian, 0)(q, q_t)
        + force
        - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t
    )
    return jnp.concatenate([q_t, q_tt])


def solve_lagrangian(
    lagrangian, initial_state, times, generalized_force=external_thrust
):
    return odeint(
        partial(equation_of_motion, lagrangian, generalized_force=generalized_force),
        initial_state,
        times,
        rtol=1e-10,
        atol=1e-10,
    )


def propulsion_only_state(s):
    """Closed-form reference with stellar gravity and radiation turned off."""
    q = jnp.sqrt(1 + s**2) - 1
    beta = s / jnp.sqrt(1 + s**2)
    return jnp.stack([q, beta])


def make_data(seed, count, beta_min, beta_max):
    """Sample q across the trip and beta in the requested speed band."""
    q_key, beta_key, sign_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.uniform(q_key, (count,), minval=0, maxval=Q_HALF)
    beta = jax.random.uniform(beta_key, (count,), minval=beta_min, maxval=beta_max)
    signs = jax.random.choice(sign_key, jnp.array([-1.0, 1.0]), (count,))
    state = jnp.stack([q, beta * signs], axis=1)
    return state, jax.vmap(analytical_dynamics)(state)


# Copied from the self-contained tutorial; only the state has changed.
def learned_lagrangian(params, nn_forward_fn):
    def learned(q, q_t):
        state = jnp.concatenate([q, q_t])
        return jnp.squeeze(nn_forward_fn(params, state), axis=-1)

    return learned


@partial(jax.jit, static_argnums=2)
def loss(params, batch, nn_forward_fn):
    state, targets = batch
    learned = learned_lagrangian(params, nn_forward_fn)
    preds = jax.vmap(partial(equation_of_motion, learned))(state)
    return jnp.mean((preds - targets) ** 2)


# Copied from lnn/core.py in the original repository.
def custom_init(init_params, seed=0):
    """Optimized LNN initialization for a uniform-width MLP."""
    new_params = []
    rng = jax.random.PRNGKey(seed)
    i = 0
    number_layers = len([0 for layer in init_params if len(layer) != 0])
    for layer in init_params:
        if len(layer) == 0:
            new_params.append(())
            continue
        new_layer = []
        for value in layer:
            if len(value.shape) == 1:
                new_layer.append(jnp.zeros_like(value))
            else:
                n = max(value.shape)
                first = int(i == 0)
                last = int(i == number_layers - 1)
                mid = int(i != 0 and i != number_layers - 1) * i
                std = (2.2 * first + 0.58 * mid + n * last) / np.sqrt(n)
                new_layer.append(jax.random.normal(rng, value.shape) * std)
                rng, _ = jax.random.split(rng)
                i += 1
        new_params.append(new_layer)
    return new_params


def make_model(seed=0):
    init_random_params, nn_forward_fn = stax.serial(
        stax.Dense(128),
        stax.Softplus,
        stax.Dense(128),
        stax.Softplus,
        stax.Dense(1),
    )
    _, params = init_random_params(jax.random.PRNGKey(seed), (-1, 2))
    return custom_init(params, seed), nn_forward_fn


def train(steps, samples, checkpoint):
    """Tutorial-style full-batch derivative training."""
    x_train, xt_train = make_data(0, samples, 0.0, BETA_TRAIN)
    x_test, xt_test = make_data(1, samples, BETA_TRAIN, BETA_HALF)
    params, nn_forward_fn = make_model()

    schedule = lambda t: jnp.select(
        [t < steps // 3, t < 2 * steps // 3], [1e-3, 3e-4], 1e-4
    )
    opt_init, opt_update, get_params = optimizers.adam(schedule)
    opt_state = opt_init(params)

    @jax.jit
    def update(i, state):
        current = get_params(state)
        gradient = jax.grad(loss)(current, (x_train, xt_train), nn_forward_fn)
        return opt_update(i, gradient, state)

    train_losses, test_losses = [], []
    for iteration in range(steps + 1):
        if iteration % 1000 == 0:
            params = get_params(opt_state)
            train_loss = float(loss(params, (x_train, xt_train), nn_forward_fn))
            test_loss = float(loss(params, (x_test, xt_test), nn_forward_fn))
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(
                f"iteration={iteration}, train_loss={train_loss:.6e}, "
                f"test_loss={test_loss:.6e}"
            )
        if iteration < steps:
            opt_state = update(iteration, opt_state)

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as handle:
        pickle.dump(
            {
                "params": jax.device_get(get_params(opt_state)),
                "train_losses": train_losses,
                "test_losses": test_losses,
            },
            handle,
        )
    return get_params(opt_state), nn_forward_fn, train_losses, test_losses


def learned_energy(learned, state):
    q, q_t = jnp.split(state, 2)
    momentum = jax.grad(learned, 1)(q, q_t)
    return jnp.dot(momentum, q_t) - learned(q, q_t)


def exact_energy(state):
    q, q_t = jnp.split(state, 2)
    return hamiltonian(q, q_t)


def evaluate(params, nn_forward_fn, output_dir, train_losses=None, test_losses=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    learned = learned_lagrangian(params, nn_forward_fn)
    times = jnp.linspace(0, S_HALF, 500)
    exact = solve_lagrangian(lagrangian, jnp.array([0.0, 0.0]), times)
    model = solve_lagrangian(learned, jnp.array([0.0, 0.0]), times)
    propulsion_only = jax.vmap(propulsion_only_state)(times)

    learned_h = jax.vmap(partial(learned_energy, learned))(model)
    learned_work = model[:, 0] - model[0, 0]
    learned_balance_error = jnp.abs(learned_h - learned_h[0] - learned_work)
    valid = jnp.abs(model[:, 1]) < 1
    exact_h_on_model = jax.vmap(exact_energy)(model[valid])
    exact_work = model[valid, 0] - model[0, 0]
    exact_balance_error = jnp.abs(exact_h_on_model - exact_h_on_model[0] - exact_work)
    beta_rmse = float(jnp.sqrt(jnp.mean((model[:, 1] - exact[:, 1]) ** 2)))

    metrics = {
        "beta_rmse": beta_rmse,
        "learned_energy_balance_max_abs_residual": float(jnp.max(learned_balance_error)),
        "exact_energy_balance_max_abs_residual_on_learned_path": float(
            jnp.max(exact_balance_error)
        ),
        "full_model_final_beta": float(exact[-1, 1]),
        "propulsion_only_reference_final_beta": float(BETA_HALF),
        "learned_final_beta": float(model[-1, 1]),
        "speed_limit_violation_count": int(jnp.sum(~valid)),
        "trajectory_sample_count": int(len(times)),
        "first_speed_limit_violation_s": (
            float(times[jnp.where(~valid)[0][0]]) if bool(jnp.any(~valid)) else None
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, exact[:, 1], label="Exact PDF-based model", linewidth=2.2)
    plt.plot(times, model[:, 1], "--", label="LNN", linewidth=2.0)
    plt.plot(
        times,
        propulsion_only[:, 1],
        ":",
        label="Propulsion-only reference",
        linewidth=1.5,
    )
    plt.axhline(1, color="black", linewidth=1)
    plt.ylim(
        min(0.0, float(jnp.min(model[:, 1])) - 0.1),
        max(1.15, float(jnp.max(model[:, 1])) + 0.1),
    )
    plt.xlabel("Dimensionless time s = at/c")
    plt.ylabel("Velocity beta = v/c")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    floor = np.finfo(float).eps
    plt.semilogy(
        times,
        jnp.maximum(learned_balance_error, floor),
        label="learned energy-balance residual",
        color=GREEN,
        linewidth=2.2,
    )
    plt.semilogy(
        times[valid],
        jnp.maximum(exact_balance_error, floor),
        label="true PDF-model energy-balance residual",
        color=RED,
        linewidth=2.2,
    )
    plt.xlabel(r"Dimensionless time $s=at/c$")
    plt.ylabel("Absolute balance residual")
    plt.title("Stored-energy change minus known engine work")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "learned_conservation.png", dpi=180)
    plt.close()

    if train_losses:
        plt.figure(figsize=(7, 4))
        plt.plot(train_losses, label="Train")
        plt.plot(test_losses, label="Test: beta > 0.7")
        plt.yscale("log")
        plt.xlabel("Checkpoint (1000 optimizer steps)")
        plt.ylabel("Mean squared derivative error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss.png", dpi=180)
        plt.close()
    print(json.dumps(metrics, indent=2))


def baseline(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    times = jnp.linspace(0, S_HALF, 500)
    full = solve_lagrangian(lagrangian, jnp.array([0.0, 0.0]), times)
    propulsion_only = jax.vmap(propulsion_only_state)(times)
    plt.figure(figsize=(8, 4.5))
    plt.plot(times, full[:, 1], label="Gravity + radiation + thrust", linewidth=2.2)
    plt.plot(
        times,
        propulsion_only[:, 1],
        "--",
        label="Thrust-only relativistic reference",
        linewidth=2,
    )
    plt.axhline(1, color="black", linewidth=1, label="Speed of light")
    plt.ylim(0, 1.15)
    plt.xlabel("Dimensionless time s = at/c")
    plt.ylabel("Velocity beta = v/c")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "baseline.png", dpi=180)
    plt.close()
    print(f"full-model final beta={float(full[-1, 1]):.10f}")
    print(f"propulsion-only final beta={BETA_HALF:.10f}")
    print(f"plot={output_dir / 'baseline.png'}")


def figures(output_dir):
    """Create PDF-based physics figures without training the LNN."""
    output_dir.mkdir(parents=True, exist_ok=True)
    times = jnp.linspace(0, S_HALF, 1000)
    full = solve_lagrangian(lagrangian, jnp.array([0.0, 0.0]), times)
    propulsion_only = jax.vmap(propulsion_only_state)(times)
    q_exact = np.asarray(full[:, 0])
    beta_exact = np.asarray(full[:, 1])
    times_np = np.asarray(times)
    reference_q = np.asarray(propulsion_only[:, 0])
    reference_beta = np.asarray(propulsion_only[:, 1])
    crossing_index = int(np.argmax(beta_exact >= BETA_TRAIN))
    train_crossing = times_np[crossing_index]

    # 1. The exact forced trajectory generated from the conservative Lagrangian.
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(times_np, q_exact, color=BLUE, linewidth=2.5, label="PDF-based")
    axes[0].plot(
        times_np,
        reference_q,
        color=DARK,
        linestyle=":",
        linewidth=1.6,
        label="thrust only",
    )
    axes[0].set_ylabel(r"Position $q=ax/c^2$")
    axes[0].set_title("Relativistic PHM trajectory with stellar fields")
    axes[0].legend(frameon=False)
    axes[1].plot(times_np, beta_exact, color=ORANGE, linewidth=2.5, label="PDF-based")
    axes[1].plot(
        times_np,
        reference_beta,
        color=DARK,
        linestyle=":",
        linewidth=1.6,
        label="thrust only",
    )
    axes[1].axhline(1, color=DARK, linestyle=":", label="speed of light")
    axes[1].axhline(
        BETA_TRAIN, color=GREEN, linestyle="--", label="training-speed boundary"
    )
    axes[1].axvspan(
        train_crossing,
        S_HALF,
        color=RED,
        alpha=0.08,
        label=r"trajectory has $\beta>0.7$",
    )
    axes[1].set_xlabel(r"Dimensionless time $s=at/c$")
    axes[1].set_ylabel(r"Speed $\beta=v/c$")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, loc="lower right")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "exact_trajectory.png", dpi=180)
    plt.close(fig)

    # 2. Each conservative stellar-force contribution from Section 14.
    q_grid = jnp.linspace(0, Q_HALF, 1000)
    components = np.asarray(
        jax.vmap(conservative_accelerations)(q_grid[:, None])
    )
    distance_ly = np.asarray(q_grid) * LENGTH_SCALE / LIGHT_YEAR
    fig, axis = plt.subplots(figsize=(8, 4.8))
    labels = ["Sun gravity", "Tau Ceti gravity", "Sun radiation", "Tau Ceti radiation"]
    colors = [BLUE, ORANGE, GREEN, RED]
    for column, (label, color) in enumerate(zip(labels, colors)):
        axis.semilogy(
            distance_ly,
            np.maximum(np.abs(components[:, column]), np.finfo(float).tiny),
            color=color,
            linewidth=2,
            label=label,
        )
    axis.axhline(PROPER_ACCELERATION, color=DARK, linestyle=":", label="engine acceleration")
    axis.set_xlabel("Distance from the Sun (light-years)")
    axis.set_ylabel(r"Acceleration magnitude (m s$^{-2}$)")
    axis.set_title("Conservative force terms used by the axial LNN")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "force_components.png", dpi=180)
    plt.close(fig)

    # 3. Phase-space view, including the tutorial-style interpolation/test split.
    train_mask = beta_exact <= BETA_TRAIN
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(
        q_exact[train_mask],
        beta_exact[train_mask],
        color=GREEN,
        linewidth=3,
        label=r"inside training-speed band ($\beta\leq0.7$)",
    )
    axis.plot(
        q_exact[~train_mask],
        beta_exact[~train_mask],
        color=RED,
        linewidth=3,
        label=r"extrapolation band ($\beta>0.7$)",
    )
    axis.scatter(
        [q_exact[-1]],
        [beta_exact[-1]],
        color=DARK,
        s=45,
        zorder=3,
        label="end of reference interval",
    )
    axis.set_xlabel(r"Position $q=ax/c^2$")
    axis.set_ylabel(r"Speed $\beta=v/c$")
    axis.set_title("Exact phase-space path and the LNN test region")
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_space.png", dpi=180)
    plt.close(fig)

    # 4. Because thrust is external, H changes by the known engine work.
    energies = np.asarray(jax.vmap(exact_energy)(full))
    work = q_exact - q_exact[0]
    energy_change = energies - energies[0]
    residual = energy_change - work
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(times_np, energy_change, color=BLUE, linewidth=2.4, label=r"$H-H_0$")
    axes[0].plot(times_np, work, color=ORANGE, linestyle="--", linewidth=2, label="engine work")
    axes[0].set_ylabel("Dimensionless energy/work")
    axes[0].set_title("Energy bookkeeping for the PDF-based forced model")
    axes[0].legend(frameon=False)
    axes[1].semilogy(
        times_np,
        np.maximum(np.abs(residual), np.finfo(float).eps),
        color=RED,
        linewidth=2,
    )
    axes[1].set_xlabel(r"Dimensionless time $s=at/c$")
    axes[1].set_ylabel(r"$|H-H_0-W_{engine}|$")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "energy_balance.png", dpi=180)
    plt.close(fig)

    names = [
        "exact_trajectory.png",
        "force_components.png",
        "phase_space.png",
        "energy_balance.png",
    ]
    print("created analytic figures:")
    for name in names:
        print(output_dir / name)


def check_physics():
    q_samples = jnp.linspace(0, Q_HALF, 128)
    beta_samples = jnp.linspace(-0.95, 0.95, 128)
    states = jnp.stack([q_samples, beta_samples], axis=1)
    autograd = jax.vmap(partial(equation_of_motion, lagrangian))(states)
    analytical = jax.vmap(analytical_dynamics)(states)
    dynamics_error = float(jnp.max(jnp.abs(autograd - analytical)))
    assert dynamics_error < 1e-10

    start_terms = np.asarray(conservative_accelerations(jnp.array([0.0])))
    assert start_terms[0] < 0
    assert start_terms[1] > 0
    assert start_terms[2] > 0
    assert start_terms[3] < 0

    times = jnp.linspace(0, S_HALF, 1000)
    trajectory = solve_lagrangian(lagrangian, jnp.array([0.0, 0.0]), times)
    energies = jax.vmap(exact_energy)(trajectory)
    engine_work = trajectory[:, 0] - trajectory[0, 0]
    balance_error = float(
        jnp.max(jnp.abs(energies - energies[0] - engine_work))
    )
    # The residual is about 1e-8 relative to the total work over this long,
    # relativistic integration interval.
    assert balance_error < 2e-7

    print(
        json.dumps(
            {
                "physics_checks": "passed",
                "max_dynamics_identity_error": dynamics_error,
                "max_energy_balance_residual": balance_error,
                "start_acceleration_terms_m_per_s2": {
                    "sun_gravity": float(start_terms[0]),
                    "tau_ceti_gravity": float(start_terms[1]),
                    "sun_radiation": float(start_terms[2]),
                    "tau_ceti_radiation": float(start_terms[3]),
                },
                "full_model_final_beta": float(trajectory[-1, 1]),
                "propulsion_only_reference_final_beta": float(BETA_HALF),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results_full_conservative/phm_lnn.pkl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_full_conservative")
    )
    args = parser.parse_args()

    if args.check:
        check_physics()
    if args.baseline:
        baseline(args.output_dir)
    if args.figures:
        figures(args.output_dir)
    if args.train:
        params, apply_fn, train_losses, test_losses = train(
            args.steps, args.samples, args.checkpoint
        )
        evaluate(params, apply_fn, args.output_dir, train_losses, test_losses)
    if args.evaluate:
        with args.checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        _, apply_fn = make_model()
        evaluate(
            saved["params"],
            apply_fn,
            args.output_dir,
            saved.get("train_losses"),
            saved.get("test_losses"),
        )
    if not any([args.check, args.baseline, args.figures, args.train, args.evaluate]):
        parser.print_help()


if __name__ == "__main__":
    main()
