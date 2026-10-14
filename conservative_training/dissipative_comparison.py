"""Compare a conservative LNN with the existing COMPLETE 1D thrust-and-drag field.

This is not a dissipative nine-coordinate spacecraft model. Drag is a known
external force, not a term learned by the conservative scalar Lagrangian.
"""
import argparse
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
from .engine import network
from .train import predictor, record_json
from .rollout_utils import integrate
from . import physics as p

DRAG = .02


def drag_force(beta, strength=DRAG):
    momentum = np.asarray(beta) / np.sqrt(1 - np.asarray(beta)**2)
    return -strength * momentum * np.abs(momentum)


def comparison(checkpoint_path, out):
    checkpoint_path = Path(checkpoint_path)
    with checkpoint_path.open('rb') as stream:
        checkpoint = pickle.load(stream)
    config = checkpoint['config']
    if config['model'] != 'incomplete' or config['baseline']:
        raise ValueError('This comparison requires the one-term conservative LNN')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    _, _, get_params = optimizers.adam(.001)
    parameters = get_params(checkpoint['optimizer_state'])
    _, forward = network(2, seed=config['seed'], dtype=jnp.float64)
    predict = jax.jit(predictor('incomplete', forward, parameters))
    horizon = float(np.sqrt((p.Q_ROUTE/2)*(p.Q_ROUTE/2+2)))
    times = np.linspace(0, horizon, 501)
    initial = [0., 0.]
    def field(learned=False, dissipative=False):
        def rhs(t, z):
            q, beta = z
            if abs(beta) >= 1:
                raise FloatingPointError(f'Superluminal trial state at t={t}')
            force = 1. + (float(drag_force(beta)) if dissipative else 0.)
            acc = float(predict(jnp.array([q]), jnp.array([beta]), jnp.array([force]))[0]) if learned else force*(1-beta*beta)**1.5
            return [beta, acc]
        return rhs
    traces = {}; statuses = {}
    cases = [('Conservative reference', False, False), ('Full 1D dissipative reference', False, True),
             ('Conservative LNN', True, False), ('LNN + known drag force', True, True)]
    for name, learned, dissipative in cases:
        traces[name], statuses[name] = integrate(field(learned, dissipative), initial, times)
    # Energy includes the constant-thrust potential. dH/ds = -kappa*p*|p|*beta.
    def total_energy(trace):
        return 1/np.sqrt(1-trace[1]**2)-1-trace[0]
    runtime = checkpoint_path.parent/'runtime.json'
    full_budget = runtime.exists() and json.loads(runtime.read_text(encoding='utf-8')).get('scientific_result', False)
    phase = 'Full-budget evaluation' if full_budget else 'Provisional smoke result — not a final trained model'
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colors = ['#2864b4', '#d16d18', '#7a48a5', '#168667']
    for (name, _, _), color in zip(cases, colors):
        trace = traces[name]
        for ax, vals in zip(axes, [trace[1], trace[0], total_energy(trace)]):
            ax.plot(times, vals, color=color, label=name, linewidth=1.8)
            if statuses[name]['failed']:
                ax.axvline(statuses[name]['last_accepted_time'], color=color, linestyle=':', alpha=.7)
        if statuses[name]['failed']:
            axes[0].annotate(f'{name}: stopped (failed)',
                             (statuses[name]['last_accepted_time'], 0), xytext=(8, 12 + 16*colors.index(color)),
                             textcoords='offset points', color=color, fontsize=8)
    axes[0].set_ylabel('Speed / light speed'); axes[0].legend(fontsize=9, loc='upper left')
    axes[1].set_ylabel('Position / (c²/a)')
    axes[2].set_ylabel('Energy H / (mc²)'); axes[2].set_xlabel('Time s = a t / c')
    axes[2].axhline(0., color='#888888', linewidth=.6)
    for ax in axes: ax.set_xlim(0, horizon); ax.grid(alpha=.18)
    fig.suptitle('Conservative LNN versus the full 1D thrust-and-drag system\n'+phase, fontsize=13)
    fig.text(.09, .025, 'Same start: position = 0, speed = 0. Thrust = 1; drag strength = 0.02.\n'
             'Drag lowers H; it is supplied externally to the LNN. Dotted lines mark failed rollouts; blank tails are not results.', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .94)); fig.savefig(out/'dissipative_comparison.png', dpi=160); plt.close(fig)
    reference = traces['Full 1D dissipative reference']
    metrics = dict(checkpoint=str(checkpoint_path.resolve()), checkpoint_step=checkpoint['step'],
                   completed_full_budget=full_budget, scientific_result=full_budget,
                   domain='complete 1D fixed-mass thrust-and-drag total field; not a dissipative nine-coordinate ship',
                   initial=initial, thrust=1., drag_strength=DRAG, horizon=horizon,
                   energy_definition='H/(mc²) = gamma - 1 - q; dH/ds = drag_force(beta)*beta',
                   comparison_policy='Common initial state, horizon and sample grid; failed tails stay NaN and count as failures',
                   rollouts=statuses,
                   errors={name: None if status['failed'] else float(np.mean(np.abs(traces[name][1]-reference[1])))
                           for name, status in statuses.items()},
                   conservative_law_beta_MAE=None if statuses['Conservative LNN']['failed'] else float(np.mean(np.abs(traces['Conservative LNN'][1]-traces['Conservative reference'][1]))),
                   conservative_reference_energy_max_residual=float(np.nanmax(np.abs(total_energy(traces['Conservative reference'])))),
                   physical_dissipation_distinct_from_learning_error=True)
    record_json(out/'metrics.json', metrics)
    np.savez(out/'traces.npz', times=times, **{name.replace(' ', '_'): value for name, value in traces.items()})
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('checkpoint'); parser.add_argument('--out', required=True)
    args = parser.parse_args(); comparison(args.checkpoint, args.out)
