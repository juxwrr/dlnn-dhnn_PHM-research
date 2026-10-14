"""Dissipative HNN for the fixed-mass relativistic PHM model.

The neural architecture and training recipe follow Sosanya and Greydanus,
"Dissipative Hamiltonian Neural Networks" (2022), and the public reference
implementation at https://github.com/greydanus/dissipative_hnns.

This file is intentionally independent of phm_lnn.py. It uses the canonical
state z=(q,p) derived in the accompanying research document:

    H(q,p) = sqrt(1+p^2) - 1 - q
    q'     = p / sqrt(1+p^2)
    p'     = 1 - kappa*p*abs(p)

Here kappa = K*c^2/(m*a) is the dimensionless drag strength obtained from
F_drag = -K*gamma^2*v*abs(v). The paper's signed dissipative potential is
D(p) = -kappa*abs(p)^3/3, so its gradient supplies the drag field.

The paper's derivative-only loss identifies the summed vector field. For this
problem, a constant field can be moved between H and D by adding harmonic
linear terms, so the raw component split is not uniquely identifiable without
an additional boundary condition or component label. Evaluation reports that
cancellation explicitly and does not add non-paper supervision to hide it.

Examples:
    python phm_dhnn.py --check
    python phm_dhnn.py --train
    python phm_dhnn.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


# Same dimensionless PHM scales used by phm_lnn.py.
C = 300_000_000.0
A = 15.0
MASS = 100_000.0
LIGHT_YEAR = 9.461e15
TRIP_DISTANCE = 11.9 * LIGHT_YEAR
Q_HALF = (TRIP_DISTANCE / 2) / (C**2 / A)
S_HALF = np.sqrt(Q_HALF * (Q_HALF + 2))
P_HALF = S_HALF
DEFAULT_DRAG = 0.02

BLUE = "#2F6BFF"
ORANGE = "#F28E2B"
GREEN = "#2A9D6F"
RED = "#D1495B"
DARK = "#263238"


class MLP(nn.Module):
    """Residual tanh MLP from the authors' public D-HNN implementation."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.lin_1 = nn.Linear(input_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, t: Tensor | None = None) -> Tensor:
        inputs = torch.cat([x, t], dim=-1) if t is not None else x
        hidden = self.lin_1(inputs).tanh()
        hidden = hidden + self.lin_2(hidden).tanh()
        return self.lin_3(hidden)


class DHNN(nn.Module):
    """Paper-faithful D-HNN with separate scalar H and D subnetworks."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.mlp_h = MLP(input_dim, 1, hidden_dim)
        self.mlp_d = MLP(input_dim, 1, hidden_dim)

    def scalars(self, x: Tensor, t: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.mlp_d(x, t=t), self.mlp_h(x, t=t)

    def forward(
        self,
        x: Tensor,
        t: Tensor | None = None,
        as_separate: bool = False,
        create_graph: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not x.requires_grad:
            x = x.requires_grad_(True)

        dissipative_scalar, hamiltonian = self.scalars(x, t=t)
        dissipative = torch.autograd.grad(
            dissipative_scalar.sum(),
            x,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        hamiltonian_gradient = torch.autograd.grad(
            hamiltonian.sum(), x, create_graph=create_graph
        )[0]

        d_h_dq, d_h_dp = torch.split(
            hamiltonian_gradient, hamiltonian_gradient.shape[-1] // 2, dim=1
        )
        conservative = torch.cat([d_h_dp, -d_h_dq], dim=-1)
        if as_separate:
            return dissipative, conservative
        return dissipative + conservative


def beta_from_momentum(p):
    p = np.asarray(p, dtype=float)
    return p / np.sqrt(1.0 + p * p)


def exact_hamiltonian(q, p):
    return np.sqrt(1.0 + np.asarray(p, dtype=float) ** 2) - 1.0 - np.asarray(q, dtype=float)


def exact_dissipative_potential(p, drag: float):
    return -float(drag) * np.abs(np.asarray(p, dtype=float)) ** 3 / 3.0


def analytic_vector_field(
    states: np.ndarray,
    drag: float,
    as_separate: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the exact canonical field in dimensionless time s=a*t/c."""
    states = np.asarray(states, dtype=float)
    p = states[..., 1]
    conservative = np.stack(
        [beta_from_momentum(p), np.ones_like(p)], axis=-1
    )
    dissipative = np.stack(
        [np.zeros_like(p), -float(drag) * p * np.abs(p)], axis=-1
    )
    if as_separate:
        return dissipative, conservative
    return dissipative + conservative


def make_dataset(
    grid_size: int,
    drag: float,
    train_split: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Paper-style shuffled phase-space grid with an 80/20 split."""
    q_grid, p_grid = np.meshgrid(
        np.linspace(0.0, Q_HALF, grid_size),
        np.linspace(-P_HALF, P_HALF, grid_size),
    )
    x = np.stack([q_grid.ravel(), p_grid.ravel()], axis=-1)
    dx_dissipative, dx_conservative = analytic_vector_field(
        x, drag=drag, as_separate=True
    )
    dx = dx_dissipative + dx_conservative

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x, dx = x[order], dx[order]
    dx_dissipative = dx_dissipative[order]
    dx_conservative = dx_conservative[order]

    split = int(len(x) * train_split)
    t = np.zeros_like(x[..., :1])
    return {
        "x": x[:split],
        "x_test": x[split:],
        "t": t[:split],
        "t_test": t[split:],
        "dx": dx[:split],
        "dx_test": dx[split:],
        "dx_dissipative": dx_dissipative[:split],
        "dx_dissipative_test": dx_dissipative[split:],
        "dx_conservative": dx_conservative[:split],
        "dx_conservative_test": dx_conservative[split:],
    }


def get_batch(values: np.ndarray, step: int, batch_size: int, device: str) -> Tensor:
    """Contiguous cyclic batches, matching the reference training code."""
    start = (step * batch_size) % len(values)
    batch = values[start : start + batch_size]
    return torch.tensor(
        batch, requires_grad=True, dtype=torch.float32, device=device
    )


def train(args: argparse.Namespace) -> tuple[DHNN, dict]:
    """Train with the hyperparameters used by the D-HNN synthetic experiment."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = make_dataset(args.grid_size, args.drag, args.train_split, args.seed)
    model = DHNN(input_dim=3, hidden_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    history = {"train_loss": [], "test_loss": [], "test_steps": []}

    model.train()
    latest_test_loss = float("nan")
    for step in range(args.steps):
        x, t, dx = [
            get_batch(data[name], step, args.batch_size, args.device)
            for name in ("x", "t", "dx")
        ]
        prediction = model(x, t=t)
        loss = (dx - prediction).pow(2).mean()
        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()
        history["train_loss"].append(loss.item())

        if step % args.test_every == 0:
            x_test, t_test, dx_test = [
                get_batch(data[name], 0, args.batch_size, args.device)
                for name in ("x_test", "t_test", "dx_test")
            ]
            test_prediction = model(x_test, t=t_test)
            latest_test_loss = (dx_test - test_prediction).pow(2).mean().item()
            history["test_loss"].append(latest_test_loss)
            history["test_steps"].append(step)

        if step % args.print_every == 0 or step == args.steps - 1:
            print(
                f"step {step}, train_loss {loss.item():.3e}, "
                f"test_loss {latest_test_loss:.3e}"
            )

    model = model.cpu()
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "input_dim": 3,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "train_split": args.train_split,
            "steps": args.steps,
            "grid_size": args.grid_size,
            "drag": args.drag,
            "seed": args.seed,
        },
        "history": history,
        "reference": {
            "paper": "Sosanya and Greydanus (2022), arXiv:2201.10085",
            "code": "https://github.com/greydanus/dissipative_hnns",
            "loss": "derivative mean squared error",
        },
    }
    torch.save(payload, checkpoint)
    print(f"saved checkpoint: {checkpoint}")
    return model, payload


def load_checkpoint(path: Path) -> tuple[DHNN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = DHNN(
        input_dim=int(config.get("input_dim", 3)),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def rk4(field, initial_state: np.ndarray, times: np.ndarray) -> np.ndarray:
    states = np.empty((len(times), len(initial_state)), dtype=float)
    states[0] = initial_state
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        state = states[index - 1]
        k1 = field(state)
        k2 = field(state + 0.5 * dt * k1)
        k3 = field(state + 0.5 * dt * k2)
        k4 = field(state + dt * k3)
        states[index] = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return states


def learned_field(model: DHNN):
    def field(state: np.ndarray) -> np.ndarray:
        x = torch.tensor(
            state[None, :], dtype=torch.float32, requires_grad=True
        )
        t = torch.zeros((1, 1), dtype=torch.float32)
        derivative = model(x, t=t, create_graph=False)
        return derivative.detach().numpy()[0].astype(float)

    return field


def evaluate(model: DHNN, payload: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = payload["config"]
    drag = float(config["drag"])
    data = make_dataset(
        int(config["grid_size"]), drag, float(config["train_split"]), int(config["seed"])
    )

    x = torch.tensor(data["x_test"], dtype=torch.float32, requires_grad=True)
    t = torch.tensor(data["t_test"], dtype=torch.float32)
    predicted_dissipative, predicted_conservative = model(
        x, t=t, as_separate=True, create_graph=False
    )
    predicted_dissipative = predicted_dissipative.detach().numpy()
    predicted_conservative = predicted_conservative.detach().numpy()
    predicted_total = predicted_dissipative + predicted_conservative

    conservative_error = predicted_conservative - data["dx_conservative_test"]
    dissipative_error = predicted_dissipative - data["dx_dissipative_test"]
    total_mse = float(np.mean((predicted_total - data["dx_test"]) ** 2))
    conservative_mse = float(
        np.mean(conservative_error**2)
    )
    dissipative_mse = float(
        np.mean(dissipative_error**2)
    )
    cancellation_correlation = float(
        np.corrcoef(conservative_error.ravel(), -dissipative_error.ravel())[0, 1]
    )

    times = np.linspace(0.0, S_HALF, 500)
    truth = rk4(
        lambda state: analytic_vector_field(state[None, :], drag=drag)[0],
        np.array([0.0, 0.0]),
        times,
    )
    learned = rk4(learned_field(model), np.array([0.0, 0.0]), times)
    truth_beta = beta_from_momentum(truth[:, 1])
    learned_beta = beta_from_momentum(learned[:, 1])

    truth_energy = exact_hamiltonian(truth[:, 0], truth[:, 1])
    learned_true_energy = exact_hamiltonian(learned[:, 0], learned[:, 1])
    analytic_energy_rate = -drag * truth_beta * truth[:, 1] * np.abs(truth[:, 1])
    energy_rate_numeric = np.gradient(truth_energy, times)
    energy_balance_rmse = float(
        np.sqrt(np.mean((energy_rate_numeric - analytic_energy_rate) ** 2))
    )

    metrics = {
        "test_total_derivative_mse": total_mse,
        "test_conservative_component_mse": conservative_mse,
        "test_dissipative_component_mse": dissipative_mse,
        "component_error_cancellation_correlation": cancellation_correlation,
        "trajectory_q_rmse": float(np.sqrt(np.mean((learned[:, 0] - truth[:, 0]) ** 2))),
        "trajectory_beta_rmse": float(np.sqrt(np.mean((learned_beta - truth_beta) ** 2))),
        "analytic_energy_balance_rmse": energy_balance_rmse,
        "analytic_max_energy_increase": float(np.max(np.diff(truth_energy))),
        "learned_final_true_energy": float(learned_true_energy[-1]),
        "drag": drag,
        "decomposition_note": (
            "The derivative-only objective identifies the total field; harmonic "
            "terms can move between H and D without changing that field."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    history = payload.get("history", {})
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])
    test_steps = history.get("test_steps", [])
    if train_loss:
        fig, axis = plt.subplots(figsize=(8, 4.6))
        axis.semilogy(train_loss, color=BLUE, alpha=0.65, label="train")
        axis.semilogy(test_steps, test_loss, color=RED, linewidth=2, label="test")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Derivative MSE")
        axis.set_title("D-HNN derivative training")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "loss.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(times, truth[:, 0], color=DARK, linewidth=2.5, label="analytic")
    axes[0].plot(times, learned[:, 0], "--", color=BLUE, linewidth=2, label="D-HNN")
    axes[0].set_ylabel(r"Position $q$")
    axes[0].set_title("Relativistic motion with quadratic canonical drag")
    axes[0].legend(frameon=False)
    axes[1].plot(times, truth_beta, color=DARK, linewidth=2.5, label="analytic")
    axes[1].plot(times, learned_beta, "--", color=ORANGE, linewidth=2, label="D-HNN")
    axes[1].axhline(1.0, color=RED, linestyle=":", label="speed of light")
    axes[1].set_xlabel(r"Dimensionless time $s=at/c$")
    axes[1].set_ylabel(r"Speed $\beta=v/c$")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close(fig)

    p_line = np.linspace(-P_HALF, P_HALF, 401)
    states = np.column_stack([np.full_like(p_line, Q_HALF / 2), p_line])
    exact_dissipative, exact_conservative = analytic_vector_field(
        states, drag=drag, as_separate=True
    )
    x_line = torch.tensor(states, dtype=torch.float32, requires_grad=True)
    t_line = torch.zeros((len(states), 1), dtype=torch.float32)
    learned_dissipative, learned_conservative = model(
        x_line, t=t_line, as_separate=True, create_graph=False
    )
    learned_dissipative = learned_dissipative.detach().numpy()
    learned_conservative = learned_conservative.detach().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(p_line, exact_conservative[:, 0], color=DARK, linewidth=2.5, label=r"exact $q'_H$")
    axes[0].plot(p_line, learned_conservative[:, 0], "--", color=GREEN, linewidth=2, label=r"learned $q'_H$")
    axes[0].set_ylabel(r"Conservative $q'$")
    axes[0].set_title(r"Learned decomposition $J\nabla H + \nabla D$")
    axes[0].legend(frameon=False)
    axes[1].plot(p_line, exact_dissipative[:, 1], color=DARK, linewidth=2.5, label=r"exact $p'_D$")
    axes[1].plot(p_line, learned_dissipative[:, 1], "--", color=RED, linewidth=2, label=r"learned $p'_D$")
    axes[1].set_xlabel("Canonical momentum p")
    axes[1].set_ylabel(r"Dissipative $p'$")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "decomposition.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    return metrics


def check_physics(drag: float = DEFAULT_DRAG) -> None:
    states = torch.tensor(
        [[0.0, -2.0], [Q_HALF / 2, 0.0], [Q_HALF, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    q, p = states[:, 0], states[:, 1]
    hamiltonian = torch.sqrt(1.0 + p**2) - 1.0 - q
    signed_dissipation = -float(drag) * torch.abs(p) ** 3 / 3.0
    grad_h = torch.autograd.grad(hamiltonian.sum(), states, create_graph=True)[0]
    grad_d = torch.autograd.grad(signed_dissipation.sum(), states)[0]
    conservative = torch.stack([grad_h[:, 1], -grad_h[:, 0]], dim=-1)
    expected_dissipative, expected_conservative = analytic_vector_field(
        states.detach().numpy(), drag=drag, as_separate=True
    )
    np.testing.assert_allclose(conservative.detach().numpy(), expected_conservative, atol=1e-12)
    np.testing.assert_allclose(grad_d.detach().numpy(), expected_dissipative, atol=1e-12)

    model = DHNN(input_dim=3, hidden_dim=16)
    x = torch.zeros((5, 2), requires_grad=True)
    t = torch.zeros((5, 1))
    assert model(x, t=t).shape == (5, 2)

    sample_p = np.linspace(-P_HALF, P_HALF, 1001)
    sample_beta = beta_from_momentum(sample_p)
    energy_rate = -drag * sample_beta * sample_p * np.abs(sample_p)
    assert np.max(energy_rate) <= 1e-12
    print("D-HNN architecture and relativistic drag checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-split", type=float, default=0.80)
    parser.add_argument("--test-every", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--drag", type=float, default=DEFAULT_DRAG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("results_dhnn/phm_dhnn.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_dhnn")
    )
    args = parser.parse_args()

    if args.check:
        check_physics(args.drag)
    if args.train:
        model, payload = train(args)
        evaluate(model, payload, args.output_dir)
    if args.evaluate:
        model, payload = load_checkpoint(args.checkpoint)
        evaluate(model, payload, args.output_dir)
    if not any([args.check, args.train, args.evaluate]):
        parser.print_help()


if __name__ == "__main__":
    main()
