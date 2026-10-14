"""Shared scalar-Lagrangian engine and source-compatible generic network."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import stax

jax.config.update("jax_enable_x64", True)


def acceleration(action, q, velocity, force):
    mass = jax.hessian(action, 1)(q, velocity)
    mixed = jax.jacfwd(jax.grad(action, 1), 0)(q, velocity)
    return jnp.linalg.pinv(mass) @ (jax.grad(action, 0)(q, velocity) + force - mixed @ velocity)


def energy(action, q, velocity):
    return jnp.dot(jax.grad(action, 1)(q, velocity), velocity) - action(q, velocity)


def diagnostics(action, q, velocity):
    hessian = jax.hessian(action, 1)(q, velocity)
    singular = jnp.linalg.svd(hessian, compute_uv=False)
    return dict(min_singular=singular[-1], max_singular=singular[0],
                condition=singular[0] / singular[-1], eigenvalues=jnp.linalg.eigvalsh(hessian),
                retained_rank=jnp.linalg.matrix_rank(hessian))


def network(inputs, outputs=1, seed=0, width=500, layers=4, dtype=jnp.float64):
    blocks = []
    for _ in range(layers):
        blocks.extend([stax.Dense(width), stax.Softplus])
    blocks.append(stax.Dense(outputs))
    initialize, forward = stax.serial(*blocks)
    _, parameters = initialize(jax.random.PRNGKey(seed), (-1, inputs))
    # Exact pinned lnn/core.py custom_init key progression, including rng += 1.
    key = jax.random.PRNGKey(seed)
    count = sum(bool(block) for block in parameters)
    index, result = 0, []
    for block in parameters:
        if not block:
            result.append(())
            continue
        weight, bias = block
        n = max(weight.shape)
        factor = 2.2 if index == 0 else n if index == count - 1 else 0.58 * index
        result.append((jax.random.normal(key, weight.shape, dtype=dtype) * factor / np.sqrt(n),
                       jnp.zeros_like(bias, dtype=dtype)))
        key += 1
        index += 1
    return result, forward


def learning_rate(progress):
    u = 0.3 + 0.7 * progress
    return 1e-7 + (1e-3 - 1e-7) * (1 - jnp.cos(2 * jnp.pi * u)) / 2


def weighted_error(predictions, targets, beta, scales):
    weight = 1 + 1 / jnp.sqrt(1 - beta**2)
    error = jnp.mean(jnp.abs(predictions - targets) / scales, axis=-1)
    return jnp.sum(weight * error) / jnp.sum(weight)
