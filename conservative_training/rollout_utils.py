"""Bounded integrations retain the accepted prefix and report failures explicitly."""
import numpy as np
from scipy.integrate import DOP853


def integrate(rhs, initial, times, *, rtol=1e-9, atol=1e-11, max_calls=10000):
    times = np.asarray(times)
    values = np.full((len(initial), len(times)), np.nan)
    values[:, 0] = initial
    calls = 0
    last_accepted = float(times[0])
    index = 1
    def bounded(t, y):
        nonlocal calls
        calls += 1
        if calls > max_calls:
            raise FloatingPointError(f'RHS evaluation budget {max_calls} exceeded at t={t}')
        if not np.isfinite(y).all():
            raise FloatingPointError(f'Nonfinite state at t={t}')
        derivative = np.asarray(rhs(t, y), dtype=float)
        if not np.isfinite(derivative).all():
            raise FloatingPointError(f'Nonfinite derivative at t={t}')
        return derivative
    failure = None
    try:
        solver = DOP853(bounded, times[0], np.asarray(initial), times[-1], rtol=rtol, atol=atol)
        while solver.status == 'running':
            message = solver.step()
            if solver.status == 'failed':
                raise FloatingPointError(message)
            last_accepted = float(solver.t)
            dense = solver.dense_output()
            while index < len(times) and times[index] <= solver.t:
                values[:, index] = dense(times[index])
                index += 1
        if index != len(times):
            raise FloatingPointError('Solver ended before the requested final sample')
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        failure = str(error)
    return values, dict(failed=failure is not None, reason=failure, last_accepted_time=last_accepted,
                       completed_samples=index, requested_samples=len(times), rhs_calls=calls,
                       method='DOP853', rtol=rtol, atol=atol, max_rhs_calls=max_calls)
