"""Independent checks of drag, energy accounting, rollout failure, and audit gating."""
import argparse
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .dissipative_comparison import drag_force
from .engine import acceleration, energy
from . import physics as p
from .rollout_utils import integrate
from .train import train


class ComparisonTests(unittest.TestCase):
    def test_drag_energy_identity_and_canonical_equivalence(self):
        for beta in [-.999, -.8, 0., .3, .95]:
            momentum=beta/np.sqrt(1-beta*beta)
            force=1+float(drag_force(beta))
            acc=float(acceleration(p.incomplete_action,jnp.array([.4]),jnp.array([beta]),jnp.array([force]))[0])
            # dp/dbeta=gamma^3. This independently checks the canonical drag field.
            self.assertAlmostEqual(acc/(1-beta*beta)**1.5,1-.02*momentum*abs(momentum),places=9)
            H=lambda z: energy(p.incomplete_action,z[:1],z[1:])-z[0]
            edot=float(jnp.dot(jax.grad(H)(jnp.array([.4,beta])),jnp.array([beta,acc])))
            self.assertAlmostEqual(edot,float(drag_force(beta))*beta,places=9)
            self.assertLessEqual(float(drag_force(beta))*beta,0.)

    def test_failed_prefix_is_explicit_and_tail_stays_missing(self):
        def field(t,y):
            if t>.4: raise FloatingPointError('known test failure')
            return [1.]
        values,status=integrate(field,[0.],np.linspace(0,1,101),max_calls=1000)
        self.assertTrue(status['failed'])
        count=status['completed_samples']
        self.assertGreater(count,1)
        self.assertTrue(np.isnan(values[:,count:]).all())
        np.testing.assert_allclose(values[0,:count],np.linspace(0,1,101)[:count],atol=1e-12)

    def test_physical_long_run_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'run'
            args=argparse.Namespace(model='full_effective',steps=300000,out=str(target),mathematical_benchmark=False)
            with self.assertRaises(RuntimeError):train(args)
            self.assertFalse(target.exists())

    def test_benchmark_requires_successful_numerical_audit(self):
        args=argparse.Namespace(model='full_effective',steps=300000,mathematical_benchmark=True,physics_audit=None)
        with self.assertRaises(ValueError):train(args)


if __name__=='__main__':unittest.main()
