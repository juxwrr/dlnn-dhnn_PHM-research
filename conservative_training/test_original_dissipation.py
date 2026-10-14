"""Checks against the archived implementation of the original derivation."""
import unittest
import numpy as np
from .original_dissipative_comparison import load_original


class OriginalDissipationTests(unittest.TestCase):
    def test_rayleigh_gradient_recovers_original_drag(self):
        src=load_original();p=src.Parameters()
        for beta in [-.95,-.5,-.05,.05,.5,.95]:
            v=beta*p.c;h=p.c*1e-6
            derivative=(src.drag_rayleigh(v+h,p)-src.drag_rayleigh(v-h,p))/(2*h)
            expected=-src.drag_force(np.arctanh(beta),p)
            np.testing.assert_allclose(derivative,expected,rtol=2e-8,atol=1e-15)

    def test_original_loss_terms_have_nonpositive_power(self):
        src=load_original();p=src.Parameters();rng=np.random.default_rng(170)
        _,_,_,ch=src.cable_properties(p,'h');_,_,_,cc=src.cable_properties(p,'c')
        for _ in range(32):
            beta=rng.uniform(-.999,.999);vh,vc,spin=rng.normal(size=3);w=rng.normal(size=3)
            contributions=[src.drag_force(np.arctanh(beta),p)*beta*p.c,
                           -ch*vh**2-p.radial_coulomb*vh*np.tanh(vh/1e-4),
                           -cc*vc**2-p.radial_coulomb*vc*np.tanh(vc/1e-4),
                           -p.bearing_damping*spin**2-p.bearing_coulomb*spin*np.tanh(spin/1e-4),
                           -p.attitude_kd*np.dot(w,w)]
            self.assertTrue(np.isfinite(contributions).all())
            self.assertTrue(np.all(np.array(contributions)<=0))


if __name__=='__main__':unittest.main()
