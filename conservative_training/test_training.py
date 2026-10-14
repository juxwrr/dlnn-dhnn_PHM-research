"""Small correctness checks for the new trainer, not scientific results."""
from functools import partial
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from .engine import network,acceleration,weighted_error,learning_rate
from . import physics as p
from .train import predictor,source_epoch


class TrainingTests(unittest.TestCase):
    def test_source_custom_initialization(self):
        # Read only the archived, pinned source function, with its original key increment.
        from pathlib import Path
        namespace={}
        text=(Path(__file__).parent/'provenance/core.py').read_text(encoding='utf-8')
        exec('import jax\nimport jax.numpy as jnp\n'+text[text.index('def custom_init'):],namespace)
        parameters,_=network(2,width=16,layers=2)
        shapes=[(jnp.zeros_like(w),jnp.zeros_like(b)) if block else () for block in parameters for w,b in ([block] if block else [(None,None)])]
        expected=namespace['custom_init'](shapes,seed=0)
        for block,reference in zip(parameters,expected):
            if block:
                np.testing.assert_allclose(block[0],reference[0],rtol=1e-14)
                np.testing.assert_array_equal(block[1],reference[1])

    def test_chunked_weighted_gradient(self):
        params,forward=network(2,width=16,layers=2)
        batch=p.sample_incomplete(jax.random.PRNGKey(21),64)
        denom=jnp.sum(1+1/jnp.sqrt(1-batch[1][:,0]**2))
        def objective(params,batch):
            q,v,g,target=batch
            pred=jax.vmap(predictor('incomplete',forward,params))(q,v,g)
            w=1+1/jnp.sqrt(1-v[:,0]**2)
            return jnp.sum(w*jnp.abs(pred-target)[:,0])/denom
        whole=jax.grad(objective)(params,batch)
        chunk1=jax.grad(objective)(params,tuple(z[:32] for z in batch))
        chunk2=jax.grad(objective)(params,tuple(z[32:] for z in batch))
        summed=jax.tree.map(jnp.add,chunk1,chunk2)
        for x,y in zip(jax.tree.leaves(whole),jax.tree.leaves(summed)):
            np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-10)

    def test_source_repeating_velocity(self):
        first=source_epoch(0,batch=1);second=source_epoch(1,batch=1)
        np.testing.assert_array_equal(first[1],second[1])
        self.assertFalse(np.array_equal(first[0],second[0]))
        self.assertFalse(np.array_equal(first[2],second[2]))

    def test_independent_streams_and_speed(self):
        key=jax.random.PRNGKey(0)
        a=p.sample_incomplete(jax.random.fold_in(key,0),512)
        b=p.sample_incomplete(jax.random.fold_in(key,1),512)
        self.assertFalse(np.array_equal(a[1],b[1]))
        data=p.sample_joint(key,16,p.equilibrium())
        self.assertLess(float(jnp.max(jnp.linalg.norm(data[1][:,:3],axis=1))),1.)
        self.assertTrue(np.isfinite(data[3]).all())

    def test_schedule(self):
        self.assertGreater(float(learning_rate(.2)),float(learning_rate(0.)))
        self.assertAlmostEqual(float(learning_rate(1.)),1e-7,places=15)

    def test_zero_force_solution(self):
        result=np.asarray(p.constant_force_solution(jnp.array([0.,.5,1.]),0.,q0=2.,beta0=.4))
        np.testing.assert_allclose(result[:,0],[2.,2.2,2.4])
        np.testing.assert_allclose(result[:,1],.4)


if __name__=='__main__':unittest.main()
