"""Windows/Unix continuation tests using the actual unchanged network and optimizer."""
import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import jax
import numpy as np
from .train import train
from .portable import train_arguments,checkpoint_for,main
from .original_dissipative_comparison import load_original,DEFAULT_SOURCE


class PortableTests(unittest.TestCase):
    def test_archived_reference_is_relative_and_available(self):
        src=load_original()
        self.assertTrue((DEFAULT_SOURCE/'src/hail_mary_model.py').is_file())
        self.assertGreater(src.Parameters().gas_density,0)

    def test_coupled_command_keeps_recipe_and_physics_gate(self):
        arguments=train_arguments('coupled',0,Path('output'),Path('resume.pkl'))
        self.assertEqual(arguments[arguments.index('--steps')+1],'300000')
        self.assertEqual(arguments[arguments.index('--microbatch')+1],'16')
        self.assertIn('--mathematical-benchmark',arguments)
        self.assertIn('--physics-audit',arguments)

    def test_default_study_starts_fresh_without_transfer(self):
        with patch('sys.argv',['portable','study']),patch('conservative_training.portable.run_study') as study:
            main()
            self.assertTrue(study.call_args.args[0].fresh)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(checkpoint_for(Path(tmp),'one-term',0,'main',False))
            self.assertNotIn('--resume',train_arguments('one-term',0,Path(tmp)))

    def test_optimizer_resume_matches_uninterrupted_training(self):
        def args(out,steps,resume=None):
            return argparse.Namespace(model='incomplete',out=str(out),steps=steps,seed=0,microbatch=32,
                                      optimizer='corrected',baseline=False,low_speed=False,profile=True,
                                      resume=None if resume is None else str(resume),mathematical_benchmark=False,physics_audit=None)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)
            train(args(root/'first',1))
            train(args(root/'resumed',2,root/'first/final.pkl'))
            train(args(root/'whole',2))
            a=pickle.loads((root/'resumed/final.pkl').read_bytes())
            b=pickle.loads((root/'whole/final.pkl').read_bytes())
            self.assertEqual(a['step'],2)
            self.assertEqual(a['next_integer_step'],2)
            for x,y in zip(jax.tree.leaves(a['optimizer_state']),jax.tree.leaves(b['optimizer_state'])):
                np.testing.assert_array_equal(x,y)
            self.assertEqual(a['config']['dtype'],'float64')


if __name__=='__main__':unittest.main()
