# Hail Mary Lagrangian neural network research

Private transfer package for the two generic conservative LNN studies, with their source notebook, original derivation, research reports, reference implementation and saved progress.

The architecture remains Cranmer's executable SpecialRelativity recipe: **four hidden layers of 500 Softplus units, one scalar Lagrangian, custom initialization, Adam, logical batch 512 and 300,000 updates**. The one-term network has two inputs; the coupled network has 18. Chunk sizes remain 32 and 16 and accumulate into one optimizer update. No extra losses, speed clipping or Hessian regularizers are introduced. PHM float64 and the known external-force treatment remain unchanged.

This is the existing modern JAX port, not a recreation of the original 2020 Python/JAX environment. The original paper methods mention batch 32; the archived executed notebook uses 512. The source control retains its fractional Adam index and float32, while the spacecraft adaptation uses integer Adam updates and float64. The source control has completed its 300,000 updates and passed the diagnostic gate used to start these PHM runs. That gate is not complete scientific acceptance.

## Research limits

The coupled action is authorized as a **mathematical benchmark**. Physical high-speed spacecraft claims remain gated by unresolved clock, field and material conventions. Full training cannot repair those physical assumptions. Original dissipative-system overlays use different hardware and internal time conventions and are explicitly scope comparisons, not controlled coupled accuracy scores.

The portable study preserves five seeds and failed-seed reporting, validation-only checkpoint selection, final tests after training and the original-derivation comparison after a completed checkpoint pair. Interaction ablations, verified multirate mission resolution, internal-energy transfer checks and complete seed aggregation remain additional scientific work.

## Files and provenance

`conservative_training/train.py` is the shared trainer. `physics.py` and `engine.py` are the reference and Euler–Lagrange engine. `portable.py` starts a fresh study by default and can continue checkpoints saved on the new computer. Original dated Mac coordinators are retained for provenance; use the portable commands on the new computer.

The older `phm_lnn.py`, `phm_rotating_lnn.py` and `phm_dhnn.py` are separate earlier experiments. Their architectures and recipes differ from this generic training track. The D-HNN additionally needs `requirements_dhnn.txt` and PyTorch; it is not required for these two LNN runs.

Primary sources: [Cranmer notebook at the pinned commit](https://github.com/MilesCranmer/lagrangian_nns/blob/3b994720c87ebcb4b32a9bae5b09994567a79c2a/notebooks/SpecialRelativity.ipynb), [LNN paper](https://arxiv.org/pdf/2003.04630), [JAX installation](https://docs.jax.dev/en/latest/installation.html). Third-party licenses are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
