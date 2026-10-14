# Hail Mary Lagrangian neural network research

Private transfer package for the two generic conservative LNN studies, with their source notebook, original derivation, research reports, reference implementation and saved progress.

The architecture remains Cranmer's executable SpecialRelativity recipe: **four hidden layers of 500 Softplus units, one scalar Lagrangian, custom initialization, Adam, logical batch 512 and 300,000 updates**. The one-term network has two inputs; the coupled network has 18. Chunk sizes remain 32 and 16 and accumulate into one optimizer update. No extra losses, speed clipping or Hessian regularizers are introduced. PHM float64 and the known external-force treatment remain unchanged.

This is the existing modern JAX port, not a recreation of the original 2020 Python/JAX environment. The original paper methods mention batch 32; the archived executed notebook uses 512. The source control retains its fractional Adam index and float32, while the spacecraft adaptation uses integer Adam updates and float64. The source control has completed its 300,000 updates and passed the diagnostic gate used to start these PHM runs. That gate is not complete scientific acceptance.

## Start on Windows

1. Install **Python 3.12** from [Python](https://www.python.org/downloads/windows/). Download this repository (Code → Download ZIP) or clone it. Open PowerShell in the project folder.
2. Run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\Setup-Windows.ps1
   ```

3. Open [Releases](https://github.com/juxwrr/hail-mary-lnn-research/releases), download `training-transfer.zip`, put it in this folder and extract it:

   ```powershell
   Expand-Archive .\training-transfer.zip -DestinationPath .
   .\.venv\Scripts\python.exe -m conservative_training.portable check --device cpu --transfer
   ```

4. Continue the five main seeds of both models:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\Start-Training.ps1
   ```

   To run only the coupled model, use `-Model coupled`. To run only seed zero, add `-Seeds 0`. To also run all matched baselines and low-speed studies, add `-Controls`.

The script uses native Windows **CPU** JAX. It does not pretend to use the Radeon GPU. For your RX 7900 XT, see [AMD GPU setup](AMD_GPU.md). Native Windows GPU support for JAX is not available; an AMD Linux/WSL ROCm environment must be verified separately.

Outputs and logs go to `training_results/`. Running the same command again resumes its newest committed checkpoint. Ctrl+C asks the trainer to save an interrupted checkpoint; allow it to finish writing before closing the terminal. The default setup runs one experiment at a time to avoid competing for device memory.

## What is transferred

- One-term seed zero: resume at **60,000** updates, matching best checkpoint at **56,000**.
- Coupled mathematical seed zero: resume at **20,000** updates, matching best checkpoint at **17,000**.
- Original immutable validation data, optimizer state, RNG position, fixed acceleration scales and completed source-control evidence.
- Architecture, physics reference and gradient/resumption tests.
- `reports/`: the final Word reports, with clearly labelled preliminary results.
- `references/`: the original derivation PDF and its deterministic companion source, parameters and data.
- `evidence/`: existing results and charts.
- `research-archive.zip` in Releases: full historical code, runs, saved checkpoints, frozen execution snapshots, report work and original reference material. It excludes the machine-specific virtual environment and transfer staging directory. Extract it into a separate folder for archival use; it is not the portable runnable code.

Checkpoint weights are in release assets so the Git history remains practical to clone. `SHA256SUMS` verifies the downloaded assets. `transfer/manifest.json` also verifies each resumable state and its corresponding best checkpoint and validation split.

The Mac training is preserved and continues independently. These are committed snapshots, not a synchronized live transfer. Device kernels can introduce floating-point differences after resumption; changing hardware does not guarantee bit-for-bit continuation. Every invocation records devices, dtype, package versions and source hashes. Stored coordinate scales and validation data are retained.

## Check before a long run

```powershell
.\.venv\Scripts\python.exe -m unittest conservative_training.test_training conservative_training.test_comparisons conservative_training.test_original_dissipation conservative_training.test_portable
.\.venv\Scripts\python.exe -m conservative_training.audit --out verification_runs\audit
.\.venv\Scripts\python.exe -m conservative_training.portable profile --model coupled --device cpu --out verification_runs\coupled_profile
```

A five-update profile measures compatibility and speed, not scientific success. Its learning-rate schedule still spans 300,000 updates. Compare its timing on your actual hardware before estimating total training time.

## Research limits

The coupled action is authorized as a **mathematical benchmark**. Physical high-speed spacecraft claims remain gated by unresolved clock, field and material conventions. Full training cannot repair those physical assumptions. Original dissipative-system overlays use different hardware and internal time conventions and are explicitly scope comparisons, not controlled coupled accuracy scores.

The portable study preserves five seeds and failed-seed reporting, validation-only checkpoint selection, final tests after training and the original-derivation comparison after a completed checkpoint pair. Interaction ablations, verified multirate mission resolution, internal-energy transfer checks and complete seed aggregation remain additional scientific work.

## Files and provenance

`conservative_training/train.py` is the shared trainer. `physics.py` and `engine.py` are the reference and Euler–Lagrange engine. `portable.py` continues the transferred study. Original dated Mac coordinators are retained for provenance; use the portable commands on the new computer.

The older `phm_lnn.py`, `phm_rotating_lnn.py` and `phm_dhnn.py` are separate earlier experiments. Their architectures and recipes differ from this generic training track. The D-HNN additionally needs `requirements_dhnn.txt` and PyTorch; it is not required for these two LNN runs.

Primary sources: [Cranmer notebook at the pinned commit](https://github.com/MilesCranmer/lagrangian_nns/blob/3b994720c87ebcb4b32a9bae5b09994567a79c2a/notebooks/SpecialRelativity.ipynb), [LNN paper](https://arxiv.org/pdf/2003.04630), [JAX installation](https://docs.jax.dev/en/latest/installation.html). Third-party licenses are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
