# AMD Radeon RX 7900 XT

The project stays in JAX to preserve the original notebook's architecture and Euler–Lagrange autodiff method. No PyTorch rewrite, float32 conversion, smaller network or shorter training budget is silently substituted.

## Native Windows

Use `Setup-Windows.ps1` and `Start-Training.ps1` for CPU training. JAX's native Windows package does not use your AMD GPU. A faster CPU can still improve speed, but the card will not accelerate that route.

## AMD GPU route

JAX's ROCm GPU backend runs under Linux. AMD lists the RX 7900 XT in its WSL ROCm GPU matrix, but framework support must be checked separately: JAX-on-WSL is experimental, and the Radeon guides have limitations on validated JAX workloads. These checks are particularly important here because LNN training differentiates through Hessians and pseudoinverses in float64.

Use AMD's current instructions for a **matching ROCm, driver and JAX environment**:

- [Radeon WSL GPU matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
- [AMD JAX installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html)
- [JAX ROCm installation and platform table](https://docs.jax.dev/en/latest/installation.html)

For a fully documented AMD JAX path, Linux on this computer is the better-established route. WSL may work, but it must pass the checks below on your card. The Linux Docker commands that map `/dev/kfd` and `/dev/dri` must not be copied unchanged into WSL, which has a different GPU interface.

The transferred CPU environment uses JAX/JAXlib **0.11.2**. AMD's published supported combinations may use a different JAX version, such as 0.11.1. Keep the AMD-supported JAX/plugin pair together; do not blindly upgrade JAX alone or apply `requirements.txt` over a provider-managed GPU environment. Record that environment change and verify continuation rather than claiming a historical environment reproduction.

After installing the matching AMD environment, clone this repository inside the Linux filesystem and install the non-JAX dependencies:

```bash
python -m pip install numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2 psutil tzdata
python -m conservative_training.portable check --device gpu
python -m unittest conservative_training.test_training conservative_training.test_comparisons conservative_training.test_original_dissipation conservative_training.test_portable
python -m conservative_training.audit --out verification_runs/amd_audit
python -m conservative_training.portable profile --model coupled --device gpu --out verification_runs/amd_profile
```

The device check requests `rocm` explicitly and fails rather than silently falling back to CPU. It must list your Radeon device. The unit tests exercise the actual network, gradients and optimizer continuation. The audit tests the physical reference; the profile exercises third-order training differentiation, the pseudoinverse and float64. Passing import/device discovery alone is not enough.

Start both models from scratch (no old checkpoints required):

```bash
python -m conservative_training.portable study --model both --device gpu --out training_results
```

Measured performance on your card remains unknown until this profile runs. GPU or CPU migration does not alter the mathematical interpretation of the coupled model or resolve its physical audit. This repository has Windows/Linux CPU verification; AMD hardware verification requires your actual PC.
