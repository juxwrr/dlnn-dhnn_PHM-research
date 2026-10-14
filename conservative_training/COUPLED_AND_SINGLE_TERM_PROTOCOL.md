# Coupled and single term LNN training

The user authorized the full-budget coupled **mathematical benchmark** on October 12 2026. Physical high-speed spacecraft claims remain gated. This supplements TRAINING_PLAN.md without modifying its reference contents.

The existing durable suite runs the pinned source control, then five one-term LNN seeds, five matched direct-acceleration baselines and five low-speed-only models. The newly launched mathematical benchmark suite waits for that source-control gate and the joint 1,000-update smoke result, then runs five coupled LNN seeds, five matched baselines and five low-speed-only models. Each full run uses 300,000 updates. Both coordinators save status and logs and continue outside the chat while this computer stays awake and the processes remain alive. Their code snapshots are separate and frozen. The second queue does not modify or interrupt the source control.

## The original process and the adaptations

The actual executable SpecialRelativity notebook at pinned commit `3b994720c87ebcb4b32a9bae5b09994567a79c2a` uses four 500-unit hidden layers with Softplus and a scalar output, custom initialization, logical batches of 512, gamma-weighted absolute acceleration error, and 100 epochs of 3,000 updates. Its cosine schedule is retained. The paper methods describe batch 32; this work follows the notebook execution rather than combining the two recipes. No energy or trajectory supervision, gradient clipping, Hessian regularizer or speed clipping is added.

The source control preserves the historical float32, conditional-force input and fractional Adam index. The PHM adaptations use float64, correctly differentiate coordinate rescaling, supply known external generalized force, use independent data streams and integer Adam updates with the same learning-rate schedule. The one-term model learns only the standard relativistic kinetic action; the known thrust excites its dynamics and is not a learned potential. The coupled model learns one scalar action for nine coordinates and includes relativistic translation, centered finite-hub internal motion, springs, stellar gravity/radiation, electromagnetic coupling and gravity-gradient energy. Its common coordinate-time interpretation is a mathematical approximation, not a validated covariant spacecraft theory.

Validation seed 1000 and 10,000 states select the checkpoint. Final tests use seed 2000 and 10,000 states only after training; smoke seed 1001 and wiring-verification seed 1002 never select final checkpoints. Main seeds are 0 through 4. Incomplete microbatches contain 32 states; coupled microbatches contain 16. Their gradients accumulate into one weighted 512-state objective and one optimizer update.

## Evaluation and the requested chart

The new coupled evaluator records component errors in four speed bands, reference and learned Hessian rank, 40 held-out rollouts, a short mission case and a bounded attempt at the mission-duration case. Its one-second and ten-second windows resolve internal motion directly with DOP853; this is not a verified multirate mission solver. The direct mission attempt stops after 10,000 RHS evaluations and counts as failed if it cannot reach the horizon. Accepted prefixes are saved; invalid or missing tails stay absent and are marked as failures. It records attitude, extensions, rotor rate, habitat point acceleration, internal energy and true versus learned energy/work residuals. No physical acceptance is asserted until the remaining audit, solver-refinement, interaction and seed-aggregation checks are complete.

`comparisons/one_term_smoke/dissipative_comparison.png` uses the completed one-term 1,000-update checkpoint, clearly marked provisional. All traces start at zero position and zero speed and use the same horizon and grid. The complete **existing one-dimensional** dissipative field is `q' = p/sqrt(1+p²)`, `p' = 1 - 0.02 p |p|`. The chart includes the conservative reference, dissipative reference, conservative LNN and that LNN with known drag supplied externally. It plots speed, position and physical energy including the thrust potential, `H = gamma - 1 - q`. Drag has negative power. Neither the purple-versus-orange discrepancy nor conservation of the network's own energy alone proves whether its conservative law is correct.

The source derivation also includes tether damping/friction, bearing losses, attitude damping, impacts and supplied time-dependent mass. The existing 1D D-HNN field does not contain those terms. A full dissipative nine-coordinate comparison needs their coefficients and a consistent mass/impact prescription. They are not invented for this chart. Full-budget one-term charts are regenerated automatically by the new queue's watcher after each primary seed finishes.

## Current evidence and practical timing

The one-term 1,000-update trial has not met the declared accuracy criteria. A five-update coupled profile is only a wiring check; its test predictions fail. Numerical reference checks and 10 training/comparison tests passed. A new internal-acceleration check confirms one g for the anchored equilibrium geometry and zero mass-weighted internal acceleration. These are code/reference checks, not training success.

Observed single-job profiling suggests about 36 hours per one-term seed and 9.6 days per coupled seed, excluding validation/evaluation and CPU sharing. Five primary coupled seeds alone could take roughly 48 days. Coupled matched baselines and low-speed controls add substantial work. These are rough estimates rather than completion promises. The current source control must finish and pass before either PHM long study begins.

Status: `jobs/status.json` and `jobs/mathematical_benchmark_status.json`.

Remaining scientific work includes complete trained results, failed-seed aggregation, solver refinement and verified multirate mission resolution, axial interaction ablations, nominal coupled interaction identifiability and the physical proper-time/field/material audit. Queuing training does not complete those requirements.

## Primary sources

- [Lagrangian Neural Networks paper](https://arxiv.org/pdf/2003.04630)
- [Original executable SpecialRelativity notebook](https://github.com/MilesCranmer/lagrangian_nns/blob/3b994720c87ebcb4b32a9bae5b09994567a79c2a/notebooks/SpecialRelativity.ipynb)
- Pinned copies and hashes: `provenance/`.
- Read-only project derivation: `/Users/junxiwu/Downloads/ISEF PROJECT FINAL/PHM_Derivations.pdf`, particularly sections 7 and 12 through 14.

## Original derivation comparison selected by the user

The user's follow-up selected PHM_Derivations.pdf rather than the earlier 0.02-drag illustration. The primary report comparison is now in `comparisons/original_derivation_report/`. It reads the original PDF and the archived `hail_mary_physics_complete/src/hail_mary_model.py` and parameter assumptions without modifying them. The regenerated original reference includes changing mass and propulsion, stellar gravity/radiation, nominal gas/dust drag, cable viscous/Coulomb losses, bearing viscous/Coulomb losses, attitude damping/control and the original prescribed internal impacts. Nominal EM and translational impacts remain zero as in that archived scenario.

The reference's original hardware differs from the coupled LNN hardware, and its internal clock is proper time. The internal overlay is therefore a scope comparison, not a controlled learning-error score. The one-term neural comparison is given the original mass and thrust histories; it does not learn fuel or propulsion. The original reference partitions travel and internal dynamics; no continuous multiyear full-coupling claim is made. Dry friction is regularized near zero exactly as in the archived numerical implementation and the report discloses this difference from the PDF's nonsmooth sign law.

An independent coordinate-time travel calculation agrees with the original proper-time reference to a maximum beta difference of 1.75e-6c and a maximum position difference of 8.36e-8 of the route. Rayleigh-gradient and nonpositive-loss-power tests pass. The provisional one-term external-field rollout fails when its path crosses a star; the coupled five-update profile fails near the beginning with a nonpositive radius. Missing tails are retained as failures.

The separate one-study postprocessor `original_comparison_worker.py` waits for each pair of completed primary full-budget checkpoints and generates the same original-reference figures for seeds 0 through 4. It neither changes the running training snapshots nor chooses checkpoints from these comparisons. Its status is `jobs/original_comparison_status.json`. It stops with an explicit record if a primary training suite is blocked or fails. The earlier toy comparison remains archived as a separate diagnostic and is no longer the requested report figure.
