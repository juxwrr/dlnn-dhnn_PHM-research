# Conservative LNN execution report

Started October 12, 2026 (America/Phoenix). The user has authorized execution of TRAINING_PLAN.md and confirmed that long physical full-model training must remain gated. This report is provisional; no complete two-model scientific result has been obtained.

## What is implemented and checked

The new shared trainer uses four hidden layers of 500 Softplus units, a scalar neural action, a logical batch of 512 states, the source initialization, weighted absolute acceleration error, and the specified cosine schedule over 300,000 updates. Computational chunks of 32 states (16 for joint debugging) accumulate one weighted gradient before one Adam update. The chunked gradient was checked against the whole-batch gradient. PHM uses float64 and integer Adam steps; the source control uses float32 and fractional Adam progress. No clipping, Hessian regularization, energy supervision, or trajectory supervision was added.

Checkpoints contain the optimizer state, deterministic RNG state/position, configuration and validation score. Validation uses 10,000 fixed states with seed 1000; final tests use 10,000 separate states with seed 2000. Smoke evaluation uses seed 1001 and does not access the final test. Resuming the optimizer produced bitwise identical parameters and moments to an uninterrupted run. Existing checkpoints, reports, and synced project files were preserved.

The incomplete action is the relativistic translation-only action with known thrust external. The joint implementation is a nine-coordinate **effective mathematical benchmark**, with 3D translation, ZYX attitude coordinates, two extensions and a relative spin angle. Body angular velocity is computed from attitude coordinate rates. One action includes centered hub/point-mass kinematics, internal velocity cross terms, stellar gravity/radiation, electromagnetic coupling and position-dependent gravity-gradient energy.

The benchmark uses one coordinate time and one energy scale. It has not passed the physical proper-time/frame audit and cannot support a physical high-speed flexible-spacecraft claim.

## Reference validation

The authoritative numerical audit is [physics_audit.json](audit_complete/physics_audit.json). Earlier audits are retained as development records; the first audit's torque-sign concern was corrected by an explicit virtual-work check.

| Check | Result |
| --- | ---: |
| Incomplete explicit vs automatic acceleration | maximum error 7.11e-15 |
| Incomplete numerical vs exact constant-force trajectory | maximum state error 2.51e-11 |
| Incomplete true energy/work identity | maximum residual 4.42e-10 |
| Vector relativistic Lorentz/stellar-force limit | maximum error 7.11e-15 m/s² |
| Reduced five-coordinate axial/roll prototype | maximum acceleration error 7.53e-16 |
| Anchored-hub isolated PDF centrifuge limit | maximum error 1.44e-15 |
| Centered vs uncentered kinetic-energy identity | 0 J error in the tested state |
| Internal-action directional finite differences | best relative error 4.78e-11 |
| Gravity-gradient potential vs printed torque | maximum error 3.97e-23 N m |
| Isolated angular-momentum derivative | maximum magnitude 1.75e-10 N m |

Hub and rotor momentum changes were separately nonzero and opposite in the isolated test. The corrected equilibrium was recomputed, yielding 5.4455819 rpm and extensions 0.1560777 m and 0.1040518 m. The old value happens to remain valid at this balanced equilibrium; it was not forced onto the correction.

Natural mission/internal coordinate scales produced an exact-action Hessian condition number near 3e17, and the default pseudoinverse retained only six modes in a tested low-speed state. Applying nominal mass-metric **coordinate scales to the whole action** reduced that condition number to about 1376 and retained all nine modes. The balanced default-pseudoinverse acceleration agreed with the independent preconditioned solve to a maximum scaled discrepancy of 5.72e-11 in the tested batch. No relative subsystem energy weights were changed.

The published benchmark domain and actual scales are in the audit. It includes the declared 50/25/25 uniform/near-star position mixture, positive star distances, bounded nonsingular attitudes, independently excited extensions and rates, and force strata of 50% synthetic vector excitation, 25% zero thrust and 25% axial mission thrust. These force strata deliberately depart from the faithful control to expose internal signals.

## Smoke tests and runtime

| Experiment | Completed updates | Fixed validation weighted error | Interpretation |
| --- | ---: | ---: | --- |
| Source control, faithful fractional Adam, float32 | 1,000 | 0.419521 | finite smoke test; not a scientific result |
| Source control, corrected integer Adam, float32 | 1,000 | 0.423877 | approximately 1% difference from faithful smoke |
| Incomplete PHM, float64 | 1,000 | 0.815989 | finite smoke test; not a scientific result |
| Balanced nine-coordinate benchmark | 5 | not evaluated in profile | finite profile only; physical training gated |

The source smoke's 40 held-out rollouts and the notebook's seed-4 rollout remained finite. Their errors do not yet establish reproduction. The incomplete smoke failed its accuracy targets and had five rollout failures; these are retained in [smoke metrics](runs/incomplete_smoke_seed0/smoke_evaluation/metrics.json). A finite loss is not evidence of a correctly learned law.

On the available CPU, the faithful float32 source smoke projected roughly 9 hours per 300,000-update run. The incomplete smoke projected roughly 36 hours per run, or about 7.5 days for five seeds before evaluation and baselines. The coupled profile projected roughly 9.6 days per seed. These are short-run estimates, not completion promises.

An earlier source attempt accidentally promoted its parameters to float64. It was interrupted, preserved, marked invalid as a faithful control, and replaced with a fresh verified float32 run. Early failed joint profiles and diagnostic logging failures are likewise retained rather than presented as successful training.

## Execution and gates

[Live status](LIVE_STATUS.md) and [machine-readable status](jobs/status.json) record the running process, current task and log. The durable runner first completes the full source-control budget and evaluates a checkpoint selected solely on validation. Its predeclared diagnostic gate is fixed-validation weighted acceleration MAE below 0.5 and notebook-seed-4 speed MAE below 0.05c; these are project diagnostic gates, not published Cranmer tolerances or the PHM acceptance targets.

If that control is healthy, the runner executes the 1,000-update joint benchmark smoke, five independent full-budget incomplete PHM initializations, a matched five-seed direct acceleration baseline, and five separate low-speed-only incomplete models. Every main run retains the final and validation-selected checkpoints. A numerical failure is retained and does not silently remove that seed from the protocol.

Long physical full-model training remains gated. The supplied action does not define a common proper-time/coordinate-time flexible-body model with consistent field/attitude transport. For example, converting a proper-time quadratic internal action requires factors of dτ/dt in both kinetic and potential terms; simply using the coordinate-time sum does not perform that conversion. Cable Young's modulus alone also does not establish allowable elastic stress. The finite-hub model agrees with the isolated PDF equations in the anchored-hub limit; suppressing hub angular velocity alone does not remove its translational recoil.

Full-model physical training, matched coupled baselines, full-model extrapolation, coupled stiff/multirate rollouts, the matched axial interaction-ablation ladder, and paired interaction measurements remain outstanding. They will not be represented as completed by this provisional execution report.

## Source provenance

The source files are archived at commit [3b994720c87ebcb4b32a9bae5b09994567a79c2a](https://github.com/MilesCranmer/lagrangian_nns/tree/3b994720c87ebcb4b32a9bae5b09994567a79c2a). The source [notebook](https://github.com/MilesCranmer/lagrangian_nns/blob/3b994720c87ebcb4b32a9bae5b09994567a79c2a/notebooks/SpecialRelativity.ipynb), builder and initialization are preserved with hashes. The archived code-cell indices/execution counts and the port's selected cell order are recorded in provenance/cell_order.json. This is a training-cell port on modern JAX 0.11.2, not execution of the entire historical notebook or a byte-for-byte historical-environment reproduction. The notebook's later external checkpoint load and out-of-order execution counts are explicitly excluded.

The control target is g(1−β²)^(5/2)/(1+2β²). The incomplete PHM target is g(1−β²)^(3/2). The PHM equation was not replaced with the notebook target.

## Research document replacement

The conservative final DOCX was rebuilt on October 12, 2026 using the current action, audit, smoke evidence and a training snapshot at 17:33 Phoenix time. All old embedded images and earlier separate-network performance claims were removed. The replacement has 10 pages and native editable equations; all pages were rendered and visually inspected. The original is retained outside the final directory in report_archive. Training is still ongoing and physical full-model long training remains gated.
