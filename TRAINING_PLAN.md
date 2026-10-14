# Conservative PHM LNN: controlled reproduction and coupled-model training plan

Status: plan only. No training is authorized or performed by this document. Existing checkpoints and reports remain unchanged.

## 1. What we are trying to reproduce

Reproduce Cranmer's *method and tests*, then test whether they work on the PHM Lagrangians. Changing the Lagrangian changes the target dynamics, so matching the paper's numerical results is not something we can promise.

There will be two primary PHM experiments:

| Experiment | Physical model | What the network learns |
| --- | --- | --- |
| Incomplete conservative | Only relativistic center-of-mass translation; fixed mass; known external thrust for excitation | One generic scalar Lagrangian of position and velocity |
| Full conservative | Translation, electromagnetism, attitude, gravity-gradient interactions, tether extensions, and centrifuge spin in **one coupled model** | One scalar Lagrangian for the joint state, not three independently trained models |

Here “incomplete” means deliberately omitting environmental and internal terms. It does not mean the relativistic kinetic term is mathematically incorrect. Neither experiment includes drag, damping, impacts, changing mass, exhaust dynamics, or bearing friction.

The current axial code uses a generic network, but the existing attitude and centrifuge experiments learn separate potentials with known kinetic energies and different clocks. Those are useful component tests, not the full coupled experiment proposed here.

## 2. Establish a notebook control before changing the physics

Run a small source-faithful control using the original relativistic-particle notebook, followed by its full training budget if the control is healthy. Archive the repository commit, dependency versions, executed cell order, initialization, seeds, optimizer state, and output metrics. Pin a commit before execution; a moving `master` URL is not sufficient for reproducibility.

The notebook's analytic target is

\[
\ddot q = g\frac{(1-v^2)^{5/2}}{1+2v^2}.
\]

This is different from the standard relativistic kinetic term in the PHM model, whose one-dimensional response to force is proportional to \((1-v^2)^{3/2}\). Do not replace the PHM equation with the notebook equation just to obtain a similar plot.

The control should evaluate both the derivative fit and the rollout. It establishes that our implementation can recover the behavior of the source experiment before the PHM model is substituted. It is a diagnostic control, not a third spacecraft model.

### Settings to carry over

Use four hidden layers of 500 units, Softplus activations, a scalar output, Cranmer's custom initialization, minibatches of 512, Adam, and 300,000 updates: 100 epochs of 3,000 minibatches. The architecture interpretation comes from the actual `extended_mlp` implementation. The notebook's executed batch size is 512; the paper's methods summary says 32, and some notebook configuration fields are unused. Do not silently mix those settings.

Use the notebook's normalized, high-speed-weighted absolute acceleration loss and its cosine schedule. For training progress \(p\in[0,1]\),

\[
u=0.3+0.7p,\qquad
\mathrm{lr}(p)=10^{-7}+(10^{-3}-10^{-7})\frac{1-\cos(2\pi u)}{2}.
\]

This schedule briefly rises to its peak and then decays. Calling it a simple monotonic decay would be inaccurate.

### Source quirks to preserve only in the faithful control

- The training sampler uses a fixed velocity random key, so its velocity sequence repeats across epochs while position and force samples change.
- The notebook passes fractional progress to Adam's update function. In the JAX optimizer implementation inspected locally, that argument also controls moment bias correction, not just the learning rate.
- Its equation wrapper rescales position and force inside the derivative calculation. The PHM adaptation must instead differentiate a correctly transformed action, including all chain-rule factors.

For the PHM runs, use independent random streams and an integer Adam step counter, with the learning-rate function separately converting the integer step to progress. Record these as deliberate departures. Compare corrected and source-faithful optimization on the notebook control if their outcomes differ materially. Do not present a corrected adaptation as a byte-for-byte replication.

## 3. Audit the incomplete PHM physics

Use the physical Lagrangian

\[
L_{\mathrm{inc}}=-mc^2\sqrt{1-\dot x^2/c^2}.
\]

With \(q=ax/c^2\), \(s=at/c\), and \(\beta=v/c\), an equivalent dimensionless form, after adding a constant rest offset, is

\[
\ell_{\mathrm{inc}}=1-\sqrt{1-\beta^2}.
\]

Engine thrust remains a known external generalized force, consistent with the supplied derivation. For dimensionless force \(g=T/(ma)\), the reference equations are

\[
q'=\beta,\qquad \beta'=g(1-\beta^2)^{3/2}.
\]

For constant force and initial rest, use the independent exact solution

\[
\beta(s)=\frac{gs}{\sqrt{1+(gs)^2}},\qquad
q(s)=\frac{\sqrt{1+(gs)^2}-1}{g},
\]

with the continuous zero-force limit handled separately. Use these formulas to validate generated labels and the numerical reference solver, not as training labels for trajectories.

A free particle with thrust set to zero is a necessary check but an insufficient training experiment: straight-line trajectories alone cannot identify the relativistic kinetic law. Known applied forces provide the excitation needed to test that law.

## 4. Rebuild the full experiment as one coupled model

This is a required implementation milestone, not an optional later improvement.

### Joint state and action

The target full state has nine generalized coordinates: three center-of-mass coordinates, three local attitude coordinates, two tether extensions, and one relative centrifuge angle. Supply the matching nine coordinate velocities. Use a nonsingular local attitude chart for bounded attitude tests and chart transitions for wider rotations; do not treat the three components of angular velocity as if they were coordinate derivatives.

Start with the Section 13 action:

\[
\begin{aligned}
L_{\mathrm{full}}={}&-mc^2\sqrt{1-|\dot{\mathbf r}|^2/c^2}
+\frac{GM_sm}{r_s}+\frac{GM_Tm}{r_T}
-\frac{K_s}{r_s}-\frac{K_T}{r_T}\\
&+Q\mathbf A\cdot\dot{\mathbf r}-Q\phi
+\boldsymbol\mu_b\cdot\mathbf B
+\tfrac12\boldsymbol\omega^T I\boldsymbol\omega-U_{gg}\\
&+\sum_{j=h,c}\left[\tfrac12M_j\dot\xi_j^2
+\tfrac12M_j(r_j+\xi_j)^2\dot\vartheta^2
-\tfrac12k_j\xi_j^2\right].
\end{aligned}
\]

Use vector distances to the two stationary stars in the three-dimensional model; the document's axial distance formulas are recovered on the travel axis. For a stationary uniform magnetic field, a possible fixed gauge is \(\mathbf A=\tfrac12\mathbf B\times\mathbf r\), with \(\phi=-\mathbf E\cdot\mathbf r\) for a uniform electric field. Rotate the body-fixed dipole into the field frame before evaluating its energy. Holding the dipole direction fixed while rotating the ship would erase magnetic attitude coupling.

Derive the dynamics from the entire action at once. Do not concatenate the old component accelerations. In particular, \(U_{gg}(\mathbf r,R)\) affects both attitude torque and center-of-mass force; the position derivative of this term is absent from the old reduced axial labels and must be included in joint labels.

### Correct the missing hub/centrifuge kinematic coupling

Merely adding the PDF's separate attitude and centrifuge kinetic energies does not describe exchange of angular momentum between the hub and spinning masses. If \(\vartheta\) is a relative angle in the rotating hub, derive the point-mass velocities from body-frame locations \(\rho_j(\xi_j,\vartheta)\). Their internal velocities contain both

\[
\boldsymbol\omega\times\rho_j\quad\text{and}\quad\dot\rho_j.
\]

Squaring their sum gives the cross terms that the separated experiment lacks. Define the translational coordinate at the total center of mass, recenter the hub and attached masses when unequal extensions shift that center, and avoid counting those masses twice in the rotational inertia. Require this derivation to reduce to the PDF's isolated centrifuge equations when hub rotation is suppressed. Document these corrections rather than silently changing the supplied model.

### Resolve the time convention before high-speed joint training

Use one integration time and one action. The present code mixes mission coordinate time with separate local comoving subsystem clocks; it cannot simply be summed unchanged. Explicitly determine which internal derivatives are with respect to proper time and transform their contributions to the common action, including the associated time and velocity factors. Field/frame conventions must also be consistent.

Until that audit passes, a same-time sum of the printed terms may be tested as an **effective mathematical benchmark**, but it must not be claimed to predict physical 1g internal motion at a center-of-mass speed of 0.995c. A fully relativistic flexible spacecraft is not already established by the PDF. Do not train the final high-speed coupled model on an unresolved clock convention.

### Joint architecture and execution

Train one generic scalar network on the joint coordinates and velocities, and obtain all accelerations from the same Euler-Lagrange velocity Hessian. Keep the known thrust external. Use a single optimizer, joint loss, and checkpoint. Periodic features for genuinely angular coordinates may be used, but keep extension/velocity dependence and required interactions available to the network.

Scale coordinates and time by transforming the whole physical action. Do not independently multiply subsystem energies by arbitrary normalizing factors and then sum them: that would change coupling strengths. Normalize the acceleration *loss* separately by fixed coordinate scales.

A planar five-coordinate joint model is a useful debugging milestone, not a replacement for the full vector experiment. It must be labelled reduced, since it cannot recover general Lorentz deflection or three-axis attitude motion.

## 5. Generate data that covers the intended tests

### Shared speed sampling

For the main PHM runs, use the source-inspired distribution

\[
\beta=0.99999\tanh(2z),\qquad z\sim\mathcal N(0,1).
\]

Generate fresh samples each epoch with independent position, velocity, force, and initialization keys. For vector translation, sample a speed magnitude from the absolute value of this distribution and a direction independently; never sample three near-light-speed components independently, which can create a superluminal speed magnitude.

Use a notebook-style varying-force benchmark with \(g\sim\mathcal N(0,10^2)\), alongside a mission test with fixed \(g=1\). Negative and large values are synthetic excitation tests, not claims about actual engine operation. In PHM models thrust is supplied to the equation of motion as a known external force, not learned as an engine potential. A force input is therefore not required in the neural Lagrangian, unlike the source's conditional uniform-potential family; record this difference.

Sample positions over the declared mission region. Keep distances positive and exclude point-source singularities. Uniform mission-position sampling underrepresents the small near-star regions: for the full model use a stated mixture, 50% uniform along the route and 25% logarithmically distributed near each star, with the 1 AU boundary exclusions retained. Publish the distance limits and sampling weights.

For the full model, independently excite attitudes, angular rates, extensions, and radial rates rather than using only equilibrium data. Use the existing engineering assumptions as the baseline: 100,000 kg total mass, 30 m habitat rest radius, 20 m counterweight radius, and 9.80665 m/s² habitat gravity at equilibrium. Recompute the exact equilibrium and spin for the corrected coupled kinematics; the prior 5.44558 rpm is a check for the isolated benchmark, not a value to force onto a changed model.

Start with the existing extension and spin neighborhoods as a provisional domain, then verify tension, positive radii, stable elastic limits, and validity of the slow internal-motion approximation. Publish actual physical bounds before generating labels. Sample attitudes through a safe nonsingular chart. These bounds are part of the experiment, not values to change after looking at the final test.

Use a stratified sampler for the full adaptation so weaker internal signals are not drowned out by propulsion. Keep the source-inspired distribution unchanged in the source-faithful control. Keep all reported physical parameters fixed initially; varying them as conditional inputs is a separate generalization experiment.

### Separate three evaluation questions

1. **Interpolation:** new states throughout the same speed/position/internal domain as training. This is the closest analogue of the source experiment.
2. **Extrapolation:** a separate pair of models trained only below 0.7c, then tested at higher speeds. Do not call high-speed tests extrapolation if the training sampler already includes those speeds.
3. **Mission rollout:** independent initial conditions and the fixed engineering scenario, including the original rest-to-half-trip translation test.

Reserve independent immutable validation and final-test seeds. Suggested fixed seeds: training/initialization 0-4, validation 1000, final test 2000, with independent split keys inside each generator. Use at least 10,000 validation states and 10,000 test states, stratified across speed bands: 0-0.7c, 0.7-0.9c, 0.9-0.99c, and 0.99-0.99999c. Hold out whole trajectories, not adjacent points from one training trajectory.

## 6. Train both primary models under matched conditions

First run a 1,000-update smoke test; this is not the scientific result. Check finite losses/gradients, units, Hessian conditioning, runtime, and memory. Profile the full joint model before committing to the long runs.

For each primary model, train five independent initializations, each with the complete 300,000-update budget and batch size 512. Use the shared architecture, initialization, scheduler, and speed sampling. Generate 3,000 batches per epoch by streaming or deterministic chunking; it is unnecessary to retain the whole epoch in memory. Archive every intentional deviation and do not silently shrink the full model's budget.

For one-dimensional acceleration, use

\[
\mathcal J=\frac{\sum_i w_i|a_{\theta,i}-a_{*,i}|}{\sum_i w_i},
\qquad w_i=1+\gamma_i.
\]

For the joint model, replace the scalar absolute error by the mean of the nine absolute acceleration errors divided by fixed, predeclared coordinate acceleration scales. Set these scales from nondimensionalization and reference-domain statistics before training, with positive floors for nearly zero components. Use the same translational gamma weight. Report each component separately; one aggregate loss can hide an incorrect attitude or tether law.

Use float64 for the PHM adaptation and log the dtype. Different precisions in a historical control must be reported. Keep the source pseudoinverse convention for the generic reproduction track, but monitor minimum singular values, sign/eigenvalue behavior, and condition numbers. A finite pseudoinverse result is not proof of a well-posed learned law. Do not silently add Hessian regularizers, gradient clipping, or speed clipping; test these as separately named interventions if needed.

Log training and fixed-validation errors every 1,000 updates. Save resumable checkpoints every 10,000 updates and at the final update, including optimizer/RNG states and configuration. Retain both the final checkpoint and the checkpoint selected by a predeclared validation score. Never choose the checkpoint using the final test set. Keep all seeds in the report, including failed seeds.

## 7. Verify correctness, not just a small training loss

### Reference and equation tests before training

- Incomplete model: closed-form constant-force solution versus explicit equations versus automatic Euler-Lagrange equations.
- Full model: automatic equations versus independently derived component-limit equations and directional finite-difference checks of the joint action.
- Full coupling: moving the center of mass must change gravity-gradient torque; changing attitude must change the corresponding position force; extension changes must change spin inertia; hub/rotor motion must produce the derived angular-momentum exchange.
- Validate the Legendre energy and work identity; verify conserved total angular momentum only in an isolated test with external fields/torques disabled. Centrifuge spin momentum alone is not generally conserved in a coupled hub model.
- Refine solver tolerances/time resolution and confirm reference errors are comfortably smaller than the model acceptance thresholds.

### Rollouts and physical measurements

Use common initial conditions and output times for reference, generic LNN, and a direct acceleration-network baseline. Run at least 40 independent rollouts plus the mission scenario, at short horizons and longer horizons. Predeclare the physical horizons.

The full problem is multiscale: tether motion occurs on fractions of a second while travel spans years. A single time convention does not require a single crude timestep. Use a verified stiff/multirate reference and matched learned-model solver. Do not promise a fully resolved multiyear flexible-body rollout without profiling. Test full coupling on resolvable windows at several mission locations and velocities; any averaged long-mission model must be separately validated and labelled an approximation.

Measure acceleration error in every speed band; position/velocity rollout errors; attitude error; habitat/counterweight extension error; spin and artificial-gravity errors; total momentum/energy identities where applicable; and numerical failures.

For autonomous unforced cases, evaluate true physical energy along the learned trajectory. For powered cases, evaluate

\[
E_{\mathrm{true}}(t)-E_{\mathrm{true}}(0)
-\int_0^t\mathbf Q_{\mathrm{ext}}\cdot\dot{\mathbf q}\,dt.
\]

Also report the learned-energy residual separately. Conserving the network's own energy is not evidence that it learned the physical energy. Neither is a close plotted value of the learned Lagrangian: additive constants, total derivatives, and sometimes scaling ambiguities make that comparison non-unique.

Record the first superluminal prediction, integration failure, invalid radius, or nonfinite state. Stop physical evaluation there and count it as failure. Do not hide failure by clipping velocities or silently discarding invalid trajectory segments.

### Proposed acceptance targets, fixed before final testing

These are project targets, not tolerances claimed by Cranmer, and not guarantees of success:

| Measurement | Initial acceptance target |
| --- | --- |
| Acceleration error | Mean absolute error / fixed acceleration scale below 1% in every coordinate and speed band; also report unscaled error |
| Speed prediction | MAE below 0.001c and maximum error below 0.005c on the predeclared rollouts |
| Speed limit | No predicted speed at or above c on any evaluated rollout |
| True energy or work balance | Maximum normalized residual below 0.1% on the tested horizon |
| Habitat artificial gravity | Error below 0.01g on the declared coupled internal-motion tests |
| Attitude and extension | MAE below 1 degree and 1 cm, respectively, on the declared internal-motion tests |

Use fixed reference energy/work scales rather than dividing by a nearly zero initial energy. Report individual subsystem energy transfers as well as total energy; a large translational energy scale can conceal incorrect internal motion. A result passes robustly only if all five seeds meet the declared targets; otherwise report seed success rate, median, range, and failures. Stronger claims require additional tests, not just these thresholds.

## 8. Check whether each added term was actually learned

Run a matched axial ablation ladder on common states: relativistic term only; plus gravity/radiation; plus electric coupling. Train the full coupled model separately. An axial-versus-nine-coordinate comparison changes dimensionality as well as physics and cannot alone identify the source of failure.

In the full model, compare paired states or reference runs with one interaction enabled/disabled. Measure predicted *changes* in acceleration, torque, extension, and momentum exchange, not just total trajectory overlap. Some nominal EM effects are extremely small compared with propulsion: agreement on the main trajectory does not establish that the network learned them.

If an interaction's signal is below solver/model resolution, report it as unresolved. An additional explicitly amplified, dimensionless stress test may test identifiability, but it must not be described as the nominal spacecraft's performance.

Keep a direct acceleration network with the same width, optimizer budget, and data as a baseline. If a structured LNN with exact relativistic kinetic geometry is later tried, report it as a separate model, not a successful reproduction of the generic LNN. Do not add energy supervision or trajectory losses to the main reproduction run; those are distinct interventions.

## 9. Deliverables and execution order

1. Pin and validate the source control.
2. Audit incomplete physics and label generation.
3. Derive and validate the common-time, genuinely coupled full action, including the kinematic corrections.
4. Validate a reduced joint prototype, then the nine-coordinate target.
5. Smoke-test and profile both generic PHM networks.
6. Run matched full-budget training and independent evaluation.
7. Run interaction ablations and, separately, the low-speed-only extrapolation study.
8. Update the research report with definitions, corrections, complete settings, seed distributions, plots, failures, and limitations.

Keep the code small: reuse the shared Euler-Lagrange engine; isolate analytic physics/data generation; use one configurable training/evaluation driver. Store each run's config, manifest, checkpoint, metrics, and figures in a distinct directory without overwriting earlier results. Do not duplicate entire notebooks or keep temporary experiments in the main code path.

Required plots: training/validation loss; acceleration error by speed band and coordinate; phase-space and velocity rollouts; true and learned energy/work residuals; full-model attitude/extension/spin/1g traces; paired interaction effects; and five-seed summaries.

The current training commands do **not** implement this plan. New runnable commands should be delivered only after the common trainer and coupled model are implemented and checked. No training or code/report modification is included in this planning turn.

## Sources and provenance

- Supplied `PHM_Derivations.pdf`, Sections 9-14: `/Users/junxiwu/Downloads/ISEF PROJECT FINAL/PHM_Derivations.pdf` (read-only).
- Current implementation: `phm_lnn.py`, `phm_rotating_lnn.py`, `README.md`, and `results_extended/assumptions.json` (inspected, unchanged).
- [Cranmer's SpecialRelativity notebook](https://github.com/MilesCranmer/lagrangian_nns/blob/master/notebooks/SpecialRelativity.ipynb): executed architecture setup, sampler, acceleration loss, learning-rate function, and training loop inspected on 2026-10-12.
- [Original network builder](https://github.com/MilesCranmer/lagrangian_nns/blob/master/examples/hyperopt/HyperparameterSearch.py): interpretation of `layers=4` as four hidden layers.
- [Lagrangian Neural Networks paper](https://arxiv.org/html/2003.04630): methodology, particle example, and distinction between canonical and observed coordinates.

The experiment designs, coupling audit, validation splits, acceptance targets, and repeated-seed protocol above are proposed for this project. They are not claims that the source notebook already performed those tests.
