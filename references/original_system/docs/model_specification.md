# Frozen physical model specification

## Scope and independent variable

The deterministic model uses crew proper time `tau` as its integration variable. The external source/target fields are evaluated in the stars' inertial frame. Translation uses `(t, x, eta, m)` with `dt/dtau = cosh(eta)`, `dx/dtau = c sinh(eta)`, `v = c tanh(eta)`, and `gamma = cosh(eta)`. Centrifuge and attitude states are evaluated in the instantaneous spacecraft rest frame.

The complete state is `y = [t, x, eta, m, xi_h, xi_h', xi_c, xi_c', theta, theta', q0, q1, q2, q3, omega_x, omega_y, omega_z]`; primes mean `d/dtau`. The quaternion is scalar-first and normalized after every attitude integration segment.

## Corrected action model

Because the source PDF added coordinate-time translation terms directly to proper-time internal terms, it was not one variational system. A consistent first-order, proper-time-parameterized effective action is

`L_tau = m c sinh(eta) x' - m c^2 cosh(eta) t' - U_ext(x) t' + Q A_x x' - Q phi t' + L_internal(q, q')`,

with `U_ext = -G M_s m/r_s - G M_T m/r_T + K_s/r_s + K_T/r_T`, `r_s=r_s0+x`, `r_T=D+r_T0-x`, and `K_*=C_R A L_*/(4 pi c)`. Proper-time gauge gives `t'=cosh(eta)` and `x'=c sinh(eta)`. The time-varying mass and thrust are an open propulsion subsystem, not a conservative potential.

`L_internal = sum_j [0.5 M_j xi_j'^2 + 0.5 M_j(r_j+xi_j)^2 theta'^2 - 0.5 k_j xi_j^2] + 0.5 omega^T I omega - U_gg`, with `U_gg = -mu/(2r^3)[tr(I)-3 rhat^T I rhat]`.

## Corrected dissipation function

`R_tau = t' C rho A c^3 [artanh(|beta|)-|beta|] + sum_j[0.5 c_j xi_j'^2 + F_C,j |xi_j'|] + 0.5 b_theta theta'^2 + tau_C |theta'| + 0.5 omega^T B omega`.

Here `C rho = C_g rho_g + C_d rho_d`. Differentiating the translational term with respect to `x'` gives `-C rho A gamma^2 v|v|`. Impacts are excluded from `R_tau` and applied as exact momentum jumps.

## Explicit equations of motion

`t'=cosh eta`; `x'=c sinh eta`; `m c eta' = T + F_g + F_rad + F_drag + F_EM`; `m'=-|T|/(chi epsilon c)` while `m>m_dry` and zero at the dry-mass limit.

`F_g=-G M_s m/r_s^2 + G M_T m/r_T^2`; `F_rad=K_s/r_s^2-K_T/r_T^2`; `F_drag=-(C_g rho_g+C_d rho_d) A gamma^2 v|v|`.

`M_j xi_j'' + c_j xi_j' + k_j xi_j + F_C,j sgn(xi_j') = M_j(r_j+xi_j) theta'^2`; `d(I_cent theta')/dtau = tau_motor-b_theta theta'-tau_C sgn(theta')`.

`I omega' + omega x (I omega) = tau_engine + tau_rad + tau_gg + tau_EM - B omega`, `q'=0.5 Omega(omega)q`, and `tau_gg=3 mu/r^3 rhat x (I rhat)`.

At an impact, `Delta p=J`, `Delta xi_j'=J_j/M_j`, and `Delta omega=I^-1(r x J)`. Electromagnetic force and torque are zero in the baseline but retain the interfaces `F_EM=Q(E+v x B)+grad(mu_b.B)` and `tau_EM=mu_b x B`.

## Propulsion and corrected mass ratio

Thrust `T` is signed proper force in newtons. `epsilon` is rest-energy-to-beam efficiency; `chi` is beam momentum/collimation efficiency. Therefore `T=chi epsilon c (-m')`. For a thrust-only constant proper acceleration leg, `Delta eta=chi epsilon ln(m_i/m_f)`. The complete accelerate/brake mission requires `m_0/m_f=exp(2 eta_h/(chi epsilon))`, not the one-leg factor used in the original Hail Mary sheet.

For `D=11.900 ly`, `alpha=15.000 m/s^2`, `chi epsilon=0.40155`, `eta_h=3.03213934`, the corrected mass ratio is `3620692.2` and the required initial mass for a `100000 kg` dry craft is `3.6206922e+11 kg`.

## Physics partition

Conservative: relativistic free motion, stellar gravity, radiation pseudo-potentials for fixed luminosity, cable elasticity, rotational kinetic energy, gravity-gradient potential, EM potentials. Dissipative: gas/dust drag, tether viscous/Coulomb losses, bearing losses, attitude damping. Externally forced: thrust, motor torque, attitude control, time-varying radiation/EM fields. Impulsive: dust/impact momentum and angular-momentum jumps.

## Assumptions and limitations

One-dimensional translation; fixed stellar positions; no orbital insertion; no fuel-tank structural mass scaling; `chi` and `epsilon` constant unless swept; medium at rest in stellar frame; fluid-like drag is a continuum approximation; radiation pressure uses a constant projected area; internal/translation coupling is retained through force bookkeeping but internal energy is negligible relative to `mc^2`; multirate propagation solves slow translation and fast internal transients in deterministic partitions; cable model is linear elastic; quaternion control is a simple PD law; EM and random impacts are zero in baseline; dense-cloud case is a stress test, not a measured forecast.
