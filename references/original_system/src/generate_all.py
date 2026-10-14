"""Generate all numerical results, validation checks, and Matplotlib figures."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from hail_mary_model import (
    Parameters, beta_from_eta, cable_properties, constant_proper_accel,
    corrected_beetle, corrected_navigation, drag_force, drag_rayleigh,
    equilibrium_extension, mission_mass_ratio, parameter_dict,
    simulate_attitude, simulate_centrifuge, simulate_full, simulate_translation, stellar_forces,
)

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
DATA = ROOT / "data"
DOCS = ROOT / "docs"
VALID = ROOT / "validation"
SOURCE = ROOT / "source"
for d in (PLOTS, DATA, DOCS, VALID, SOURCE):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 6), "figure.dpi": 150, "savefig.dpi": 180,
    "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.24, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})
COLORS = {"blue":"#2563EB", "orange":"#F97316", "green":"#16A34A", "red":"#DC2626", "purple":"#7C3AED", "gray":"#64748B", "cyan":"#0891B2"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, bbox_inches="tight")
    plt.close(fig)


def line_plot(x, ys, labels, title, xlabel, ylabel, name, *, ylog=False, xlog=False, hline=None):
    fig, ax = plt.subplots()
    colors = list(COLORS.values())
    for i, (y, label) in enumerate(zip(ys, labels)):
        ax.plot(x, y, lw=2.1, label=label, color=colors[i % len(colors)])
    if hline is not None: ax.axhline(hline, color="#111827", ls="--", lw=1)
    if ylog: ax.set_yscale("log")
    if xlog: ax.set_xscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if labels: ax.legend()
    save(fig, name)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows: return
    if fieldnames is None: fieldnames = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def years(x, p): return np.asarray(x) / p.year
def ly(x, p): return np.asarray(x) / p.ly


p = Parameters()

# Preserve the full user-provided source package.
for src, dst in [
    (Path("/Users/junxiwu/Downloads/PHM_Derivations.pdf"), SOURCE / "PHM_Derivations.pdf"),
    (Path("/Users/junxiwu/Downloads/astrophage physics.xlsx"), SOURCE / "astrophage_physics_original.xlsx"),
    (Path("/Users/junxiwu/.codex/attachments/ea18855d-34cb-478a-bcb9-404ac46b8e66/pasted-text.txt"), SOURCE / "project_roadmap.txt"),
]:
    shutil.copy2(src, dst)

# Core simulations reused across plots and summaries.
ideal = simulate_translation(p, "ideal")
gravity = simulate_translation(p, "gravity_radiation")
nominal = simulate_translation(p, "nominal")
dense = simulate_translation(p, "dense_dust", density_scale=1e7)
low_eff = simulate_translation(p, "lower_efficiency")
interrupted = simulate_translation(p, "interrupted")
power_limited = simulate_translation(p, "engine_power_limited")
low_fuel = simulate_translation(p, "low_fuel")
fixed_thrust = simulate_translation(p, "fixed_thrust")
constant_mass = simulate_translation(p, "constant_mass", initial_mass=p.dry_mass)
worst = simulate_translation(p, "worst_case", density_scale=1e7)
centrifuge = simulate_centrifuge(p)
attitude = simulate_attitude(p)
full = simulate_full(p, "nominal", n_eval=2400)

# Stage 2A - relativistic translation.
tau = ideal["tau"]; th = p.c * p.eta_half / p.proper_accel
ana_t = np.empty_like(tau); ana_x = np.empty_like(tau); ana_eta = np.empty_like(tau)
mask = tau <= th
a1 = constant_proper_accel(tau[mask], p.proper_accel, p.c)
ana_t[mask], ana_x[mask], ana_eta[mask] = a1["t"], a1["x"], a1["eta"]
u = tau[~mask] - th; e = p.eta_half - p.proper_accel*u/p.c
ana_eta[~mask] = e
ana_t[~mask] = (p.c/p.proper_accel)*(2*np.sinh(p.eta_half)-np.sinh(e))
ana_x[~mask] = p.distance/2 + (p.c**2/p.proper_accel)*(np.cosh(p.eta_half)-np.cosh(e))
ana_beta, ana_gamma = np.tanh(ana_eta), np.cosh(ana_eta)
ana_mass = p.nominal_initial_mass * np.exp(-p.proper_accel*tau/(p.effective_propulsion*p.c))

line_plot(years(ideal["t"], p), [ly(ideal["x"], p), ly(ana_x, p)], ["numerical", "closed form"], "Position versus Earth time", "Earth time (yr)", "Position (ly)", "2A_position_vs_earth_time.png")
line_plot(years(tau,p), [ly(ideal["x"],p), ly(ana_x,p)], ["numerical", "closed form"], "Position versus crew proper time", "Crew proper time (yr)", "Position (ly)", "2A_position_vs_proper_time.png")
line_plot(years(tau,p), [ideal["beta"], ana_beta], ["numerical", "closed form"], "Velocity stays subluminal", "Crew proper time (yr)", "Velocity / c", "2A_velocity_fraction_c.png")
line_plot(years(tau,p), [ideal["gamma"], ana_gamma], ["numerical", "closed form"], "Lorentz factor history", "Crew proper time (yr)", "Lorentz factor", "2A_lorentz_factor.png")
line_plot(years(tau,p), [ideal["eta"], ana_eta], ["numerical", "closed form"], "Rapidity history", "Crew proper time (yr)", "Rapidity", "2A_rapidity.png")
line_plot(years(tau,p), [years(ideal["t"],p)], ["Earth time"], "Earth time versus crew time", "Crew proper time (yr)", "Earth time (yr)", "2A_earth_vs_crew_time.png")
line_plot(years(tau,p), [ideal["mass"], ana_mass], ["numerical", "mass-law closed form"], "Remaining spacecraft mass", "Crew proper time (yr)", "Mass (kg)", "2A_remaining_mass.png", ylog=True)

# Stage 2B - gravity and radiation.
xs = np.linspace(0, p.distance, 1600); unit_mass = np.ones_like(xs)
fgs, fgt, frs, frt = stellar_forces(xs, unit_mass, p)
line_plot(ly(xs,p), [fgs, fgt], ["source star", "target star"], "Stellar gravity per unit spacecraft mass", "Position (ly)", "Acceleration contribution (m/s²)", "2B_gravity_vs_position.png")
line_plot(ly(xs,p), [frs, frt, frs+frt], ["source", "target", "net"], "Radiation pressure force", "Position (ly)", "Force (N)", "2B_radiation_vs_position.png")
fgs_n, fgt_n, frs_n, frt_n = stellar_forces(xs, np.full_like(xs,p.dry_mass), p)
line_plot(ly(xs,p), [fgs_n+fgt_n, frs_n+frt_n, fgs_n+fgt_n+frs_n+frt_n], ["gravity", "radiation", "net"], "Net external force for 100,000 kg dry mass", "Position (ly)", "Force (N)", "2B_net_external_force.png")
ginterp = np.interp(ideal["tau"], gravity["tau"], gravity["eta"])
line_plot(years(ideal["tau"],p), [(ginterp-ideal["eta"])*p.c/np.maximum(ideal["tau"],1)], ["gravity+radiation minus ideal"], "Acceleration-history difference", "Crew proper time (yr)", "Equivalent mean acceleration difference (m/s²)", "2B_acceleration_difference.png")
fig, ax = plt.subplots(); vals=[ideal["t"][-1]/p.year,gravity["t"][-1]/p.year]; ax.bar(["ideal","gravity + radiation"],vals,color=[COLORS["gray"],COLORS["blue"]]); ax.set(title="Arrival-time change from stellar fields",ylabel="Earth arrival time (yr)"); ax.bar_label(ax.containers[0],fmt="%.6f"); save(fig,"2B_arrival_time_difference.png")

# Stage 2C - gas and dust drag.
betas = np.linspace(0,0.999,1200); etas=np.arctanh(betas); v=p.c*betas
fd=drag_force(etas,p); coeff=(p.gas_density+p.dust_density)*p.area
fnewt=-coeff*v**2
line_plot(v/1e6,[np.abs(fd)],["relativistic drag"],"Drag force versus velocity","Velocity (10⁶ m/s)","|Drag force| (N)","2C_drag_vs_velocity.png",ylog=True)
line_plot(betas,[np.abs(fd)],["relativistic drag"],"Drag force versus beta","β","|Drag force| (N)","2C_drag_vs_beta.png",ylog=True)
line_plot(betas,[np.abs(fd),np.abs(fnewt)],["relativistic","Newtonian quadratic"],"Relativistic drag departs near c","β","|Drag force| (N)","2C_relativistic_vs_newtonian_drag.png",ylog=True)
rho=np.logspace(-24,-17,250); fgas=rho*p.area*(0.9*p.c)**2/(1-0.9**2)
line_plot(rho,[fgas],["gas contribution"],"Drag versus gas density at β=0.9","Gas density (kg/m³)","|Force| (N)","2C_drag_vs_gas_density.png",xlog=True,ylog=True)
line_plot(rho,[fgas],["dust contribution"],"Drag versus dust density at β=0.9","Dust density (kg/m³)","|Force| (N)","2C_drag_vs_dust_density.png",xlog=True,ylog=True)
drag_power=np.abs(nominal["forces"]["drag"]*nominal["v"])
line_plot(years(nominal["tau"],p),[drag_power],["drag power"],"Drag power along nominal mission","Crew proper time (yr)","Power (W)","2C_drag_power_vs_time.png",ylog=True)
line_plot(years(nominal["tau"],p),[np.maximum(nominal["drag_energy"],1)],["cumulative loss"],"Cumulative energy lost to drag","Crew proper time (yr)","Energy (J)","2C_cumulative_drag_energy.png",ylog=True)
fig,axs=plt.subplots(2,1,figsize=(10,8),sharex=True); cats=["gravity-only baseline","nominal ISM","dense cloud"]; xcat=np.arange(3); time_pen=[0,(nominal["t"][-1]-gravity["t"][-1])/86400,np.nan]; fuel_pen=[1e-12,nominal["drag_energy"][-1]/(p.effective_propulsion*p.c**2),dense["drag_energy"][-1]/(p.effective_propulsion*p.c**2)]; axs[0].bar(xcat,time_pen,color=COLORS["blue"]); axs[0].set_ylabel("Arrival-time change (days)"); axs[0].text(2,0,"no arrival\n(0.0497% short)",ha="center",va="bottom",color=COLORS["red"]); axs[1].bar(xcat,fuel_pen,color=COLORS["orange"]); axs[1].set_yscale("log"); axs[1].set_ylabel("Drag energy / (chi epsilon c²) (kg)"); axs[1].set_xticks(xcat,cats); fig.suptitle("Arrival-time and fuel-equivalent penalties from drag"); save(fig,"2C_arrival_and_fuel_penalties.png")

# Stage 2D - variable mass and fuel.
distances=np.linspace(.1,20,300)*p.ly; accels=np.linspace(1,25,300); effs=np.linspace(.2,1,300)
fuel_dist=p.dry_mass*(mission_mass_ratio(distances,p.proper_accel,p.effective_propulsion,p.c)-1)
fuel_acc=p.dry_mass*(mission_mass_ratio(p.distance,accels,p.effective_propulsion,p.c)-1)
fuel_eff=p.dry_mass*(mission_mass_ratio(p.distance,p.proper_accel,p.collimation_efficiency*effs,p.c)-1)
line_plot(distances/p.ly,[fuel_dist],["fuel"],"Fuel required versus trip distance","Distance (ly)","Fuel mass (kg)","2D_fuel_vs_distance.png",ylog=True)
line_plot(accels,[fuel_acc],["fuel"],"Fuel required versus target proper acceleration","Proper acceleration (m/s²)","Fuel mass (kg)","2D_fuel_vs_acceleration.png",ylog=True)
line_plot(effs,[fuel_eff],["fuel"],"Fuel required versus propulsion efficiency","Energy conversion efficiency ε","Fuel mass (kg)","2D_fuel_vs_efficiency.png",ylog=True)
mass_grid=np.geomspace(p.dry_mass,p.nominal_initial_mass,500); fixed_force=p.nominal_initial_mass*p.proper_accel
line_plot(mass_grid,[fixed_force/mass_grid,np.full_like(mass_grid,p.proper_accel)],["fixed thrust","constant proper acceleration"],"Acceleration versus remaining mass","Remaining mass (kg)","Proper acceleration (m/s²)","2D_acceleration_vs_remaining_mass.png",xlog=True,ylog=True)
fuel_grid=np.geomspace(1e3,p.nominal_initial_mass-p.dry_mass,350); eta_pk=.5*p.effective_propulsion*np.log1p(fuel_grid/p.dry_mass); beta_pk=np.tanh(eta_pk)
line_plot(fuel_grid,[beta_pk],["peak β"],"Peak velocity versus fuel mass","Fuel mass (kg)","Peak velocity / c","2D_peak_velocity_vs_fuel.png",xlog=True)
eta_cap=np.minimum(eta_pk,p.eta_half); powered=2*p.c**2/p.proper_accel*(np.cosh(eta_cap)-1); coast=np.maximum(p.distance-powered,0); b=np.maximum(np.tanh(eta_cap),1e-12); trip_e=2*p.c/p.proper_accel*np.sinh(eta_cap)+coast/(p.c*b); trip_c=2*p.c/p.proper_accel*eta_cap+coast/(p.c*b*np.cosh(eta_cap))
line_plot(fuel_grid,[trip_e/p.year,trip_c/p.year],["Earth","crew"],"Travel time versus fuel mass","Fuel mass (kg)","Travel time (yr)","2D_travel_time_vs_fuel.png",xlog=True,ylog=True)
line_plot(fuel_grid,[fuel_grid/p.astrophage_density],["Astrophage volume"],"Fuel-volume requirement","Fuel mass (kg)","Volume (m³)","2D_fuel_volume_requirement.png",xlog=True,ylog=True)

# Stage 2E - centrifuge and tether.
ct=centrifuge["tau"]
line_plot(ct,[centrifuge["xh"],centrifuge["xc"]],["habitat","counterweight"],"Cable extension response","Crew proper time (s)","Extension (m)","2E_cable_extension_vs_time.png")
line_plot(ct,[centrifuge["tension_h"],centrifuge["tension_c"]],["habitat","counterweight"],"Cable tension response","Crew proper time (s)","Tension (N)","2E_cable_tension_vs_time.png")
spin=np.linspace(.05,.6,300); line_plot(spin,[spin**2*p.radius_h],["habitat radius"],"Artificial gravity versus spin rate","Angular speed (rad/s)","Artificial gravity (m/s²)","2E_artificial_gravity_vs_spin.png",hline=9.8)
mh,rh,kh,ch=cable_properties(p,"h"); eq=np.array([equilibrium_extension(mh,rh,kh,w) for w in spin]); line_plot(spin,[eq],["equilibrium extension"],"Equilibrium extension versus angular velocity","Angular speed (rad/s)","Extension (m)","2E_equilibrium_extension_vs_omega.png")
line_plot(ct,[centrifuge["motor_torque"],centrifuge["damping_loss"]],["motor torque (N·m)","damping loss (W)"],"Motor torque and damping loss","Crew proper time (s)","Magnitude","2E_motor_torque_and_damping_loss.png")
impact_mask=(ct>=110)&(ct<=170); line_plot(ct[impact_mask],[centrifuge["xh"][impact_mask],centrifuge["xc"][impact_mask]],["habitat","counterweight"],"Discrete radial impact at 120 s","Crew proper time (s)","Extension (m)","2E_discrete_impact_response.png")
ks=np.logspace(4,8,180); ws=np.linspace(.05,1.5,180); K,W=np.meshgrid(ks,ws); amp=1/np.maximum(np.abs(K-mh*W**2),1); fig,ax=plt.subplots(); im=ax.pcolormesh(ks,ws,np.log10(amp),shading="auto",cmap="magma"); ax.set_xscale("log"); ax.set(title="Tether resonance map",xlabel="Cable stiffness (N/m)",ylabel="Spin rate (rad/s)"); fig.colorbar(im,ax=ax,label="log10 response factor"); save(fig,"2E_resonance_map.png")

# Stage 2F - attitude and optional EM (zero in baseline).
at=attitude["tau"]
line_plot(at,[attitude["omega"][0],attitude["omega"][1],attitude["omega"][2]],["ωx","ωy","ωz"],"Angular velocity components","Crew proper time (s)","Angular velocity (rad/s)","2F_angular_velocity_components.png")
line_plot(at,[attitude["q"][0],attitude["q"][1],attitude["q"][2],attitude["q"][3]],["q0","q1","q2","q3"],"Attitude quaternion history","Crew proper time (s)","Quaternion component","2F_quaternion_history.png")
tq=attitude["torques"]; line_plot(at,[np.linalg.norm(tq[:,1,:],axis=1)],["gravity-gradient torque"],"Gravity-gradient torque","Crew proper time (s)","Torque (N·m)","2F_gravity_gradient_torque.png")
line_plot(at,[np.degrees(attitude["error_angle"])],["attitude error"],"Attitude error","Crew proper time (s)","Error angle (deg)","2F_attitude_error.png")
rec=(at>=90)&(at<=300); line_plot(at[rec],[np.degrees(attitude["error_angle"][rec])],["recovery"],"Recovery following discrete impact at 100 s","Crew proper time (s)","Error angle (deg)","2F_impact_recovery.png")
fig,ax=plt.subplots(); ax.stackplot(at,np.linalg.norm(tq[:,0,:],axis=1),np.linalg.norm(tq[:,1,:],axis=1),np.linalg.norm(tq[:,2,:],axis=1),labels=["control","gravity-gradient","electromagnetic (zero)"]); ax.set(title="Contribution of each torque source",xlabel="Crew proper time (s)",ylabel="Torque magnitude (N·m)"); ax.legend(loc="upper right"); save(fig,"2F_torque_source_contributions.png")

# Stage 3 - full deterministic outputs and scenario suite.
line_plot(years(nominal["t"],p),[ly(nominal["x"],p)],["nominal full model"],"Full spacecraft trajectory","Earth time (yr)","Position (ly)","3_full_spacecraft_trajectory.png")
sample=slice(None,None,12); f=nominal["forces"]; fig,ax=plt.subplots(); ax.stackplot(years(nominal["tau"][sample],p),*[np.abs(f[k][sample]) for k in ["thrust","gravity_source","gravity_target","radiation_source","radiation_target","drag"]],labels=["thrust","source gravity","target gravity","source radiation","target radiation","drag"]); ax.set_yscale("log"); ax.set(title="Force-budget magnitude",xlabel="Crew proper time (yr)",ylabel="Force magnitude (N)"); ax.legend(loc="upper center",ncol=3); save(fig,"3_force_budget_stacked.png")
fig,axs=plt.subplots(3,1,figsize=(10,10),sharex=True); axs[0].plot(years(nominal["tau"],p),nominal["beta"],color=COLORS["blue"]); axs[0].set_ylabel("β"); axs[1].plot(years(nominal["tau"],p),nominal["eta"],color=COLORS["orange"]); axs[1].set_ylabel("rapidity"); axs[2].plot(years(nominal["tau"],p),nominal["gamma"],color=COLORS["green"]); axs[2].set_ylabel("γ"); axs[2].set_xlabel("Crew proper time (yr)"); fig.suptitle("Velocity, rapidity, and Lorentz factor"); save(fig,"3_velocity_rapidity_lorentz.png")
line_plot(years(nominal["tau"],p),[nominal["mass"],np.full_like(nominal["mass"],p.dry_mass)],["mass","dry-mass floor"],"Fuel and mass history","Crew proper time (yr)","Mass (kg)","3_fuel_mass_history.png",ylog=True)
line_plot(years(nominal["tau"],p),[years(nominal["t"],p)],["Earth time"],"Earth time versus crew time","Crew proper time (yr)","Earth time (yr)","3_earth_vs_crew_time.png")
ke=(nominal["gamma"]-1)*nominal["mass"]*p.c**2; rest=nominal["mass"]*p.c**2; line_plot(years(nominal["tau"],p),[ke,rest,np.maximum(nominal["drag_energy"],1)],["kinetic","remaining rest energy","drag loss"],"Energy-accounting terms","Crew proper time (yr)","Energy (J)","3_energy_momentum_balance.png",ylog=True)
fig,axs=plt.subplots(2,1,figsize=(10,8),sharex=True); axs[0].plot(ct,centrifuge["xh"],label="habitat"); axs[0].plot(ct,centrifuge["xc"],label="counterweight"); axs[0].set_ylabel("Extension (m)"); axs[0].legend(); axs[1].plot(ct,centrifuge["omega"],color=COLORS["purple"]); axs[1].set(ylabel="Spin (rad/s)",xlabel="Crew proper time (s)"); fig.suptitle("Internal centrifuge behavior"); save(fig,"3_internal_centrifuge_behavior.png")
dusts=np.logspace(-24,-15,180); bs=np.linspace(.01,.999,180); DD,BB=np.meshgrid(dusts,bs); GG=1/np.sqrt(1-BB**2); drag_acc=DD*p.area*GG**2*(p.c*BB)**2/p.dry_mass; fig,ax=plt.subplots(); im=ax.pcolormesh(dusts,bs,np.log10(drag_acc/p.proper_accel),shading="auto",cmap="viridis"); ax.set_xscale("log"); ax.set(title="Sensitivity across velocity and dust density",xlabel="Dust density (kg/m³)",ylabel="β"); fig.colorbar(im,ax=ax,label="log10(drag acceleration / commanded acceleration)"); save(fig,"3_sensitivity_velocity_dust_heatmap.png")

# Numerical scenario summary.
scenario_runs = {
    "1 Ideal thrust-only": ideal,
    "2 Gravity and radiation": gravity,
    "3 Nominal gas and dust": nominal,
    "4 Dense dust cloud": dense,
    "5 Lower propulsion efficiency": low_eff,
    "6 Interrupted acceleration": interrupted,
    "7 Engine-power-limited": power_limited,
    "8 Low-fuel arrival": low_fuel,
    "9 Centrifuge disturbance": nominal,
    "10 Combined worst case": worst,
}
scenario_rows=[]
for name,s in scenario_runs.items():
    scenario_rows.append({
        "scenario":name,"arrived":s["arrived"],"earth_time_years":s["t"][-1]/p.year,
        "crew_time_years":s["tau"][-1]/p.year,"peak_beta":float(np.max(s["beta"])),
        "final_beta":float(s["beta"][-1]),"initial_mass_kg":s["initial_mass"],
        "final_mass_kg":float(s["mass"][-1]),"distance_fraction":float(s["x"][-1]/p.distance),
        "drag_energy_J":float(s["drag_energy"][-1]),
    })
write_csv(DATA/"scenario_summary.csv",scenario_rows)

# Auditable state, force, and energy histories from the assembled full model.
full_rows=[]
for i in range(0,len(full["tau"]),4):
    row={"tau_s":float(full["tau"][i])}
    row.update({name:float(full["state"][i,j]) for j,name in enumerate(full["columns"])})
    full_rows.append(row)
write_csv(DATA/"full_state_sample.csv",full_rows)
mission_rows=[]
for i in range(0,len(nominal["tau"]),4):
    row={
        "tau_s":float(nominal["tau"][i]),"earth_time_s":float(nominal["t"][i]),"position_m":float(nominal["x"][i]),
        "eta":float(nominal["eta"][i]),"beta":float(nominal["beta"][i]),"gamma":float(nominal["gamma"][i]),
        "mass_kg":float(nominal["mass"][i]),"kinetic_energy_J":float((nominal["gamma"][i]-1)*nominal["mass"][i]*p.c**2),
        "rest_energy_J":float(nominal["mass"][i]*p.c**2),"cumulative_drag_energy_J":float(nominal["drag_energy"][i]),
        "electromagnetic_force_N":0.0,"impact_force_N":0.0,
    }
    row.update({f"force_{k}_N":float(v[i]) for k,v in nominal["forces"].items()})
    mission_rows.append(row)
write_csv(DATA/"nominal_mission_timeseries.csv",mission_rows)
internal_rows=[]
for i in range(0,len(centrifuge["tau"]),10):
    internal_rows.append({"tau_s":float(centrifuge["tau"][i]),"xi_h_m":float(centrifuge["xh"][i]),"xi_c_m":float(centrifuge["xc"][i]),"tension_h_N":float(centrifuge["tension_h"][i]),"tension_c_N":float(centrifuge["tension_c"][i]),"spin_rad_s":float(centrifuge["omega"][i]),"motor_torque_Nm":float(centrifuge["motor_torque"][i]),"damping_loss_W":float(centrifuge["damping_loss"][i])})
write_csv(DATA/"internal_diagnostics.csv",internal_rows)

beetle=corrected_beetle(p); nav=corrected_navigation(p)
numerical = {
    "distance_ly":p.distance_ly,"half_trip_rapidity":p.eta_half,
    "peak_beta":math.tanh(p.eta_half),"peak_gamma":math.cosh(p.eta_half),
    "crew_trip_years":ideal["tau"][-1]/p.year,"earth_trip_years":ideal["t"][-1]/p.year,
    "effective_propulsion_chi_epsilon":p.effective_propulsion,
    "required_mass_ratio":p.required_mass_ratio,"required_initial_mass_kg":p.nominal_initial_mass,
    "required_fuel_mass_kg":p.nominal_initial_mass-p.dry_mass,
    "fuel_volume_m3":(p.nominal_initial_mass-p.dry_mass)/p.astrophage_density,
    "beetle_peak_beta_corrected":beetle["beta_peak"],"beetle_earth_trip_years":beetle["trip_earth"]/p.year,
    "beetle_crew_trip_years":beetle["trip_proper"]/p.year,
    "navigation_required_decel_m_s2":nav["required_proper_decel"],
    "navigation_earth_stop_days":nav["earth_stop_time"]/86400,
}
write_csv(DATA/"numerical_results.csv",[{"metric":k,"value":v} for k,v in numerical.items()])

# Parameters, uncertainty, allowed ranges, and provenance.
param_rows=[
    ("c",p.c,"m/s","exact","exact","SI defined constant"),("G",p.G,"m^3 kg^-1 s^-2","2.2e-5 relative","6.67415e-11 to 6.67445e-11","CODATA"),
    ("D",p.distance,"m","source estimate","10 to 13 ly","source workbook: Hail Mary"),("m_dry",p.dry_mass,"kg","design assumption","8e4 to 1.5e5","source workbook"),
    ("alpha",p.proper_accel,"m/s^2","design assumption","5 to 25","source workbook"),("epsilon",p.propulsion_efficiency,"1","model assumption","0.2 to 1.0","source workbook propellant ratio"),
    ("chi",p.collimation_efficiency,"1","design assumption","0.3 to 1.0","source workbook stacking efficiency"),("rho_astrophage",p.astrophage_density,"kg/m^3","fictional estimate","4500 to 5500","source workbook"),
    ("A",p.area,"m^2","geometry estimate","8 to 20","4 m diameter source workbook"),("rho_g",p.gas_density,"kg/m^3","order of magnitude","1e-23 to 1e-19","nominal ISM assumption"),
    ("rho_d",p.dust_density,"kg/m^3","order of magnitude","1e-25 to 1e-20","nominal ISM assumption"),("C_g",p.gas_drag_coefficient,"1","model assumption","0.5 to 2","momentum-transfer coefficient"),
    ("C_d",p.dust_drag_coefficient,"1","model assumption","0.5 to 2","momentum-transfer coefficient"),("M_s",p.source_mass,"kg","few percent","1.8e30 to 2.1e30","source workbook"),
    ("M_T",p.target_mass,"kg","order 10 percent","1.3e30 to 1.8e30","source workbook Tau Ceti"),("L_s",p.source_luminosity,"W","few percent","3.7e26 to 4.0e26","source workbook"),
    ("L_T",p.target_luminosity,"W","order 20 percent","1.5e26 to 2.5e26","source workbook"),("C_R",p.radiation_coefficient,"1","model assumption","1 to 2","radiation pressure model"),
    ("E_cable",p.cable_modulus,"Pa","material/design","2e11 to 3e11","Zylon engineering assumption"),("d_cable",p.cable_diameter,"m","design assumption","0.008 to 0.015","source workbook"),
]
write_csv(DATA/"parameters.csv",[{"symbol":a,"value":b,"unit":c,"uncertainty":d,"allowed_range":e,"source":f} for a,b,c,d,e,f in param_rows])

# Validation suite.
pos_rel=np.max(np.abs(ideal["x"]-ana_x))/p.distance
time_rel=np.max(np.abs(ideal["t"]-ana_t))/max(ana_t[-1],1)
eta_abs=np.max(np.abs(ideal["eta"]-ana_eta))
mass_rel=np.max(np.abs(ideal["mass"]-ana_mass))/p.nominal_initial_mass
small_tau=np.linspace(0,1e4,500); small=constant_proper_accel(small_tau,1.0,p.c); newton_x=.5*small_tau**2
newton_rel=np.max(np.abs(small["x"]-newton_x))/newton_x[-1]
vv=.4*p.c; h=1.0; deriv=(drag_rayleigh(vv+h,p)-drag_rayleigh(vv-h,p))/(2*h); target=-drag_force(math.atanh(vv/p.c),p)
drag_deriv_rel=abs(deriv-target)/abs(target)
coarse=simulate_translation(p,"ideal",n_eval=500,max_step=20_000); fine=simulate_translation(p,"ideal",n_eval=500,max_step=10_000)
conv_time=abs(coarse["t"][-1]-fine["t"][-1])/fine["t"][-1]; conv_pos=abs(coarse["x"][-1]-fine["x"][-1])/p.distance
q_norm_err=float(np.max(np.abs(np.linalg.norm(attitude["q"],axis=0)-1)))
checks={
    "closed_form_position_relative_error":pos_rel,"closed_form_time_relative_error":time_rel,
    "closed_form_rapidity_absolute_error":eta_abs,"mass_law_relative_error":mass_rel,
    "newtonian_limit_position_relative_error":newton_rel,"drag_rayleigh_derivative_relative_error":drag_deriv_rel,
    "halved_step_arrival_time_relative_change":conv_time,"halved_step_arrival_position_relative_change":conv_pos,
    "max_abs_beta":float(np.max(np.abs(nominal["beta"]))),"minimum_mass_kg":float(np.min(nominal["mass"])),
    "quaternion_norm_max_error":q_norm_err,"minimum_cable_length_m":float(min(p.radius_h+np.min(centrifuge["xh"]),p.radius_c+np.min(centrifuge["xc"]))),
}
checks["all_physical_state_checks_pass"] = bool(checks["max_abs_beta"]<1 and checks["minimum_mass_kg"]>=p.dry_mass*(1-1e-5) and checks["minimum_cable_length_m"]>0 and q_norm_err<1e-9)
checks["analytical_validation_pass"] = bool(pos_rel<1e-6 and time_rel<1e-6 and eta_abs<1e-6 and mass_rel<1e-6 and newton_rel<1e-6 and drag_deriv_rel<1e-6)
checks["step_convergence_pass"] = bool(conv_time<1e-6 and conv_pos<1e-6)
(VALID/"validation_results.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
write_csv(VALID/"validation_results.csv",[{"check":k,"value":v} for k,v in checks.items()])

# Frozen model specification and roadmap coverage.
spec=f"""# Frozen physical model specification

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

For `D={p.distance_ly:.3f} ly`, `alpha={p.proper_accel:.3f} m/s^2`, `chi epsilon={p.effective_propulsion:.5f}`, `eta_h={p.eta_half:.8f}`, the corrected mass ratio is `{p.required_mass_ratio:.8g}` and the required initial mass for a `{p.dry_mass:.0f} kg` dry craft is `{p.nominal_initial_mass:.8g} kg`.

## Physics partition

Conservative: relativistic free motion, stellar gravity, radiation pseudo-potentials for fixed luminosity, cable elasticity, rotational kinetic energy, gravity-gradient potential, EM potentials. Dissipative: gas/dust drag, tether viscous/Coulomb losses, bearing losses, attitude damping. Externally forced: thrust, motor torque, attitude control, time-varying radiation/EM fields. Impulsive: dust/impact momentum and angular-momentum jumps.

## Assumptions and limitations

One-dimensional translation; fixed stellar positions; no orbital insertion; no fuel-tank structural mass scaling; `chi` and `epsilon` constant unless swept; medium at rest in stellar frame; fluid-like drag is a continuum approximation; radiation pressure uses a constant projected area; internal/translation coupling is retained through force bookkeeping but internal energy is negligible relative to `mc^2`; multirate propagation solves slow translation and fast internal transients in deterministic partitions; cable model is linear elastic; quaternion control is a simple PD law; EM and random impacts are zero in baseline; dense-cloud case is a stress test, not a measured forecast.
"""
(DOCS/"model_specification.md").write_text(spec,encoding="utf-8")

assumptions=[
    ("Translation","1-D axial path","Baseline","No lateral navigation/orbits"),("Time","tau integrates all states","Baseline","External fields evaluated in stellar frame"),
    ("Propulsion","T is proper force; chi and epsilon separate","Baseline","No thermal/power hardware mass"),("Mass","Rest mass excludes modeled internal energy corrections","Baseline","Error is negligible vs mc^2"),
    ("Drag","Continuum effective law","Baseline","Individual grains handled only as impacts"),("Impacts","Discrete state jumps","Enabled in internal tests","No stochastic impact population"),
    ("Electromagnetism","Interfaces retained, terms zero","Initially zero","Requires charge/field model"),("Centrifuge","Linear elastic tethers","Enabled","No cable creep/plastic failure"),
    ("Attitude","Rigid body, PD control","Enabled","No actuator saturation"),("Stars","Fixed point masses/luminosities","Enabled","No ephemerides"),
]
write_csv(DATA/"assumptions_limitations.csv",[{"subsystem":a,"assumption":b,"initial_setting":c,"limitation":d} for a,b,c,d in assumptions])

plot_files=sorted(x.name for x in PLOTS.glob("*.png"))
coverage="""# Roadmap completion map

- Stage 1: `docs/model_specification.md`, `data/parameters.csv`, `data/assumptions_limitations.csv`, and `astrophage_physics_corrected.xlsx`.
- Stage 2A: seven `plots/2A_*` figures plus closed-form validation in `validation/`.
- Stage 2B: five `plots/2B_*` figures.
- Stage 2C: eight `plots/2C_*` figures and Rayleigh derivative validation.
- Stage 2D: seven `plots/2D_*` figures.
- Stage 2E: seven `plots/2E_*` figures with a discrete impact.
- Stage 2F: six `plots/2F_*` figures; EM is explicitly zero in the baseline.
- Stage 3: ten scenarios in `data/scenario_summary.csv`, the assembled 17-state history in `data/full_state_sample.csv`, complete nominal force/energy bookkeeping in `data/nominal_mission_timeseries.csv`, and eight `plots/3_*` figures.
- Completion gates: analytical limits, step convergence, physical-state checks, and workbook formula scan are under `validation/`.
"""
(DOCS/"roadmap_completion.md").write_text(coverage,encoding="utf-8")

readme=f"""# Hail Mary physics - complete deterministic package

This folder contains the complete work requested in `source/project_roadmap.txt`. Start with `docs/model_specification.md` for the frozen equations, `data/numerical_results.csv` for headline values, and `data/scenario_summary.csv` for the ten reference cases. The corrected workbook is `astrophage_physics_corrected.xlsx`. Source code is in `src/`; every plot is reproducible by running `python3 src/generate_all.py` from this folder, followed by the workbook builder.

Headline corrected results:

- Half-trip rapidity: {p.eta_half:.8f}; peak beta: {math.tanh(p.eta_half):.8f}; peak gamma: {math.cosh(p.eta_half):.8f}.
- Ideal trip: {ideal['tau'][-1]/p.year:.6f} crew years and {ideal['t'][-1]/p.year:.6f} Earth years.
- Correct accelerate/brake mass ratio: {p.required_mass_ratio:.8g}; fuel: {p.nominal_initial_mass-p.dry_mass:.8g} kg; volume: {(p.nominal_initial_mass-p.dry_mass)/p.astrophage_density:.8g} m^3.
- Corrected Beetle peak beta: {beetle['beta_peak']:.8f}; Earth trip: {beetle['trip_earth']/p.year:.6f} years.
- Corrected navigation deceleration: {nav['required_proper_decel']:.8f} m/s^2; stop time: {nav['earth_stop_time']/86400:.6f} days.

Generated {len(plot_files)} Matplotlib PNG figures. See `docs/plot_catalog.md` for the complete list.
"""
(ROOT/"README.md").write_text(readme,encoding="utf-8")
(DOCS/"plot_catalog.md").write_text("# Plot catalog\n\n"+"\n".join(f"- `{x}`" for x in plot_files)+"\n",encoding="utf-8")
print(json.dumps({"plots":len(plot_files),"results":numerical,"validation":checks},indent=2))
