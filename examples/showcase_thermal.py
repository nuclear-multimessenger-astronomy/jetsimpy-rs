"""
Showcase: Thermal electron emission — MQ21 formalism and FM25 full-volume extension.

Produces three figures for the Thermal Electrons docs page:
  1. thermal_sed.png        — SED with/without thermal, and varying eps_T
  2. thermal_lightcurves.png — Radio light curves and full-volume vs thin-shell
  3. thermal_fm25.png       — Comparison with Ferguson & Margalit (2025) reference code
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blastwave import Jet, TopHat, Spherical, ForwardJetRes, Uniform
from blastwave import FluxDensity_tophat

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
})

outdir = os.path.join(os.path.dirname(__file__), "..", "docs", "validation", "img")
os.makedirs(outdir, exist_ok=True)

DAY = 86400.0
MPC_CM = 3.0856776e24  # 1 Mpc in cm

# ═══════════════════════════════════════════════════════════════
# Common parameters: mildly relativistic jet in dense ISM
# ═══════════════════════════════════════════════════════════════
z = 0.01
d_L = 44.0   # Mpc (approximate for z=0.01)

P_base = dict(
    Eiso=1e52, lf=10.0, theta_c=0.3,
    n0=1.0, A=0.0,
    eps_e=0.01, eps_b=0.1, p=3.0,
    theta_v=0.0, d=d_L, z=z,
)

# ═══════════════════════════════════════════════════════════════
# Figure 1: Thermal SED (1×2)
# ═══════════════════════════════════════════════════════════════
print("=== Thermal SED ===")

nu_arr = np.geomspace(1e7, 1e18, 300)
t_sed = 100 * DAY  # epoch when blast wave is trans-relativistic

jet_sed = Jet(
    TopHat(0.3, 1e52, lf0=10.0), 0.0, 1.0,
    tmin=10.0, tmax=1e9, grid=ForwardJetRes(0.3, 129),
    spread=True, eps_e=0.01, eps_b=0.1, p_fwd=3.0,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: thermal vs non-thermal at fixed eps_T
print("  Left panel: thermal vs non-thermal...")
ax = axes[0]
P_sed = dict(P_base, eps_T=0.5)

flux_nt = np.array([jet_sed.FluxDensity(t_sed, float(nu), P_base, model="sync_ssa_smooth") for nu in nu_arr])
flux_th = np.array([jet_sed.FluxDensity(t_sed, float(nu), P_sed, model="sync_thermal") for nu in nu_arr])

m = flux_nt > 0
ax.loglog(nu_arr[m], flux_nt[m], "-", color="#1f77b4", lw=2, label="Non-thermal only")
m = flux_th > 0
ax.loglog(nu_arr[m], flux_th[m], "-", color="#d62728", lw=2, label=r"Thermal + non-thermal ($\varepsilon_T=0.5$)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title(f"SED at t = {t_sed/DAY:.0f} days")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: varying eps_T
print("  Right panel: varying eps_T...")
ax = axes[1]
eps_T_vals = [0.1, 0.3, 0.5, 1.0]
colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]

for eps_T, color in zip(eps_T_vals, colors):
    P_var = dict(P_base, eps_T=eps_T)
    flux = np.array([jet_sed.FluxDensity(t_sed, float(nu), P_var, model="sync_thermal") for nu in nu_arr])
    m = flux > 0
    if m.any():
        ax.loglog(nu_arr[m], flux[m], "-", color=color, lw=2,
                  label=rf"$\varepsilon_T = {eps_T}$")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title(r"Varying $\varepsilon_T$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle("Thermal Electron Emission — Spectral Energy Distribution", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "thermal_sed.png"), dpi=150, bbox_inches="tight")
print("  Saved thermal_sed.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Figure 2: Thermal light curves (1×2)
# ═══════════════════════════════════════════════════════════════
print("=== Thermal light curves ===")

t_lc = np.geomspace(1 * DAY, 1000 * DAY, 150)
tday = t_lc / DAY

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: radio light curves with/without thermal
print("  Left panel: radio light curves...")
ax = axes[0]
P_th = dict(P_base, eps_T=0.5)
bands = [(1e9, "1 GHz"), (5e9, "5 GHz")]
colors_lc = ["#1f77b4", "#d62728"]

for (nu, nu_label), color in zip(bands, colors_lc):
    flux_nt = FluxDensity_tophat(t_lc, nu, P_base, spread=True, model="sync_ssa_smooth", tmax=1e9)
    flux_th = FluxDensity_tophat(t_lc, nu, P_th, spread=True, model="sync_thermal", tmax=1e9)
    m = flux_nt > 0
    if m.any():
        ax.loglog(tday[m], flux_nt[m], "--", color=color, lw=1.5, alpha=0.6,
                  label=f"Non-thermal ({nu_label})")
    m = flux_th > 0
    if m.any():
        ax.loglog(tday[m], flux_th[m], "-", color=color, lw=2,
                  label=f"Thermal + NT ({nu_label})")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title("Radio Light Curves")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: full-volume vs thin-shell
print("  Right panel: full-volume vs thin-shell...")
ax = axes[1]
P_thin = dict(P_base, eps_T=0.5)
P_full = dict(P_base, eps_T=0.5, full_volume=1.0, k=0.0)

for nu, nu_label, color in [(1e9, "1 GHz", "#1f77b4"), (5e9, "5 GHz", "#d62728")]:
    f_thin = FluxDensity_tophat(t_lc, nu, P_thin, spread=True, model="sync_thermal", tmax=1e9)
    f_full = FluxDensity_tophat(t_lc, nu, P_full, spread=True, model="sync_thermal", tmax=1e9)
    m = f_thin > 0
    if m.any():
        ax.loglog(tday[m], f_thin[m], "--", color=color, lw=1.5, alpha=0.6,
                  label=f"Thin-shell ({nu_label})")
    m = f_full > 0
    if m.any():
        ax.loglog(tday[m], f_full[m], "-", color=color, lw=2,
                  label=f"Full-volume ({nu_label})")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title("Full-Volume (FM25) vs Thin-Shell")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("Thermal Electron Emission — Light Curves", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "thermal_lightcurves.png"), dpi=150, bbox_inches="tight")
print("  Saved thermal_lightcurves.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Figure 3: Comparison with FM25 reference code (1×2)
# ═══════════════════════════════════════════════════════════════
print("=== FM25 comparison ===")

# Add FM25 code directory to path
fm25_dir = "/fred/oz480/mcoughli/arXiv-2509.16313v1/synchrotron_shock_model"
if os.path.isdir(fm25_dir):
    sys.path.insert(0, fm25_dir)
    import thermalsyn_v2 as fm25

    # Strategy: run blastwave with reasonable parameters, extract the fluid
    # state at a specific time, then call FM25 with matching shock conditions.
    eps_T_cmp = 0.5
    eps_B_cmp = 0.1
    eps_e_cmp = 0.01
    p_cmp = 3.0
    n0_cmp = 1.0
    d_L_cm = d_L * MPC_CM

    # Run blastwave with a mildly relativistic spherical outflow
    Eiso_cmp = 1e52
    Gamma0_cmp = 10.0

    jet_cmp = Jet(
        Spherical(Eiso_cmp, lf0=Gamma0_cmp), 0.0, n0_cmp,
        tmin=10.0, tmax=1e9,
        grid=Uniform(17), tail=False, spread=False, spread_mode="none",
        k=0.0, cal_level=1, rtol=1e-6,
        eps_e=eps_e_cmp, eps_b=eps_B_cmp, p_fwd=p_cmp,
    )

    # Pick a time when blast is trans-relativistic (bG ~ 1)
    # Stay safely within simulation bounds
    t_sample = np.geomspace(100, 5e8, 500)
    bG_sample = np.zeros(len(t_sample))
    for i, ts in enumerate(t_sample):
        try:
            bG_sample[i] = jet_cmp.beta_gamma(float(ts), 0.0)
        except Exception:
            bG_sample[i] = 0.0

    # Find time closest to bG_fluid ~ 1
    target_bG = 1.0
    mask_valid = bG_sample > 0
    if mask_valid.sum() > 10:
        from scipy.interpolate import interp1d
        log_bG = np.log10(bG_sample[mask_valid])
        log_t = np.log10(t_sample[mask_valid])
        interp_t_of_bG = interp1d(log_bG, log_t, bounds_error=False, fill_value="extrapolate")
        t_match = 10**float(interp_t_of_bG(np.log10(target_bG)))
    else:
        t_match = 1e7  # fallback

    bG_fluid = jet_cmp.beta_gamma(float(t_match), 0.0)
    R_match = jet_cmp.R(float(t_match), 0.0)
    print(f"  blastwave at t={t_match:.2e} s: bG_fluid={bG_fluid:.4f}, R={R_match:.3e} cm")

    # Convert fluid bG to shock bG
    # FM25 relation: bG_fluid = 0.5*(bG_sh^2 - 2 + sqrt(bG_sh^4 + 5*bG_sh^2 + 4))^0.5
    # Invert numerically: given bG_f, find bG_sh
    from scipy.optimize import brentq

    def _bG_fluid_from_shock(bG_sh):
        return 0.5 * (bG_sh**2 - 2.0 + (bG_sh**4 + 5.0 * bG_sh**2 + 4.0)**0.5)**0.5

    bG_sh = brentq(lambda x: _bG_fluid_from_shock(x) - bG_fluid, 0.01, 100.0)
    print(f"  Inverted: bG_sh={bG_sh:.4f}")

    # Compute observer time for FM25 (using radius_insteadof_time for direct match)
    nu_compare = np.geomspace(1e6, 1e15, 300)

    # FM25 SED at the same shock conditions
    print("  Computing FM25 SED...")
    Fnu_fm25 = fm25.Fnu_of_nu(
        bG_sh, n0_cmp, nu_compare, R_match,
        Dlum=d_L_cm, z=z,
        density_insteadof_massloss=True,
        radius_insteadof_time=True,
        epsilon_T=eps_T_cmp, epsilon_B=eps_B_cmp, epsilon_e=eps_e_cmp, p=p_cmp,
    )
    # FM25 returns erg/cm^2/s/Hz, convert to mJy
    Fnu_fm25_mJy = Fnu_fm25 / 1e-26

    # blastwave SED at matched time
    P_cmp = dict(
        Eiso=Eiso_cmp, lf=Gamma0_cmp,
        n0=n0_cmp, A=0.0,
        eps_e=eps_e_cmp, eps_b=eps_B_cmp, p=p_cmp,
        theta_v=0.0, d=d_L, z=z,
        eps_T=eps_T_cmp, full_volume=1.0, k=0.0,
    )

    print("  Computing blastwave SED...")
    Fnu_bw = np.array([
        jet_cmp.FluxDensity(t_match, float(nu), P_cmp, model="sync_thermal")
        for nu in nu_compare
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: FM25 SED
    ax = axes[0]
    m = Fnu_fm25_mJy > 0
    if m.any():
        ax.loglog(nu_compare[m], Fnu_fm25_mJy[m], "-", color="#d62728", lw=2,
                  label="FM25 thermalsyn_v2")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Flux density (mJy)")
    ax.set_title(rf"FM25 ($\beta\Gamma_{{sh}}={bG_sh:.2f}$, $n={n0_cmp}$ cm$^{{-3}}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: blastwave overlaid with FM25
    ax = axes[1]
    m_bw = Fnu_bw > 0
    if m_bw.any():
        ax.loglog(nu_compare[m_bw], Fnu_bw[m_bw], "-", color="#1f77b4", lw=2,
                  label="blastwave sync_thermal")
    if m.any():
        ax.loglog(nu_compare[m], Fnu_fm25_mJy[m], "--", color="#d62728", lw=1.5,
                  alpha=0.7, label="FM25 (reference)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Flux density (mJy)")
    ax.set_title(rf"blastwave vs FM25 ($\beta\Gamma_{{fluid}}={bG_fluid:.2f}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        rf"FM25 Comparison ($\varepsilon_T={eps_T_cmp}$, $\varepsilon_B={eps_B_cmp}$, "
        rf"$\varepsilon_e={eps_e_cmp}$, $p={p_cmp}$)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "thermal_fm25.png"), dpi=150, bbox_inches="tight")
    print("  Saved thermal_fm25.png")
    plt.close()
else:
    print(f"  FM25 directory not found at {fm25_dir}, skipping comparison plot.")
    print("  To generate thermal_fm25.png, ensure the FM25 code is available.")

print("\nDone!")
