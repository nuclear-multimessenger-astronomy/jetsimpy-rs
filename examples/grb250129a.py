"""
GRB 250129A — Shell collision model with internal shock emission.

Reproduces the multi-band afterglow of GRB 250129A (z=2.151) using a
forward shock + trailing shell collision. The trailing shell deposits
energy into the blast wave and produces internal shock synchrotron emission,
explaining the early X-ray/optical plateau.

Reference: Akl et al. 2026, A&A (arXiv:2603.08555)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from blastwave import Jet, TopHat, ForwardJetRes

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
})

outdir = os.path.join(os.path.dirname(__file__), "..", "docs", "examples", "img")
os.makedirs(outdir, exist_ok=True)

DAY = 86400.0

# ═══════════════════════════════════════════════════════════════
# Observational data
# ═══════════════════════════════════════════════════════════════
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "grb250129a.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = None
    print(f"Data file not found: {data_path}")

# ═══════════════════════════════════════════════════════════════
# Model parameters
# ═══════════════════════════════════════════════════════════════
z = 2.151
d_mpc = 5145.0  # luminosity distance [Mpc]

# Afterglow (forward shock)
E_iso = 2e52        # erg
Gamma0 = 80
n_ism = 2.0         # cm^-3
eps_e = 0.07
eps_b = 2e-3
p = 2.5
theta_j = np.deg2rad(12)
theta_obs = np.deg2rad(0.5)

# Trailing shell
shell_E = 1.5e52    # erg
shell_Gamma = 100
shell_t_launch = 5000  # s

# Collision microphysics
eps_e_coll = 0.3
eps_b_coll = 0.01
p_coll = 2.5

# ═══════════════════════════════════════════════════════════════
# Build jets
# ═══════════════════════════════════════════════════════════════
print("Building models...")

common = dict(
    nwind=0.0, nism=n_ism,
    tmin=1.0, tmax=200 * DAY, spread_mode="ode",
    eps_e=eps_e, eps_b=eps_b, p_fwd=p,
    grid=ForwardJetRes(theta_j, 129),
)
profile = TopHat(theta_j, E_iso, lf0=Gamma0)

# Afterglow only (no shell)
jet_base = Jet(profile, **common)

# Shell collision + internal shock emission
jet_shell = Jet(
    profile,
    trailing_shells=[(shell_E, shell_Gamma, shell_t_launch)],
    eps_e_coll=eps_e_coll, eps_b_coll=eps_b_coll, p_coll=p_coll,
    **common,
)

P = dict(
    theta_obs=theta_obs, theta_v=theta_obs,
    Eiso=E_iso, lf=Gamma0, theta_c=theta_j,
    n0=n_ism, A=0.0, p=p, eps_e=eps_e, eps_b=eps_b,
    d=d_mpc, z=z,
)

t_days = np.geomspace(1e-3, 100, 400)
t_sec = t_days * DAY

# ═══════════════════════════════════════════════════════════════
# Figure 1: Multi-band light curves (2×2)
# ═══════════════════════════════════════════════════════════════
print("Computing light curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Filter → frequency mapping
filter_info = {
    "bessellr": ("R", 4.68e14, "red"),
    "bessellv": ("V", 5.51e14, "green"),
    "besselli": ("I", 3.76e14, "orange"),
    "bessellb": ("B", 6.84e14, "royalblue"),
    "sdssr": ("r'", 4.87e14, "darkblue"),
    "sdssi": ("i'", 4.00e14, "purple"),
    "sdssz": ("z'", 3.46e14, "brown"),
    "sdssg": ("g'", 6.23e14, "teal"),
}
uvot_filters = {
    "uvot::u": "u", "uvot::b": "b", "uvot::v": "v",
    "uvot::white": "white", "uvot::uvw1": "UVW1",
    "uvot::uvw2": "UVW2", "uvot::uvm2": "UVM2",
}

# Panel 1: Optical
ax = axes[0, 0]
if df is not None:
    for filt, (label, nu, color) in filter_info.items():
        sub = df[df["filter"] == filt].dropna(subset=["corrected_flux_ujy"])
        if len(sub) == 0:
            continue
        ax.scatter(sub["T-T0_days"].values, sub["corrected_flux_ujy"].values / 1e3,
                   s=15, alpha=0.6, c=color, label=label, zorder=5)
    # UVOT (gray background)
    for filt in df["filter"].unique():
        if "uvot" in filt.lower():
            sub = df[df["filter"] == filt].dropna(subset=["corrected_flux_ujy"])
            if len(sub) > 0:
                ax.scatter(sub["T-T0_days"].values, sub["corrected_flux_ujy"].values / 1e3,
                           s=6, alpha=0.2, c="gray", zorder=2)

f_r_base = jet_base.FluxDensity(t_sec, 4.68e14, P)
f_r_shell = jet_shell.FluxDensity(t_sec, 4.68e14, P)
ax.loglog(t_days, f_r_base, "k--", lw=1, alpha=0.5, label="Afterglow only")
ax.loglog(t_days, f_r_shell, "r-", lw=2, label="+ Shell collision")
ax.set_xlim(1e-3, 100)
ax.set_ylim(1e-4, 10)
ax.set_xlabel("Time [days]")
ax.set_ylabel("Flux density [mJy]")
ax.set_title("Optical (R-band model)")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 2: X-ray
ax = axes[0, 1]
if df is not None:
    sub = df[df["filter"] == "2keV"].dropna(subset=["corrected_flux_ujy"])
    if len(sub) > 0:
        ax.scatter(sub["T-T0_days"].values, sub["corrected_flux_ujy"].values / 1e3,
                   s=30, c="red", marker="s", label="XRT 2 keV", zorder=5)

f_x_base = jet_base.FluxDensity(t_sec, 4.84e17, P)
f_x_shell = jet_shell.FluxDensity(t_sec, 4.84e17, P)
ax.loglog(t_days, f_x_base, "k--", lw=1, alpha=0.5, label="Afterglow only")
ax.loglog(t_days, f_x_shell, "r-", lw=2, label="+ Shell collision")
ax.set_xlim(1e-3, 100)
ax.set_ylim(1e-7, 0.1)
ax.set_xlabel("Time [days]")
ax.set_ylabel("Flux density [mJy]")
ax.set_title("X-ray (2 keV)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Radio predictions
ax = axes[1, 0]
radio_bands = [(1, "1 GHz", "blue"), (5, "5 GHz", "green"),
               (10, "10 GHz", "orange"), (100, "100 GHz", "red")]
for nu_ghz, label, color in radio_bands:
    f = jet_shell.FluxDensity(t_sec, nu_ghz * 1e9, P)
    f_base = jet_base.FluxDensity(t_sec, nu_ghz * 1e9, P)
    ax.loglog(t_days, f, "-", c=color, lw=2, label=f"{label} (shell)")
    ax.loglog(t_days, f_base, "--", c=color, lw=1, alpha=0.3)
ax.set_xlim(1e-3, 100)
ax.set_xlabel("Time [days]")
ax.set_ylabel("Flux density [mJy]")
ax.set_title("Radio predictions")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Broadband SED at 0.1, 1, 10 days
ax = axes[1, 1]
nu_arr = np.geomspace(1e8, 1e19, 150)
for t_d, color in [(0.1, "blue"), (1.0, "green"), (10.0, "red")]:
    t_s = np.array([t_d * DAY])
    sed = np.array([jet_shell.FluxDensity(t_s, float(nu), P)[0] for nu in nu_arr])
    m = sed > 0
    if m.any():
        ax.loglog(nu_arr[m], sed[m], "-", c=color, lw=2, label=f"t = {t_d} d")
    # Data near this epoch
    if df is not None:
        for filt, nu_filt in [("bessellr", 4.68e14), ("2keV", 4.84e17)]:
            sub = df[df["filter"] == filt].dropna(subset=["corrected_flux_ujy"])
            near = sub[(sub["T-T0_days"] > t_d * 0.5) & (sub["T-T0_days"] < t_d * 2)]
            if len(near) > 0:
                ax.scatter([nu_filt] * len(near), near["corrected_flux_ujy"].values / 1e3,
                           s=20, c=color, marker="x", alpha=0.5, zorder=5)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Flux density [mJy]")
ax.set_title("Broadband SED")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle("GRB 250129A — Shell Collision Model (z = 2.151)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "grb250129a_multiband.png"), dpi=150, bbox_inches="tight")
print("Saved grb250129a_multiband.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Figure 2: Collision effect breakdown (1×2)
# ═══════════════════════════════════════════════════════════════
print("Computing collision breakdown...")

# Energy injection only (no IS emission)
jet_inject = Jet(
    profile,
    trailing_shells=[(shell_E, shell_Gamma, shell_t_launch)],
    eps_e_coll=0.0,
    **common,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, nu, title in [(ax1, 4.68e14, "Optical (R-band)"), (ax2, 4.84e17, "X-ray (2 keV)")]:
    f_base = jet_base.FluxDensity(t_sec, nu, P)
    f_inject = jet_inject.FluxDensity(t_sec, nu, P)
    f_full = jet_shell.FluxDensity(t_sec, nu, P)

    ax.loglog(t_days, f_base, "k--", lw=1, alpha=0.5, label="Afterglow only")
    ax.loglog(t_days, f_inject, "b-", lw=1.5, label="+ Energy injection")
    ax.loglog(t_days, f_full, "r-", lw=2, label="+ IS emission")

    # Data
    if df is not None:
        if nu < 1e15:
            sub = df[df["filter"] == "bessellr"].dropna(subset=["corrected_flux_ujy"])
            if len(sub) > 0:
                ax.scatter(sub["T-T0_days"].values, sub["corrected_flux_ujy"].values / 1e3,
                           s=12, alpha=0.4, c="red", zorder=5)
        else:
            sub = df[df["filter"] == "2keV"].dropna(subset=["corrected_flux_ujy"])
            if len(sub) > 0:
                ax.scatter(sub["T-T0_days"].values, sub["corrected_flux_ujy"].values / 1e3,
                           s=25, c="red", marker="s", zorder=5)

    ax.set_xlim(1e-3, 100)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Flux density [mJy]")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax1.set_ylim(1e-4, 10)
ax2.set_ylim(1e-7, 0.1)

fig.suptitle("Energy Injection vs Internal Shock Emission", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "grb250129a_breakdown.png"), dpi=150, bbox_inches="tight")
print("Saved grb250129a_breakdown.png")
plt.close()

print("Done!")
