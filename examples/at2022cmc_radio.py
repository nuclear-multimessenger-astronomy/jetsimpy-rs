"""
AT2022cmc: Modeling a relativistic TDE with a spherical blast wave.

Radio data from Rhodes+2025 (arXiv:2506.13618, ApJ accepted).
Redshift: z = 1.193, Discovery: 2022 Feb 11 (MJD 59621.4458).
Cosmology: H0 = 70 km/s/Mpc, Omega_M = 0.3.

The paper's best-fit model is a spherical outflow with thermal + non-thermal
electrons (MQ21/FM25 formalism). This script compares non-thermal only
(sync_ssa) vs thermal+non-thermal (sync_thermal) to show the effect of
thermal electrons on the high-frequency spectral index.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.integrate import quad
import sys
sys.path.insert(0, "..")
from blastwave import FluxDensity_spherical

# ---------- Constants ----------
DAY = 86400.0  # seconds per day
T0 = datetime(2022, 2, 11)  # discovery date


def date_to_days(date_str):
    """Convert 'dd/mm/yyyy' to days since discovery."""
    dt = datetime.strptime(date_str, "%d/%m/%Y")
    return (dt - T0).days


def luminosity_distance(z, H0=70.0, Om=0.3):
    """Flat LCDM luminosity distance in Mpc."""
    OL = 1.0 - Om
    c_km_s = 299792.458
    result, _ = quad(lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + OL), 0, z)
    return (c_km_s / H0) * (1 + z) * result


# ---------- Cosmology ----------
z = 1.193
d_L = luminosity_distance(z)
print(f"AT2022cmc: z = {z}, d_L = {d_L:.0f} Mpc")

# ---------- Radio data from Rhodes+2025, Table 1 ----------
# Format: (date_str, freq_GHz, flux_mJy, err_mJy)
# Upper limits (prefixed with <) are excluded.
# For same-day duplicate AMI-LA observations, we include both.

raw_data = [
    # e-MERLIN 5 GHz
    ("24/05/2022", 5.0, 0.08, 0.02),
    ("17/06/2022", 5.0, 0.12, 0.02),
    ("10/07/2022", 5.0, 0.17, 0.02),
    ("03/10/2023", 5.0, 0.35, 0.02),
    ("18/11/2023", 5.0, 0.43, 0.03),
    ("24/01/2024", 5.0, 0.31, 0.02),
    ("31/03/2024", 5.0, 0.28, 0.02),
    ("01/06/2024", 5.0, 0.25, 0.02),
    ("08/08/2024", 5.0, 0.22, 0.02),
    # AMI-LA 15.5 GHz (detections only, averaged same-day pairs)
    ("31/05/2022", 15.5, 0.90, 0.07),
    ("02/06/2022", 15.5, 1.12, 0.07),
    ("05/06/2022", 15.5, 1.06, 0.09),
    ("06/06/2022", 15.5, 1.17, 0.07),
    ("11/06/2022", 15.5, 0.95, 0.10),
    ("24/06/2022", 15.5, 1.10, 0.10),
    ("26/06/2022", 15.5, 0.93, 0.07),
    ("04/07/2022", 15.5, 1.02, 0.09),
    ("05/07/2022", 15.5, 0.91, 0.12),
    ("23/08/2022", 15.5, 1.12, 0.09),
    ("27/08/2022", 15.5, 1.08, 0.09),
    ("01/09/2022", 15.5, 1.19, 0.09),
    ("08/09/2022", 15.5, 0.98, 0.08),
    ("10/09/2022", 15.5, 1.09, 0.07),
    ("07/10/2022", 15.5, 1.18, 0.07),
    ("10/10/2022", 15.5, 0.93, 0.07),
    ("13/10/2022", 15.5, 1.19, 0.07),
    ("16/10/2022", 15.5, 1.06, 0.06),
    ("23/10/2022", 15.5, 1.10, 0.20),
    ("12/11/2022", 15.5, 1.01, 0.06),
    ("14/11/2022", 15.5, 0.97, 0.09),
    ("21/11/2022", 15.5, 1.01, 0.06),
    ("01/12/2022", 15.5, 0.94, 0.07),
    ("05/12/2022", 15.5, 0.88, 0.07),
    ("19/12/2022", 15.5, 1.01, 0.09),
    ("27/12/2022", 15.5, 0.96, 0.06),
    ("17/01/2023", 15.5, 0.89, 0.06),
    ("20/01/2023", 15.5, 0.98, 0.07),
    ("05/02/2023", 15.5, 0.80, 0.05),
    ("09/02/2023", 15.5, 0.91, 0.06),
    ("12/02/2023", 15.5, 0.84, 0.06),
    ("04/03/2023", 15.5, 0.66, 0.04),
    ("10/03/2023", 15.5, 0.50, 0.10),
    ("28/03/2023", 15.5, 0.64, 0.04),
    ("06/04/2023", 15.5, 0.52, 0.05),
    ("07/04/2023", 15.5, 0.69, 0.05),
    ("15/04/2023", 15.5, 0.50, 0.05),
    ("17/04/2023", 15.5, 0.42, 0.05),
    ("21/04/2023", 15.5, 0.58, 0.04),
    ("24/04/2023", 15.5, 0.52, 0.04),
    ("05/05/2023", 15.5, 0.38, 0.05),
    ("08/05/2023", 15.5, 0.51, 0.08),
    ("10/05/2023", 15.5, 0.43, 0.04),
    ("11/05/2023", 15.5, 0.55, 0.06),
    ("14/05/2023", 15.5, 0.45, 0.04),
    ("01/06/2023", 15.5, 0.46, 0.04),
    ("04/06/2023", 15.5, 0.34, 0.07),
    ("08/07/2023", 15.5, 0.23, 0.07),
    ("10/07/2023", 15.5, 0.39, 0.08),
    ("21/07/2023", 15.5, 0.26, 0.04),
    ("12/08/2023", 15.5, 0.34, 0.07),
    ("02/09/2023", 15.5, 0.30, 0.08),
    ("16/09/2023", 15.5, 0.20, 0.05),
    ("23/09/2023", 15.5, 0.26, 0.05),
    ("29/09/2023", 15.5, 0.27, 0.07),
    ("07/10/2023", 15.5, 0.22, 0.07),
    ("15/10/2023", 15.5, 0.24, 0.04),
    ("06/11/2023", 15.5, 0.21, 0.04),
    ("12/11/2023", 15.5, 0.33, 0.06),
    ("26/11/2023", 15.5, 0.23, 0.04),
    ("18/12/2023", 15.5, 0.17, 0.04),
    ("07/01/2024", 15.5, 0.13, 0.03),
    ("14/01/2024", 15.5, 0.17, 0.04),
    ("28/01/2024", 15.5, 0.21, 0.05),
    ("14/12/2024", 15.5, 0.10, 0.03),
    ("01/08/2024", 15.5, 0.10, 0.03),
    ("11/04/2025", 15.5, 0.11, 0.03),
    # NOEMA 86.25 GHz
    ("07/06/2022", 86.25, 1.60, 0.20),
    ("19/06/2022", 86.25, 1.20, 0.10),
    ("06/07/2022", 86.25, 0.78, 0.08),
    ("21/07/2022", 86.25, 0.90, 0.10),
    ("03/08/2022", 86.25, 0.90, 0.10),
    ("20/08/2022", 86.25, 0.51, 0.07),
    ("10/09/2022", 86.25, 0.43, 0.05),
    ("29/09/2022", 86.25, 0.32, 0.05),
    ("15/10/2022", 86.25, 0.11, 0.04),
    ("26/10/2022", 86.25, 0.31, 0.05),
    ("31/10/2022", 86.25, 0.25, 0.05),
    ("18/11/2022", 86.25, 0.18, 0.03),
    ("27/12/2022", 86.25, 0.14, 0.02),
    ("16/02/2023", 86.25, 0.11, 0.01),
    ("23/04/2023", 86.25, 0.08, 0.02),
    # NOEMA 101.75 GHz
    ("07/06/2022", 101.75, 1.30, 0.10),
    ("19/06/2022", 101.75, 0.90, 0.10),
    ("06/07/2022", 101.75, 0.63, 0.07),
    ("21/07/2022", 101.75, 0.81, 0.09),
    ("03/08/2022", 101.75, 0.60, 0.10),
    ("20/08/2022", 101.75, 0.36, 0.06),
    ("10/09/2022", 101.75, 0.30, 0.04),
    ("29/09/2022", 101.75, 0.28, 0.05),
    ("15/10/2022", 101.75, 0.19, 0.05),
    ("26/10/2022", 101.75, 0.39, 0.06),
    ("31/10/2022", 101.75, 0.18, 0.04),
    ("27/12/2022", 101.75, 0.11, 0.02),
    ("16/02/2023", 101.75, 0.07, 0.01),
    ("23/04/2023", 101.75, 0.06, 0.02),
    # MeerKAT 1.28 GHz (detections only)
    ("17/02/2023", 1.28, 0.09, 0.01),
    ("28/09/2023", 1.28, 0.10, 0.01),
    ("28/06/2024", 1.28, 0.11, 0.01),
    ("10/08/2024", 1.28, 0.09, 0.02),
    ("05/10/2024", 1.28, 0.13, 0.02),
    # MeerKAT 3 GHz
    ("06/10/2023", 3.0, 0.17, 0.02),
    ("04/02/2024", 3.0, 0.30, 0.04),
    ("28/06/2024", 3.0, 0.39, 0.04),
    ("10/08/2024", 3.0, 0.33, 0.04),
    ("05/10/2024", 3.0, 0.22, 0.03),
]

# Organize data by frequency
freq_groups = {}
for date_str, freq_ghz, flux, err in raw_data:
    days = date_to_days(date_str)
    key = freq_ghz
    if key not in freq_groups:
        freq_groups[key] = []
    freq_groups[key].append([days, flux, err])

for key in freq_groups:
    freq_groups[key] = np.array(freq_groups[key])

# ---------- Model parameters ----------
# Paper best-fit spherical model:
#   (Gamma*beta)_sh = 1.795 (t/45d)^{-0.288}
#   n = 191 (R/R_45)^{-1.795}, R_45 = 9.4e17 cm
#   k ~ 1.8, eps_e = 0.1, eps_B = 0.1, p = 2.79
#   eps_T = 0.4 (thermal electron fraction)
#
# Translating density to blastwave convention:
#   n(r) = nwind * (r/1e17)^{-k}
#   At r = R_45 = 9.4e17: n = 191
#   nwind = 191 * 9.4^1.8 ~ 10770

nwind = 191.0 * (9.4)**1.8  # ~ 10770

P = {
    "Eiso": 1e52,
    "lf":   10.0,            # initially relativistic
    "A":    nwind,
    "n0":   0.0,             # wind-dominated
    "eps_e": 0.1,
    "eps_b": 0.1,
    "eps_T": 0.4,            # thermal electron fraction (Rhodes+2025)
    "p":     2.79,
    "theta_v": 0.0,
    "d":     d_L,
    "z":     z,
}

print(f"nwind = {nwind:.0f} cm^-3")
print(f"d_L = {d_L:.0f} Mpc")
print(f"Parameters: E={P['Eiso']:.1e}, Gamma0={P['lf']}, p={P['p']}, eps_T={P['eps_T']}")

# ---------- Compute models ----------
t_model = np.geomspace(50 * DAY, 1200 * DAY, 200)

freq_list = [1.28e9, 3e9, 5e9, 15.5e9, 86.25e9, 101.75e9]
freq_labels = ['1.28 GHz', '3 GHz', '5 GHz', '15.5 GHz', '86.25 GHz', '101.75 GHz']
colors = ['C3', 'C4', 'C0', 'C1', 'C2', 'C5']
markers = ['v', 'D', 's', 'o', '^', 'x']

# Non-thermal only (sync_ssa)
F_nonthermal = {}
print("\n--- Non-thermal only (sync_ssa) ---")
for nu_hz, label in zip(freq_list, freq_labels):
    print(f"  Computing {label}...")
    F_nonthermal[nu_hz] = FluxDensity_spherical(
        t_model, nu_hz * np.ones_like(t_model), P,
        k=1.8, tmin=1.0, tmax=1500 * DAY,
        model="sync_ssa",
    )

# Thermal + non-thermal (sync_thermal)
F_thermal = {}
print("\n--- Thermal + non-thermal (sync_thermal) ---")
for nu_hz, label in zip(freq_list, freq_labels):
    print(f"  Computing {label}...")
    F_thermal[nu_hz] = FluxDensity_spherical(
        t_model, nu_hz * np.ones_like(t_model), P,
        k=1.8, tmin=1.0, tmax=1500 * DAY,
        model="sync_thermal",
    )

# ---------- Plot: side-by-side comparison ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
t_days = t_model / DAY

for ax, F_model, title in zip(
    axes,
    [F_nonthermal, F_thermal],
    ['Non-thermal only (sync_ssa)', 'Thermal + non-thermal (sync_thermal, $\\epsilon_T=0.4$)']
):
    for nu_hz, freq_ghz, label, color, marker in zip(
        freq_list,
        [1.28, 3.0, 5.0, 15.5, 86.25, 101.75],
        freq_labels, colors, markers
    ):
        # Data
        if freq_ghz in freq_groups:
            d = freq_groups[freq_ghz]
            ax.errorbar(d[:, 0], d[:, 1], yerr=d[:, 2],
                        fmt=marker, color=color, label=f'{label} data',
                        capsize=2, ms=4, alpha=0.7)
        # Model
        ax.plot(t_days, F_model[nu_hz], '-', color=color, alpha=0.6, lw=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time since discovery (days)', fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(50, 1200)
    ax.set_ylim(0.01, 5)

axes[0].set_ylabel('Flux density (mJy)', fontsize=12)
axes[0].legend(fontsize=7, ncol=2, loc='upper right')

fig.suptitle(
    f'AT2022cmc — Spherical blast wave\n'
    f'$E_{{iso}}$={P["Eiso"]:.0e} erg, $\\Gamma_0$={P["lf"]:.0f}, '
    f'$k$=1.8, $n_{{wind}}$={P["A"]:.0f}, '
    f'$\\epsilon_e$={P["eps_e"]}, $\\epsilon_B$={P["eps_b"]:.2g}, $p$={P["p"]}',
    fontsize=12, y=1.02,
)
plt.tight_layout()
plt.savefig('tests/at2022cmc_radio.png', dpi=150, bbox_inches='tight')
print("\nSaved tests/at2022cmc_radio.png")
plt.show()
