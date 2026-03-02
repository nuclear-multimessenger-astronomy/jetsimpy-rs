"""
GRB 140304A: Forward shock energy injection (Laskar et al. 2018, arXiv:1707.05784).

Demonstrates the magnetar spin-down energy injection model by modeling the
multi-wavelength afterglow of GRB 140304A at z=5.283 in a wind medium.

Note: Laskar+2017 model the injection as a discrete shell collision (E ~ t^3.8
between 0.15-0.26 d), not a continuous magnetar spin-down. Our magnetar model
L(t) = L0*(1+t/t0)^(-q) injects from t=0, so it approximates but cannot exactly
reproduce the sudden onset. We show three models:
  (1) No injection (pre-injection E = 6.1e52 erg)
  (2) Magnetar injection (L0 = 1e49, t0 = 5e4 s)
  (3) Static post-injection energy (E = 4.9e54 erg)

Produces three figures:
  1. Radio light curves: 3-model comparison with VLA data
  2. Radio SEDs at multiple epochs
  3. X-ray light curve with Swift/XRT data
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")

from blastwave import Jet, TopHat, ForwardJetRes

# ── Cosmology: luminosity distance for z=5.283 ──
z = 5.283
H0_cgs = 68.0 * 1e5 / 3.086e24  # H0 = 68 km/s/Mpc in s^-1
Om, OL = 0.31, 0.69
zz = np.linspace(0, z, 10000)
integrand = 1.0 / np.sqrt(Om * (1 + zz)**3 + OL)
d_C = 2.998e10 / H0_cgs * np.trapezoid(integrand, zz)
d_L_Mpc = d_C * (1 + z) / 3.086e24
print(f"z = {z}, d_L = {d_L_Mpc:.0f} Mpc")

# ── Wind density: A* = 0.026 → nwind = A* × 29.9 ──
A_star = 0.026
nwind = A_star * 29.9  # ≈ 0.78

# ── Forward shock parameters (Table 5, Laskar+2017) ──
theta_c = 0.0197  # 1.13 deg
Eiso_pre = 6.1e52
Eiso_post = 4.9e54

P = dict(
    Eiso=Eiso_pre, lf=300, theta_c=theta_c,
    A=nwind, n0=0.0, p=2.59,
    eps_e=0.025, eps_b=0.059,
    theta_v=0.0, z=z, d=d_L_Mpc,
)

# Magnetar injection: moderate L0 to avoid overwhelming early-time dynamics
# E_inj(inf) = L0 * t0 = 1e49 * 5e4 = 5e53 erg → total ~5.6e53 (~9x E0)
mag_l0 = 1e49    # erg/s
mag_t0 = 5e4     # s (~0.58 d)
mag_q = 2.0      # magnetic dipole

# ── Helper: create jet with given Eiso and optional magnetar ──
def make_jet(eiso, magnetar_l0=0.0):
    return Jet(
        TopHat(theta_c, eiso, lf0=300),
        nwind, 0.0,
        tmin=10.0, tmax=1e9,
        grid=ForwardJetRes(theta_c, 33),
        tail=True, spread=False, k=2.0,
        magnetar_l0=magnetar_l0, magnetar_t0=mag_t0, magnetar_q=mag_q,
    )

print("Creating jets...")
jet_noinj = make_jet(Eiso_pre)              # no injection (pre-injection E)
jet_mag = make_jet(Eiso_pre, mag_l0)        # magnetar injection
jet_post = make_jet(Eiso_post)              # static post-injection E

P_post = {**P, 'Eiso': Eiso_post}

# ── Radio data (Laskar+2017 Table 4) ──
radio_data = np.array([
    [0.450,  4.90,  0.0390, 0.0120, 1],
    [0.450,  7.00,  0.0730, 0.0110, 1],
    [0.544, 85.50,  0.6560, 0.2350, 1],
    [1.496, 85.50,  0.8170, 0.2300, 1],
    [1.540,  8.55,  0.2560, 0.0150, 1],
    [1.540, 11.00,  0.2140, 0.0150, 1],
    [1.540, 13.50,  0.1968, 0.0141, 1],
    [1.540, 16.00,  0.2046, 0.0152, 1],
    [1.550, 19.20,  0.2311, 0.0252, 1],
    [1.550, 24.50,  0.3076, 0.0281, 1],
    [1.550, 30.00,  0.3840, 0.0400, 1],
    [1.550, 37.00,  0.4690, 0.0550, 1],
    [1.560,  4.90,  0.2060, 0.0150, 1],
    [1.560,  7.00,  0.2980, 0.0110, 1],
    [4.550,  4.90,  0.0280, 0.0140, 1],
    [4.550,  7.00,  0.0560, 0.0110, 1],
    [4.550,  8.55,  0.1230, 0.0150, 1],
    [4.550, 11.00,  0.2410, 0.0170, 1],
    [4.550, 13.50,  0.2800, 0.0130, 1],
    [4.550, 16.00,  0.3230, 0.0130, 1],
    [4.550, 19.20,  0.3330, 0.0180, 1],
    [4.550, 24.50,  0.3020, 0.0210, 1],
    [4.550, 30.00,  0.2360, 0.0280, 1],
    [4.550, 37.00,  0.2350, 0.0400, 1],
    [8.550,  4.90,  0.1120, 0.0160, 1],
    [8.550,  7.00,  0.1660, 0.0120, 1],
    [8.550,  8.55,  0.1870, 0.0160, 1],
    [8.550, 11.00,  0.1920, 0.0160, 1],
    [8.550, 13.50,  0.1960, 0.0130, 1],
    [8.550, 16.00,  0.1890, 0.0130, 1],
    [8.550, 19.20,  0.2020, 0.0200, 1],
    [8.550, 24.50,  0.2040, 0.0230, 1],
    [8.550, 30.00,  0.2230, 0.0260, 1],
    [8.550, 37.00,  0.2270, 0.0240, 1],
    [18.22, 19.20,  0.0790, 0.0210, 1],
    [18.22, 24.50,  0.0460, 0.0220, 1],
    [18.23, 13.50,  0.0930, 0.0120, 1],
    [18.23, 16.00,  0.0800, 0.0140, 1],
    [18.25,  8.55,  0.0900, 0.0140, 1],
    [18.25, 11.00,  0.0910, 0.0170, 1],
    [18.26,  4.90,  0.0780, 0.0140, 1],
    [18.26,  7.00,  0.0630, 0.0120, 1],
    [18.28,  2.77,  0.0337, 0.0222, 1],
    [18.28,  3.38,  0.0595, 0.0160, 1],
    [41.33, 19.20,  0.0720, 0.0200, 1],
    [41.33, 24.50,  0.0640, 0.0230, 1],
    [41.35, 13.50,  0.0410, 0.0120, 1],
    [41.35, 16.00,  0.0730, 0.0130, 1],
    [41.36,  8.55,  0.0650, 0.0140, 1],
    [41.36, 11.00,  0.0330, 0.0130, 1],
    [41.37,  4.90,  0.0480, 0.0140, 1],
    [41.37,  7.00,  0.0450, 0.0110, 1],
])
detected = radio_data[:, 4].astype(bool)

# ── XRT data (after orbital gap, spectrum 2) ──
xrt_time_s = np.array([
    5301.953, 5550.022, 5755.750, 6033.748, 6369.004,
    6688.672, 7105.618, 11055.226, 12007.391,
    22193.176, 22363.878, 22559.355, 22756.780, 22960.408,
])
xrt_flux_band = np.array([
    2.983e-12, 3.131e-12, 3.473e-12, 2.131e-12, 1.968e-12,
    1.883e-12, 2.195e-12, 8.229e-13, 9.809e-13,
    5.244e-12, 3.228e-12, 5.245e-12, 2.758e-12, 3.251e-12,
])
xrt_flux_err = np.array([
    7.304e-13, 8.186e-13, 9.037e-13, 5.593e-13, 5.229e-13,
    4.983e-13, 4.672e-13, 2.177e-13, 2.199e-13,
    1.186e-12, 8.444e-13, 1.179e-12, 7.200e-13, 8.211e-13,
])

# Convert 0.3-10 keV → 1 keV flux density [mJy]
nu_1keV = 2.418e17
nu_lo, nu_hi = 0.3 * nu_1keV, 10.0 * nu_1keV
beta_X = 1.27  # Gamma_X = 2.27 for spectrum 2
conv = nu_1keV**beta_X / (1.0 - beta_X) * (nu_hi**(1 - beta_X) - nu_lo**(1 - beta_X))
xrt_fnu_mJy = xrt_flux_band / conv / 1e-26
xrt_fnu_err_mJy = xrt_flux_err / conv / 1e-26

# ═══════════════════════════════════════════════════
# Figure 1: Radio light curves (3-model comparison)
# ═══════════════════════════════════════════════════
print("Computing radio light curves...")
t_model = np.geomspace(100, 1e8, 300)
t_days = t_model / 86400

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
titles = [
    r'No injection ($E_0 = 6.1 \times 10^{52}$ erg)',
    r'Magnetar ($L_0 = 10^{49}$ erg/s, $t_0 = 5 \times 10^4$ s)',
    r'Static post-injection ($E = 4.9 \times 10^{54}$ erg)',
]
jets = [jet_noinj, jet_mag, jet_post]
params_list = [P, P, P_post]

for ax, jet, title, params in zip(axes, jets, titles, params_list):
    for freq_GHz, color, label in zip([5.0, 15.0], ['#1f77b4', '#ff7f0e'], ['5 GHz', '15 GHz']):
        nu = freq_GHz * 1e9
        flux = jet.FluxDensity(t_model, nu, params)
        ax.loglog(t_days, flux, '-', color=color, lw=1.5, label=label)

        mask_d = detected & (np.abs(radio_data[:, 1] - freq_GHz) / freq_GHz < 0.15)
        if np.any(mask_d):
            ax.errorbar(radio_data[mask_d, 0], radio_data[mask_d, 2],
                         yerr=radio_data[mask_d, 3], fmt='o', color=color, ms=4,
                         capsize=2, lw=1)

    ax.set_xlabel('Time [days]')
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.1, 200)
    ax.set_ylim(5e-3, 2)
    ax.legend(fontsize=8)

axes[0].set_ylabel('Flux density [mJy]')
plt.suptitle('GRB 140304A Radio (Laskar+2017)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('docs/examples/img/grb140304a_radio.png', dpi=150, bbox_inches='tight')
print("Saved grb140304a_radio.png")
plt.close()

# ═══════════════════════════════════════════════════
# Figure 2: Radio SEDs at selected epochs (magnetar model)
# ═══════════════════════════════════════════════════
print("Computing radio SEDs...")
epochs = [
    (1.5,  1.45,  1.57),
    (4.5,  4.45,  4.60),
    (8.5,  8.45,  8.60),
    (18.2, 18.15, 18.35),
    (41.3, 41.25, 41.45),
]
epoch_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

fig, ax = plt.subplots(figsize=(8, 6))
nu_arr = np.geomspace(1e9, 1e11, 100)

for (t_epoch, t_lo, t_hi), color in zip(epochs, epoch_colors):
    t_s = t_epoch * 86400
    # Compute SED: one FluxDensity call per frequency point
    flux_sed = jet_mag.FluxDensity(np.full(len(nu_arr), t_s), nu_arr, P)
    ax.loglog(nu_arr / 1e9, flux_sed, '-', color=color, lw=1.5,
              label=f'{t_epoch:.1f} d')

    # Data overlay
    mask = detected & (radio_data[:, 0] >= t_lo) & (radio_data[:, 0] <= t_hi)
    if np.any(mask):
        ax.errorbar(radio_data[mask, 1], radio_data[mask, 2],
                     yerr=radio_data[mask, 3],
                     fmt='o', color=color, ms=5, capsize=2, lw=1, zorder=5)

ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Flux density [mJy]')
ax.set_title('GRB 140304A Radio SEDs — Magnetar model vs data')
ax.legend(title='Observer time', fontsize=9)
ax.set_xlim(1, 100)
ax.set_ylim(0.005, 5)
plt.tight_layout()
plt.savefig('docs/examples/img/grb140304a_sed.png', dpi=150)
print("Saved grb140304a_sed.png")
plt.close()

# ═══════════════════════════════════════════════════
# Figure 3: X-ray light curve (3 models + XRT data)
# ═══════════════════════════════════════════════════
print("Computing X-ray light curves...")
t_xray = np.geomspace(1e3, 1e7, 200)

flux_noinj = jet_noinj.FluxDensity(t_xray, nu_1keV, P)
flux_mag = jet_mag.FluxDensity(t_xray, nu_1keV, P)
flux_post = jet_post.FluxDensity(t_xray, nu_1keV, P_post)

fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(t_xray / 86400, flux_noinj, 'b--', lw=1.5, alpha=0.7,
          label=r'No injection ($E_0 = 6.1 \times 10^{52}$)')
ax.loglog(t_xray / 86400, flux_mag, 'r-', lw=2,
          label=r'Magnetar ($L_0 = 10^{49}$ erg/s)')
ax.loglog(t_xray / 86400, flux_post, 'g:', lw=1.5, alpha=0.7,
          label=r'Static ($E = 4.9 \times 10^{54}$)')
ax.errorbar(xrt_time_s / 86400, xrt_fnu_mJy, yerr=xrt_fnu_err_mJy,
            fmt='ko', ms=4, capsize=2, label='Swift/XRT (1 keV)', zorder=5)

ax.axvspan(0.15, 0.26, alpha=0.1, color='red', label='Injection epoch (paper)')
ax.set_xlabel('Time [days]')
ax.set_ylabel('Flux density at 1 keV [mJy]')
ax.set_title('GRB 140304A X-ray Afterglow')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(0.03, 20)
plt.tight_layout()
plt.savefig('docs/examples/img/grb140304a_xray.png', dpi=150)
print("Saved grb140304a_xray.png")
plt.close()

print("\nDone! Generated 3 figures.")
