# Thermal Electrons

## Introduction

In a relativistic collisionless shock, most downstream electrons are thermalized into a Maxwell-Juttner distribution while only a small fraction \(\xi_e\) are accelerated into a non-thermal power-law tail. The thermal population produces synchrotron emission with a characteristic spectral bump at a frequency

\[
\nu_\Theta \propto \Theta^2 B
\]

where \(\Theta = k_B T_e / m_e c^2\) is the dimensionless electron temperature and \(B\) is the post-shock magnetic field. This thermal synchrotron component is particularly important for trans-relativistic shocks (\(\beta\Gamma \sim 0.1\text{--}10\)), where the post-shock electrons are hot enough to radiate efficiently but the non-thermal tail may be weak.

Relevant astrophysical sources include:

- **FBOTs** (AT2018cow, CSS161010)
- **Jetted TDEs** (AT2022cmc)
- **Neutron star mergers** (GW170817-like)

The implementation follows Margalit & Quataert (2021; MQ21) with extensions from Ferguson & Margalit (2025; FM25).

## Physics Implementation

### Electron Temperature

The post-shock dimensionless electron temperature is derived from shock jump conditions (MQ21 eqs. 2-3):

\[
\Theta_0 = \varepsilon_T \frac{9 \mu m_p}{32 \mu_e m_e} \beta^2
\]

\[
\Theta = \frac{5\Theta_0 - 6 + \sqrt{25\Theta_0^2 + 180\Theta_0 + 36}}{30}
\]

where \(\varepsilon_T\) is the thermal efficiency parameter controlling what fraction of the shock energy goes into heating electrons.

### Emissivity and Absorption

**Thermal component**: Uses the Mahadevan et al. (1996) fitting function \(I'(x)\) for the thermal synchrotron emissivity (MQ21 eq. 13), where \(x = \nu / \nu_\Theta\) is the dimensionless frequency.

**Power-law component**: Non-thermal emissivity and absorption follow MQ21 eqs. 14-17, with low-frequency corrections that prevent unphysical divergences below \(\gamma_m\).

**Total emission**: The combined emissivity is \(j_\nu = j_\nu^\mathrm{th} + j_\nu^\mathrm{pl}\), with SSA applied to the total optical depth \(\tau = (\alpha_\nu^\mathrm{th} + \alpha_\nu^\mathrm{pl}) \Delta r\).

**Fast-cooling corrections**: Both thermal and non-thermal populations include synchrotron cooling corrections (MQ21 eq. 18), which suppress emission from electrons above the cooling Lorentz factor \(\gamma_c\).

### Key Parameters

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| `eps_T` | \(\varepsilon_T\) | Thermal efficiency (fraction of shock energy heating electrons) | 1.0 |
| `eps_e` | \(\varepsilon_e\) | Non-thermal electron energy fraction | (required) |
| `delta` | \(\delta\) | Power-law fraction = \(\varepsilon_e / \varepsilon_T\) | auto |
| `eps_b` | \(\varepsilon_B\) | Magnetic energy fraction | (required) |
| `p` | \(p\) | Electron spectral index | (required) |

The parameter \(\delta = \varepsilon_e / \varepsilon_T\) controls the relative strength of the non-thermal power-law tail vs the thermal bump. When \(\delta \ll 1\), thermal emission dominates near the peak frequency.

## Usage

### Basic: Thermal vs Non-Thermal SED

Select the `sync_thermal` model and provide `eps_T` in the parameter dictionary:

```python
import numpy as np
from blastwave import Jet, TopHat, ForwardJetRes

P = dict(Eiso=1e52, lf=10.0, theta_c=0.3,
         n0=1.0, A=0.0, eps_e=0.01, eps_b=0.1, p=3.0,
         theta_v=0.0, d=44.0, z=0.01,
         eps_T=0.5)   # <-- thermal efficiency

jet = Jet(TopHat(0.3, 1e52, lf0=10.0), 0.0, 1.0,
          tmin=10.0, tmax=1e9, grid=ForwardJetRes(0.3, 129),
          spread=True, eps_e=0.01, eps_b=0.1, p_fwd=3.0)

t = 100 * 86400  # 100 days

# Non-thermal only (standard synchrotron + SSA)
flux_nt = jet.FluxDensity(t, 3e9, P, model="sync_ssa_smooth")

# Thermal + non-thermal
flux_th = jet.FluxDensity(t, 3e9, P, model="sync_thermal")
```

### SED Comparison

The left panel below shows the SED at 100 days for a mildly relativistic jet (\(\Gamma_0 = 10\), \(n_0 = 1\) cm\(^{-3}\)). The thermal component adds a pronounced bump around 1-10 GHz. The right panel demonstrates sensitivity to \(\varepsilon_T\): higher thermal efficiency shifts the bump to higher frequencies and increases its amplitude.

```python exec="on" html="1"
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from blastwave import Jet, TopHat, ForwardJetRes

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
    "svg.hashsalt": "thermal_sed",
})

DAY = 86400.0
nu_arr = np.geomspace(1e7, 1e18, 300)
t_sed = 100 * DAY

P_base = dict(
    Eiso=1e52, lf=10.0, theta_c=0.3,
    n0=1.0, A=0.0, eps_e=0.01, eps_b=0.1, p=3.0,
    theta_v=0.0, d=44.0, z=0.01,
)

jet = Jet(TopHat(0.3, 1e52, lf0=10.0), 0.0, 1.0,
          tmin=10.0, tmax=1e9, grid=ForwardJetRes(0.3, 129),
          spread=True, eps_e=0.01, eps_b=0.1, p_fwd=3.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: thermal vs non-thermal
ax = axes[0]
P_sed = dict(P_base, eps_T=0.5)
flux_nt = np.array([jet.FluxDensity(t_sed, float(nu), P_base, model="sync_ssa_smooth") for nu in nu_arr])
flux_th = np.array([jet.FluxDensity(t_sed, float(nu), P_sed, model="sync_thermal") for nu in nu_arr])

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
ax = axes[1]
for eps_T, color in [(0.3, "#ff7f0e"), (0.5, "#d62728"), (1.0, "#2ca02c")]:
    P_var = dict(P_base, eps_T=eps_T)
    flux = np.array([jet.FluxDensity(t_sed, float(nu), P_var, model="sync_thermal") for nu in nu_arr])
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
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

### Light Curves

Radio light curves with and without thermal electrons, and a comparison between thin-shell and full-volume emission models:

```python exec="on" html="1"
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from blastwave import FluxDensity_tophat

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
    "svg.hashsalt": "thermal_lc",
})

DAY = 86400.0
t_lc = np.geomspace(1 * DAY, 1000 * DAY, 150)
tday = t_lc / DAY

P_base = dict(
    Eiso=1e52, lf=10.0, theta_c=0.3,
    n0=1.0, A=0.0, eps_e=0.01, eps_b=0.1, p=3.0,
    theta_v=0.0, d=44.0, z=0.01,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: radio light curves with/without thermal
ax = axes[0]
P_th = dict(P_base, eps_T=0.5)
for (nu, nu_label), color in zip([(1e9, "1 GHz"), (5e9, "5 GHz")], ["#1f77b4", "#d62728"]):
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
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

At radio frequencies the thermal component enhances the early-time flux before the non-thermal power-law component dominates at late times. The full-volume (FM25) model predicts somewhat different flux levels due to accounting for emission from the entire post-shock region rather than just the shock surface.

## Full-Volume Extension (Ferguson & Margalit 2025)

### Thin-Shell vs Full-Volume

By default, the `sync_thermal` model evaluates emission at the shock surface (**thin-shell** approximation). The FM25 **full-volume** extension instead integrates over the post-shock region using Rankine-Hugoniot conditions and a self-similar coordinate \(\xi_\mathrm{shell}\):

\[
\xi_\mathrm{shell} = \left(1 - \frac{3}{4(3-k)\Gamma^2}\right)^{1/3}
\]

where \(k\) is the circumburst density power-law index (\(k=0\) for ISM, \(k=2\) for stellar wind).

This extension is important for **trans-relativistic shocks** where a significant fraction of emission comes from deep within the post-shock region, not just the shock surface.

### Enabling Full-Volume

```python
P = dict(Eiso=1e52, lf=10.0, theta_c=0.3,
         n0=1.0, A=0.0, eps_e=0.01, eps_b=0.1, p=3.0,
         theta_v=0.0, d=44.0, z=0.01,
         eps_T=0.5,
         full_volume=1.0,  # <-- enable FM25 full-volume
         k=0.0)            # <-- CSM density index (0=ISM, 2=wind)

flux_fv = jet.FluxDensity(t, 3e9, P, model="sync_thermal")
```

## Comparison with FM25 Reference Code

To validate the implementation, we compare blastwave's `sync_thermal` (full-volume mode) against the FM25 reference code (`thermalsyn_v2.py:Fnu_of_nu`) from Ferguson & Margalit (2025, arXiv:2509.16313).

**Setup**: A spherical blast wave with \(E_\mathrm{iso} = 10^{52}\) erg and \(\Gamma_0 = 10\) in ISM (\(n_0 = 1\) cm\(^{-3}\)). We extract the fluid state at the time when \(\beta\Gamma_\mathrm{fluid} \approx 1\) and pass the corresponding shock velocity to FM25.

```python exec="on" html="1"
import io
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from blastwave import Jet, Spherical, Uniform

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
    "svg.hashsalt": "thermal_fm25",
})

DAY = 86400.0
MPC_CM = 3.0856776e24
z = 0.01
d_L = 44.0
d_L_cm = d_L * MPC_CM

eps_T_cmp = 0.5
eps_B_cmp = 0.1
eps_e_cmp = 0.01
p_cmp = 3.0
n0_cmp = 1.0
Eiso_cmp = 1e52
Gamma0_cmp = 10.0

jet_cmp = Jet(
    Spherical(Eiso_cmp, lf0=Gamma0_cmp), 0.0, n0_cmp,
    tmin=10.0, tmax=1e9,
    grid=Uniform(17), tail=False, spread=False, spread_mode="none",
    k=0.0, cal_level=1, rtol=1e-6,
    eps_e=eps_e_cmp, eps_b=eps_B_cmp, p_fwd=p_cmp,
)

# Find time when bG_fluid ~ 1
t_sample = np.geomspace(100, 5e8, 500)
bG_sample = np.zeros(len(t_sample))
for i, ts in enumerate(t_sample):
    try:
        bG_sample[i] = jet_cmp.beta_gamma(float(ts), 0.0)
    except Exception:
        bG_sample[i] = 0.0

mask_valid = bG_sample > 0
log_bG = np.log10(bG_sample[mask_valid])
log_t = np.log10(t_sample[mask_valid])
interp_t_of_bG = interp1d(log_bG, log_t, bounds_error=False, fill_value="extrapolate")
t_match = 10**float(interp_t_of_bG(np.log10(1.0)))

bG_fluid = jet_cmp.beta_gamma(float(t_match), 0.0)
R_match = jet_cmp.R(float(t_match), 0.0)

# Invert fluid bG to shock bG
def _bG_fluid_from_shock(bG_sh):
    return 0.5 * (bG_sh**2 - 2.0 + (bG_sh**4 + 5.0 * bG_sh**2 + 4.0)**0.5)**0.5

bG_sh = brentq(lambda x: _bG_fluid_from_shock(x) - bG_fluid, 0.01, 100.0)

nu_compare = np.geomspace(1e6, 1e15, 300)

# FM25 reference SED
fm25_dir = "/fred/oz480/mcoughli/arXiv-2509.16313v1/synchrotron_shock_model"
has_fm25 = os.path.isdir(fm25_dir)
if has_fm25:
    sys.path.insert(0, fm25_dir)
    import thermalsyn_v2 as fm25
    Fnu_fm25 = fm25.Fnu_of_nu(
        bG_sh, n0_cmp, nu_compare, R_match,
        Dlum=d_L_cm, z=z,
        density_insteadof_massloss=True, radius_insteadof_time=True,
        epsilon_T=eps_T_cmp, epsilon_B=eps_B_cmp, epsilon_e=eps_e_cmp, p=p_cmp,
    )
    Fnu_fm25_mJy = Fnu_fm25 / 1e-26

# blastwave SED
P_cmp = dict(
    Eiso=Eiso_cmp, lf=Gamma0_cmp, n0=n0_cmp, A=0.0,
    eps_e=eps_e_cmp, eps_b=eps_B_cmp, p=p_cmp,
    theta_v=0.0, d=d_L, z=z,
    eps_T=eps_T_cmp, full_volume=1.0, k=0.0,
)
Fnu_bw = np.array([
    jet_cmp.FluxDensity(t_match, float(nu), P_cmp, model="sync_thermal")
    for nu in nu_compare
])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: FM25 SED
ax = axes[0]
if has_fm25:
    m = Fnu_fm25_mJy > 0
    ax.loglog(nu_compare[m], Fnu_fm25_mJy[m], "-", color="#d62728", lw=2, label="FM25 thermalsyn_v2")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title(rf"FM25 ($\beta\Gamma_{{sh}}={bG_sh:.2f}$, $n={n0_cmp}$ cm$^{{-3}}$)")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: blastwave vs FM25
ax = axes[1]
m_bw = Fnu_bw > 0
if m_bw.any():
    ax.loglog(nu_compare[m_bw], Fnu_bw[m_bw], "-", color="#1f77b4", lw=2, label="blastwave sync_thermal")
if has_fm25:
    m = Fnu_fm25_mJy > 0
    ax.loglog(nu_compare[m], Fnu_fm25_mJy[m], "--", color="#d62728", lw=1.5, alpha=0.7, label="FM25 (reference)")
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
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

Both codes produce the same characteristic spectral features: SSA turnover at low frequencies, thermal bump around GHz, and power-law decline at high frequencies. The normalization offset between the two is expected: blastwave performs a full hydrodynamic simulation with EATS integration over the entire shock surface, while FM25 uses an analytic single-zone model with a volume-filling factor \(f = 3/16\). The spectral *shape* agreement confirms that the underlying MQ21 thermal physics (emissivity, absorption, cooling) is correctly implemented.

## When Thermal Emission Matters

Thermal synchrotron emission is most important when:

1. **Trans-relativistic regime** (\(\beta\Gamma \sim 0.1\text{--}1\)): The post-shock temperature is high enough for significant thermal radiation, but the shock is not ultrarelativistic (where the non-thermal tail dominates).

2. **Low \(\varepsilon_e\) / high \(\varepsilon_T\)**: When only a small fraction of electrons are accelerated (\(\delta = \varepsilon_e / \varepsilon_T \ll 1\)), the thermal population carries most of the electron energy.

3. **Dense media**: Higher upstream density increases both the post-shock magnetic field and electron density, amplifying the thermal emission.

4. **Radio/mm frequencies**: The thermal bump typically appears at GHz frequencies, making it most observable in radio and millimeter bands.

---

**Script**: [`examples/showcase_thermal.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_thermal.py)
