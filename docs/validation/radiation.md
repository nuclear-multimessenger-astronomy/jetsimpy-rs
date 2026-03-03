# Radiation Models

blastwave supports several radiation models selected via the `model` parameter in `FluxDensity()`. This page covers models beyond the basic synchrotron (documented in [Radiation Models](../physics/radiation.md)).

## Synchrotron Self-Compton (SSC)

In the SSC process, synchrotron photons are upscattered by the same relativistic electrons that produced them. This becomes important when the Compton Y parameter is large (\(Y \gg 1\)), which occurs for high \(\varepsilon_e\) and/or low \(\varepsilon_B\).

The `sync_ssc` model computes both the synchrotron and inverse-Compton components, including Klein-Nishina suppression at high energies:

```python
from blastwave import Jet, TopHat, ForwardJetRes

# High Compton-Y parameters
P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.3, eps_b=0.001, p=2.2, theta_v=0.0, d=474.33, z=0.1)

jet = Jet(TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
          grid=ForwardJetRes(0.1, 129), spread=False,
          eps_e=0.3, eps_b=0.001, p_fwd=2.2)

# Synchrotron only (no IC radiation, but IC cooling is still included)
flux_sync = jet.FluxDensity(t, 2.4e17, P, model="sync_ssa_smooth")

# Synchrotron + SSC (includes KN corrections)
flux_ssc = jet.FluxDensity(t, 2.4e17, P, model="sync_ssc")
```

SSC effects manifest in two ways:

1. **IC cooling**: Even when SSC radiation is not computed explicitly, the Compton Y parameter enhances the electron cooling rate, lowering \(\gamma_c\) and steepening the synchrotron spectrum above \(\nu_c\). This is included in all models via the Thomson Y correction.

2. **IC radiation**: The `sync_ssc` model additionally computes the upscattered photon spectrum, which dominates at hard X-ray and gamma-ray energies (\(\gtrsim 1\) MeV).

At 1 keV the synchrotron component dominates, so the `sync_ssa_smooth` and `sync_ssc` models give similar results. At 1 MeV the IC component takes over, producing significantly higher flux.

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
    "svg.hashsalt": "radiation_ssc",
})

t = np.geomspace(10, 1e8, 200)
tday = t / 86400

P_ssc = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
             eps_e=0.3, eps_b=0.001, p=2.2, theta_v=0.0, d=474.33, z=0.1)

jet_ssc = Jet(TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
              tmin=10.0, tmax=1e10, grid=ForwardJetRes(0.1, 129),
              spread=False, eps_e=0.3, eps_b=0.001, p_fwd=2.2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ssc_bands = [(2.4e17, "X-ray 1 keV"), (2.4e20, "Hard X-ray 1 MeV")]

for ax, (nu, nu_label) in zip(axes, ssc_bands):
    f_sync = jet_ssc.FluxDensity(t, nu, P_ssc, model="sync_ssa_smooth")
    f_ssc  = jet_ssc.FluxDensity(t, nu, P_ssc, model="sync_ssc")

    for f, label, color, ls in [
        (f_sync, "Synchrotron only", "#1f77b4", "--"),
        (f_ssc,  "Synchrotron + SSC", "#d62728", "-"),
    ]:
        m = f > 0
        if m.any():
            ax.loglog(tday[m], f[m], ls, color=color, lw=2, label=label)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux density (mJy)")
    ax.set_title(nu_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(r"Synchrotron Self-Compton ($\varepsilon_e=0.3$, $\varepsilon_B=10^{-3}$)",
             fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

**Script**: [`examples/showcase_radiation.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_radiation.py)

---

## Chang-Cooper Solver

The `numeric` model solves the electron energy distribution using the Chang-Cooper finite-difference scheme rather than assuming the analytic piecewise power-law approximation. This captures:

- Time-dependent injection and cooling evolution
- Smooth spectral breaks (rather than sharp broken power laws)
- Adiabatic cooling effects
- Pile-up near the cooling break

```python
# Analytic model (fast, ~0.1s)
flux_analytic = jet.FluxDensity(t, 3e9, P, model="sync_ssa_smooth")

# Numeric Chang-Cooper solver (slower, ~10s, more accurate breaks)
flux_numeric = jet.FluxDensity(t, 3e9, P, model="numeric")
```

The numeric model is particularly valuable when fitting data near spectral breaks, where the analytic approximation introduces artificial features. The trade-off is computational cost: the numeric solver is roughly 100x slower than the analytic model.

---

## Thermal Electrons

The `sync_thermal` model adds thermal (Maxwell-Juttner) electron emission to the non-thermal power-law, following Margalit & Quataert (2021). This produces a characteristic spectral bump at radio/mm frequencies that is important for trans-relativistic shocks.

See the dedicated **[Thermal Electrons](thermal.md)** page for full details, including the FM25 full-volume extension, usage examples, and validation against the Ferguson & Margalit (2025) reference code.

---

## EBL Absorption

For high-redshift sources observed at very high energies (\(\gtrsim 100\) GeV), the extragalactic background light (EBL) attenuates the observed flux through \(\gamma\gamma \to e^+e^-\) pair production. blastwave implements the Franceschini & Rodighiero (2018) EBL model.

Enable EBL absorption by passing `ebl=True` to `FluxDensity()`:

```python
# Intrinsic (unabsorbed) flux at ~1 TeV
flux_intrinsic = jet.FluxDensity(t, 2.4e26, P, model="sync_ssa_smooth", ebl=False)

# Observed flux with EBL absorption
flux_observed = jet.FluxDensity(t, 2.4e26, P, model="sync_ssa_smooth", ebl=True)
```

The EBL optical depth \(\tau_\mathrm{EBL}(\nu, z)\) increases steeply with both photon energy and source redshift. At ~1 TeV:

- \(z = 0.1\): \(\tau \sim 1\) (moderate absorption)
- \(z = 0.5\): \(\tau \sim 5\) (strong absorption)
- \(z = 1.0\): \(\tau \gg 10\) (effectively opaque)

This is critical for modeling GRB afterglows detected by Cherenkov telescopes (H.E.S.S., MAGIC, CTA).

---

## Deep Newtonian Phase

At very late times (months to years), the blast wave decelerates to non-relativistic speeds (\(\beta\Gamma \ll 1\)). The standard synchrotron model assumes relativistic electrons, which breaks down in this regime. The `sync_dnp` model extends the emission calculation into the deep Newtonian phase:

```python
# Standard synchrotron (valid for relativistic blast wave)
flux_std = jet.FluxDensity(t, 1e9, P, model="sync_ssa_smooth")

# Deep Newtonian correction (valid at all velocities)
flux_dnp = jet.FluxDensity(t, 1e9, P, model="sync_dnp")
```

The deep Newtonian correction modifies the electron distribution and emission coefficients for sub-relativistic shocks. The difference becomes significant at \(t \gtrsim 100\) days for typical ISM parameters, and is most apparent at radio frequencies where the blast wave is still detectable at late times.

---

The plot below shows all four features: Chang-Cooper vs analytic, thermal electrons, EBL absorption at three redshifts, and the deep Newtonian correction.

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
    "svg.hashsalt": "radiation_advanced",
})

t = np.geomspace(10, 1e8, 200)
tday = t / 86400

P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.1, eps_b=0.01, p=2.2, theta_v=0.0, d=474.33, z=0.1)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# -- Panel 1: Chang-Cooper vs analytic --
ax = axes[0, 0]
for model, label, color, ls in [
    ("sync_ssa_smooth", "Analytic", "#1f77b4", "-"),
    ("numeric",         "Chang-Cooper", "#d62728", "--"),
]:
    for nu, nu_label in [(3e9, "3 GHz"), (2.4e17, "1 keV")]:
        flux = FluxDensity_tophat(t, nu, P, spread=True, model=model)
        m = flux > 0
        if m.any():
            ax.loglog(tday[m], flux[m], ls, color=color, lw=1.5,
                     label=f"{label} ({nu_label})")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title("Chang-Cooper vs Analytic")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# -- Panel 2: Thermal electrons --
ax = axes[0, 1]
for model, label, color, ls in [
    ("sync_ssa_smooth", "Non-thermal only", "#1f77b4", "-"),
    ("sync_thermal",    "Non-thermal + thermal", "#d62728", "--"),
]:
    flux = FluxDensity_tophat(t, 3e9, P, spread=True, model=model)
    m = flux > 0
    if m.any():
        ax.loglog(tday[m], flux[m], ls, color=color, lw=2, label=label)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title("Thermal Electrons (3 GHz)")
ax.legend()
ax.grid(True, alpha=0.3)

# -- Panel 3: EBL absorption --
ax = axes[1, 0]
nu_tev = 2.4e26
colors_z = {"z=0.1": "#1f77b4", "z=0.5": "#ff7f0e", "z=1.0": "#d62728"}

for z_val, label in [(0.1, "z=0.1"), (0.5, "z=0.5"), (1.0, "z=1.0")]:
    d_L = z_val * 299792.458 / 70.0 * (1 + z_val / 2)
    P_ebl = dict(P, z=z_val, d=d_L)

    f_no  = FluxDensity_tophat(t, nu_tev, P_ebl, spread=True,
                                model="sync_ssa_smooth", ebl=False)
    f_ebl = FluxDensity_tophat(t, nu_tev, P_ebl, spread=True,
                                model="sync_ssa_smooth", ebl=True)

    m = f_no > 0
    if m.any():
        ax.loglog(tday[m], f_no[m], ":", color=colors_z[label], lw=1, alpha=0.5)
    m = f_ebl > 0
    if m.any():
        ax.loglog(tday[m], f_ebl[m], "-", color=colors_z[label], lw=2, label=label)

ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title("EBL Absorption (~1 TeV)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.02, "dotted = intrinsic", transform=ax.transAxes,
        fontsize=8, color="gray", verticalalignment="bottom")

# -- Panel 4: Deep Newtonian --
ax = axes[1, 1]
t_late = np.geomspace(1e4, 1e11, 200)
tday_late = t_late / 86400

for model, label, color, ls in [
    ("sync_ssa_smooth", "Standard synchrotron", "#1f77b4", "-"),
    ("sync_dnp",        "Deep Newtonian", "#d62728", "--"),
]:
    flux = FluxDensity_tophat(t_late, 1e9, P, spread=True, model=model, tmax=1e11)
    m = flux > 0
    if m.any():
        ax.loglog(tday_late[m], flux[m], ls, color=color, lw=2, label=label)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title("Deep Newtonian Phase (1 GHz)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Advanced Radiation Models", fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

**Script**: [`examples/showcase_radiation.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_radiation.py)
