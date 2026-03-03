# Jets and Medium

## Forward Shock Synchrotron

The forward shock drives a relativistic blast wave into the circumburst medium, accelerating electrons that emit synchrotron radiation. This is the dominant emission component for most GRB afterglows.

The `sync_ssa_smooth` model produces smooth broken power-law spectra with self-absorption, suitable for broadband modeling. For best accuracy, enable the radiative correction by passing `eps_e`, `eps_b`, and `p_fwd` to the `Jet` constructor — this accounts for the energy radiated away by electrons when computing the blast wave dynamics.

```python
import numpy as np
from blastwave import Jet, TopHat, ForwardJetRes

P = dict(
    Eiso=1e52, lf=300, theta_c=0.1,
    n0=1.0, A=0.0, eps_e=0.1, eps_b=0.01, p=2.2,
    theta_v=0.0, d=474.33, z=0.1,
)

t = np.geomspace(10, 1e8, 200)  # seconds

jet = Jet(
    TopHat(0.1, 1e52, lf0=300),
    0.0, 1.0,  # nwind, nism
    tmin=10.0, tmax=1e10,
    grid=ForwardJetRes(0.1, 129),
    spread=True,
    eps_e=0.1, eps_b=0.01, p_fwd=2.2,  # radiative correction
)

flux_radio   = jet.FluxDensity(t, 1e9,  P, model="sync_ssa_smooth")  # 1 GHz
flux_optical = jet.FluxDensity(t, 4.56e14, P, model="sync_ssa_smooth")
flux_xray    = jet.FluxDensity(t, 2.4e17, P, model="sync_ssa_smooth")  # 1 keV
```

Characteristic features of the afterglow light curve:

- **X-ray**: Decays as a power law from early times, since \(\nu_X > \nu_c > \nu_m\)
- **Optical**: Peaks at \(\sim 10^{-3}\) days as \(\nu_m\) passes through the band
- **Radio**: Rises over days to weeks as the self-absorption frequency \(\nu_a\) drops below the observing band
- **Jet break**: Visible as a steepening at \(\sim 10\) days when the jet edge becomes visible

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
    "svg.hashsalt": "jets_multiband",
})

t = np.geomspace(10, 1e8, 200)
tday = t / 86400

P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.1, eps_b=0.01, p=2.2, theta_v=0.0, d=474.33, z=0.1)

jet = Jet(TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
          tmin=10.0, tmax=1e10, grid=ForwardJetRes(0.1, 129),
          spread=True, eps_e=0.1, eps_b=0.01, p_fwd=2.2)

bands = [
    (1e9,    "1 GHz (radio)",   "#1f77b4"),
    (5e9,    "5 GHz (radio)",   "#2ca02c"),
    (4.56e14,"R-band (optical)","#ff7f0e"),
    (2.4e17, "1 keV (X-ray)",   "#d62728"),
]

fig, ax = plt.subplots(figsize=(8, 6))
for nu, label, color in bands:
    flux = jet.FluxDensity(t, nu, P, model="sync_ssa_smooth")
    m = flux > 0
    ax.loglog(tday[m], flux[m], "-", color=color, lw=2, label=label)

ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux density (mJy)")
ax.set_title("On-axis Tophat Jet: Multi-band Afterglow")
ax.legend(loc="upper right")
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-8, 1e2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

**Script**: [`examples/showcase_jets.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_jets.py)

---

## Jet Structure

blastwave supports three angular energy profiles, set through the first argument to `Jet()`:

| Profile | Constructor | Energy profile \(E(\theta)\) |
|---------|-------------|------------------------------|
| TopHat | `TopHat(theta_c, E_iso, lf0)` | Uniform inside \(\theta_c\), zero outside |
| Gaussian | `Gaussian(theta_c, E_iso, lf0)` | \(\propto \exp(-\theta^2 / 2\theta_c^2)\) |
| PowerLaw | `PowerLaw(theta_c, E_iso, lf0, s)` | \(\propto (1 + (\theta/\theta_c)^2)^{-s/2}\) |

The jet structure primarily matters for off-axis observers (\(\theta_v > \theta_c\)), where the wing emission from structured jets produces smoother, brighter light curves than the sharp-edged tophat. On-axis, all three profiles produce nearly identical light curves since the core dominates the emission.

```python
from blastwave import FluxDensity_tophat, FluxDensity_gaussian, FluxDensity_powerlaw

P_off = {**P, 'theta_v': np.radians(15)}

flux_th = FluxDensity_tophat(t, 2.4e17, P_off, spread=True, model="sync_ssa_smooth")
flux_ga = FluxDensity_gaussian(t, 2.4e17, P_off, spread=True, model="sync_ssa_smooth")

P_pl = {**P_off, 's': 4}
flux_pl = FluxDensity_powerlaw(t, 2.4e17, P_pl, spread=True, model="sync_ssa_smooth")
```

Off-axis, the structured jets (Gaussian, PowerLaw) show earlier and brighter peaks than the TopHat, because their gradual energy wings provide partial illumination at all viewing angles. The TopHat shows a sharp rise only when the jet decelerates enough for the beaming cone to reach the observer.

```python exec="on" html="1"
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from blastwave import FluxDensity_tophat, FluxDensity_gaussian, FluxDensity_powerlaw

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
    "svg.hashsalt": "jets_structure",
})

t = np.geomspace(10, 1e8, 200)
tday = t / 86400

P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.1, eps_b=0.01, p=2.2, theta_v=0.0, d=474.33, z=0.1)

P_off = dict(P, theta_v=np.radians(15))
P_pl = dict(P_off, s=4)
P_on = dict(P, theta_v=0.0)
P_pl_on = dict(P_on, s=4)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: on-axis
ax = axes[0]
for func, label, color, ls, extra_P in [
    (FluxDensity_tophat,   "TopHat",   "#1f77b4", "-",  P_on),
    (FluxDensity_gaussian, "Gaussian", "#ff7f0e", "--", P_on),
    (FluxDensity_powerlaw, "PowerLaw (s=4)", "#2ca02c", "-.", P_pl_on),
]:
    flux = func(t, 2.4e17, extra_P, spread=True, model="sync_ssa_smooth")
    m = flux > 0
    if m.any():
        ax.loglog(tday[m], flux[m], ls, color=color, lw=2, label=label)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux density at 1 keV (mJy)")
ax.set_title(r"On-axis ($\theta_v = 0^\circ$)")
ax.legend()
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-8, 1e1)
ax.grid(True, alpha=0.3)

# Right: off-axis
ax = axes[1]
for func, label, color, ls, extra_P in [
    (FluxDensity_tophat,   "TopHat",   "#1f77b4", "-",  P_off),
    (FluxDensity_gaussian, "Gaussian", "#ff7f0e", "--", P_off),
    (FluxDensity_powerlaw, "PowerLaw (s=4)", "#2ca02c", "-.", P_pl),
]:
    flux = func(t, 2.4e17, extra_P, spread=True, model="sync_ssa_smooth")
    m = flux > 0
    if m.any():
        ax.loglog(tday[m], flux[m], ls, color=color, lw=2, label=label)
ax.set_xlabel("Time (days)")
ax.set_title(r"Off-axis ($\theta_v = 15^\circ$)")
ax.legend()
ax.set_xlim(1e-4, 1e3)
ax.set_ylim(1e-10, 1e1)
ax.grid(True, alpha=0.3)

plt.suptitle("Jet Structure: X-ray Afterglow", fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

**Script**: [`examples/showcase_jets.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_jets.py)

---

## Circumburst Medium

The ambient density profile is controlled by `nism` (constant ISM), `nwind` (wind-like \(\propto r^{-k}\)), and the power-law index `k`:

| Medium | Parameters | Density profile |
|--------|-----------|-----------------|
| ISM | `nism=1.0, nwind=0.0, k=0` | \(n = n_0\) = const |
| Wind | `nism=0.0, nwind=A, k=2` | \(n = A \cdot (r / 10^{17}\,\mathrm{cm})^{-2}\) |
| Hybrid | `nism=n_0, nwind=A, k=2` | \(n = A \cdot (r/10^{17})^{-2} + n_0\) |

The wind parameter `A` relates to the standard mass-loss parameter \(A_*\) through the density normalization at \(r = 10^{17}\) cm. For \(A_* = 0.1\), use `A=2.99`.

```python
# ISM environment
jet_ism = Jet(TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
              grid=ForwardJetRes(0.1, 129), spread=False, k=0.0,
              eps_e=0.1, eps_b=0.01, p_fwd=2.2)

# Wind environment (A*=0.1)
jet_wind = Jet(TopHat(0.1, 1e52, lf0=300), 2.99, 0.0,
               grid=ForwardJetRes(0.1, 129), spread=False, k=2.0,
               eps_e=0.1, eps_b=0.01, p_fwd=2.2)
```

Wind environments produce earlier and brighter radio emission — the higher circumburst density at small radii boosts early-time synchrotron. X-ray light curves decay faster in wind media due to the falling density profile, which reduces the energy injection from swept-up mass.

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
    "svg.hashsalt": "jets_medium",
})

t = np.geomspace(10, 1e8, 200)
tday = t / 86400

P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.1, eps_b=0.01, p=2.2, theta_v=0.0, d=474.33, z=0.1)

media = [
    ("ISM", dict(n0=1.0, A=0.0), 0.0, 1.0, 0.0, "#1f77b4"),
    ("Wind ($A_*=0.1$)", dict(n0=0.0, A=2.99), 2.99, 0.0, 2.0, "#d62728"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (nu, nu_label) in zip(axes, [(3e9, "Radio 3 GHz"), (2.4e17, "X-ray 1 keV")]):
    for name, p_over, nwind, nism, k, color in media:
        P_m = dict(P, **p_over)
        jet_m = Jet(TopHat(0.1, 1e52, lf0=300), nwind, nism,
                    tmin=10.0, tmax=1e10, grid=ForwardJetRes(0.1, 129),
                    spread=False, k=k, eps_e=0.1, eps_b=0.01, p_fwd=2.2)
        flux = jet_m.FluxDensity(t, nu, P_m, model="sync_ssa_smooth")
        m = flux > 0
        if m.any():
            ax.loglog(tday[m], flux[m], "-", color=color, lw=2, label=name)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux density (mJy)")
    ax.set_title(nu_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Circumburst Medium: ISM vs Wind", fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

**Script**: [`examples/showcase_jets.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_jets.py)
