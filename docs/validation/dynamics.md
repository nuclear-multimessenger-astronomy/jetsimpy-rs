# Dynamics

## Lateral Spreading

As the jet decelerates, the sound speed in the shocked fluid becomes comparable to the bulk velocity, causing the jet to expand sideways. blastwave offers three spreading modes:

| Mode | `spread_mode` | Description |
|------|---------------|-------------|
| PDE | `"pde"` (default when `spread=True`) | Finite-volume lateral transport; couples all \(\theta\)-cells through Godunov fluxes |
| ODE | `"ode"` | Per-cell velocity-driven expansion; cells evolve independently |
| None | `"none"` or `spread=False` | No spreading; each cell evolves as a spherical segment |

The PDE mode captures lateral structure self-consistently through momentum and energy transport between cells. The ODE mode is faster and follows the standard analytic prescription where each cell's opening angle grows at the local sound speed.

```python
from blastwave import Jet, TopHat, ForwardJetRes

P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.1, eps_b=0.01, p=2.2, theta_v=0.0, d=474.33, z=0.1)

# PDE spreading (default)
jet_pde = Jet(TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
              grid=ForwardJetRes(0.1, 129),
              spread=True, spread_mode="pde",
              eps_e=0.1, eps_b=0.01, p_fwd=2.2)

# ODE spreading
jet_ode = Jet(TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
              grid=ForwardJetRes(0.1, 129),
              spread=True, spread_mode="ode",
              eps_e=0.1, eps_b=0.01, p_fwd=2.2)

t = np.geomspace(10, 1e8, 200)
flux_pde = jet_pde.FluxDensity(t, 1e9, P, model="sync_ssa_smooth")
flux_ode = jet_ode.FluxDensity(t, 1e9, P, model="sync_ssa_smooth")
```

Both modes produce a jet break — a steepening in the light curve at \(t_\mathrm{jet} \sim\) days to weeks — but differ in the post-break slope and transition sharpness. The PDE mode generally produces a smoother, more gradual break due to the lateral structure that develops self-consistently, while the ODE mode gives a sharper transition.

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
})

t = np.geomspace(10, 1e8, 200)
tday = t / 86400

P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.1, eps_b=0.01, p=2.2, theta_v=0.0, d=474.33, z=0.1)

modes = [
    ("No spreading", "none",  "#9467bd", ":"),
    ("ODE spreading","ode",   "#2ca02c", "--"),
    ("PDE spreading","pde",   "#d62728", "-"),
]

jets_spread = {}
for label, mode, color, ls in modes:
    sp = mode != "none"
    jet = Jet(TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
              tmin=10.0, tmax=1e10, grid=ForwardJetRes(0.1, 129),
              spread=sp, spread_mode=mode,
              eps_e=0.1, eps_b=0.01, p_fwd=2.2)
    jets_spread[mode] = jet

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
band_info = [(1e9, "Radio 1 GHz"), (4.56e14, "Optical R-band"), (2.4e17, "X-ray 1 keV")]

for ax, (nu, nu_label) in zip(axes, band_info):
    for label, mode, color, ls in modes:
        flux = jets_spread[mode].FluxDensity(t, nu, P, model="sync_ssa_smooth")
        m = flux > 0
        if m.any():
            ax.loglog(tday[m], flux[m], ls, color=color, lw=2, label=label)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux density (mJy)")
    ax.set_title(nu_label)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Lateral Spreading Modes", fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
print(buf.getvalue())
plt.close()
```

**Script**: [`examples/showcase_dynamics.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_dynamics.py)

---

## Reverse Shock

When the relativistic ejecta collide with the decelerated forward shock, a reverse shock propagates back through the ejecta shell. This produces an additional emission component that peaks at early times, particularly bright at radio and optical wavelengths.

Enable the reverse shock by passing shell and microphysics parameters to `Jet()`:

```python
jet = Jet(
    TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
    grid=ForwardJetRes(0.1, 129),
    spread=False,
    eps_e=0.1, eps_b=0.01, p_fwd=2.2,
    # Reverse shock parameters
    include_reverse_shock=True,
    duration=1000.0,        # engine activity time (s) — sets shell thickness
    sigma=0.0,              # ejecta magnetization (0 = unmagnetized)
    eps_e_rs=0.1,           # RS electron energy fraction
    eps_b_rs=0.01,          # RS magnetic energy fraction
    p_rs=2.2,               # RS electron spectral index
)
```

The `duration` parameter controls the shell thickness: longer duration means a thicker shell and a longer-lived reverse shock. The magnetization `sigma` affects the shock compression ratio and post-shock magnetic field.

Once the jet is created with `include_reverse_shock=True`, the emission can be decomposed:

```python
flux_total   = jet.FluxDensity(t, 3e9, P, model="sync_ssa_smooth")         # FS + RS
flux_forward = jet.FluxDensity_forward(t, 3e9, P, model="sync_ssa_smooth")  # FS only
flux_reverse = jet.FluxDensity_reverse(t, 3e9, P, model="sync_ssa_smooth")  # RS only
```

The reverse shock component typically dominates the radio emission at \(t \lesssim 1\) day, producing a characteristic early-time bump before the forward shock takes over.

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
})

P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.1, eps_b=0.01, p=2.2, theta_v=0.0, d=474.33, z=0.1)

t_rs = np.geomspace(10, 1e7, 200)
tday_rs = t_rs / 86400

jet_rs = Jet(
    TopHat(0.1, 1e52, lf0=300), 0.0, 1.0,
    tmin=10.0, tmax=1e10, grid=ForwardJetRes(0.1, 129),
    spread=False, eps_e=0.1, eps_b=0.01, p_fwd=2.2,
    include_reverse_shock=True,
    duration=1000.0, sigma=0.0,
    eps_e_rs=0.1, eps_b_rs=0.01, p_rs=2.2,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
band_info_rs = [(3e9, "Radio 3 GHz"), (4.56e14, "Optical R-band")]

for ax, (nu, nu_label) in zip(axes, band_info_rs):
    f_total = jet_rs.FluxDensity(t_rs, nu, P, model="sync_ssa_smooth")
    f_fwd   = jet_rs.FluxDensity_forward(t_rs, nu, P, model="sync_ssa_smooth")
    f_rvs   = jet_rs.FluxDensity_reverse(t_rs, nu, P, model="sync_ssa_smooth")

    for f, label, color, ls in [
        (f_total, "Total (FS+RS)", "k",       "-"),
        (f_fwd,   "Forward shock", "#1f77b4", "--"),
        (f_rvs,   "Reverse shock", "#d62728", ":"),
    ]:
        m = f > 0
        if m.any():
            ax.loglog(tday_rs[m], f[m], ls, color=color, lw=2, label=label)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux density (mJy)")
    ax.set_title(nu_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Reverse Shock Decomposition (duration = 1000 s)", fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
print(buf.getvalue())
plt.close()
```

**Script**: [`examples/showcase_dynamics.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_dynamics.py)

---

## Energy Injection

blastwave supports energy injection from a central engine via a magnetar spin-down model. The injected luminosity follows:

\[
L(t) = \frac{L_0}{(1 + t/t_0)^q}
\]

where \(L_0\) is the initial luminosity, \(t_0\) is the spin-down timescale, and \(q\) controls the decay steepness (\(q = 2\) for dipole radiation).

Energy injection sustains the blast wave against deceleration, producing an X-ray plateau (the luminosity stays roughly constant while injection dominates) and delayed radio re-brightening.

```python
jet = Jet(
    TopHat(0.1, 1e52, lf0=100), 0.0, 1.0,
    grid=ForwardJetRes(0.1, 129),
    spread=False,
    eps_e=0.1, eps_b=0.01, p_fwd=2.2,
    magnetar_l0=1e47,   # injection luminosity (erg/s)
    magnetar_t0=1e4,    # spin-down timescale (s)
    magnetar_q=2,        # decay index
)
```

Multiple injection episodes can be specified by passing lists:

```python
jet = Jet(...,
    magnetar_l0=[1e47, 1e46],   # two injection components
    magnetar_t0=[1e4, 1e6],
    magnetar_q=[2, 1.5],
    magnetar_ts=[0, 1e5],       # start times (s)
)
```

!!! note
    Energy injection currently requires `spread=False` or `spread_mode="ode"`. The PDE spread mode does not support injection.

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
})

t = np.geomspace(10, 1e8, 200)
tday = t / 86400

P = dict(Eiso=1e52, lf=300, theta_c=0.1, n0=1.0, A=0.0,
         eps_e=0.1, eps_b=0.01, p=2.2, theta_v=0.0, d=474.33, z=0.1)

P_mag = dict(P, lf=100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
band_info_mag = [(2.4e17, "X-ray 1 keV"), (5e9, "Radio 5 GHz")]

for case, mag_kw, color, ls, label in [
    ("baseline", {}, "#1f77b4", "--", "No injection"),
    ("magnetar", dict(magnetar_l0=1e47, magnetar_t0=1e4, magnetar_q=2),
     "#d62728", "-", r"Magnetar ($L_0=10^{47}$, $t_0=10^4$ s)"),
]:
    jet_mag = Jet(TopHat(0.1, 1e52, lf0=100), 0.0, 1.0,
                  tmin=10.0, tmax=1e10, grid=ForwardJetRes(0.1, 129),
                  spread=False, eps_e=0.1, eps_b=0.01, p_fwd=2.2,
                  **mag_kw)
    for ax, (nu, nu_label) in zip(axes, band_info_mag):
        flux = jet_mag.FluxDensity(t, nu, P_mag, model="sync_ssa_smooth")
        m = flux > 0
        if m.any():
            ax.loglog(tday[m], flux[m], ls, color=color, lw=2, label=label)

for ax, (nu, nu_label) in zip(axes, band_info_mag):
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux density (mJy)")
    ax.set_title(nu_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(r"Magnetar Energy Injection ($\Gamma_0 = 100$)", fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
print(buf.getvalue())
plt.close()
```

**Script**: [`examples/showcase_dynamics.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/showcase_dynamics.py)
