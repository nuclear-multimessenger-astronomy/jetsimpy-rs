# Numeric Model Comparison: Chang-Cooper vs Analytic

This example compares the analytic synchrotron model (`model="sync"`) with the full numeric Chang-Cooper electron distribution solver (`model="numeric"`) to validate consistency and highlight where the numeric model differs.

## Background

The default `"sync"` model uses the standard analytic broken-power-law synchrotron spectrum (Sari, Piran & Narayan 1998). The `"numeric"` model instead constructs the steady-state electron energy distribution on a log-spaced grid and computes synchrotron emissivity and self-absorption by integrating the exact spectral kernel \(F(\nu/\nu_c)\) over the distribution. This naturally captures:

- **Exact spectral shapes** from the Dermer (2009) synchrotron kernel, rather than broken power-law approximations
- **Synchrotron self-absorption** computed directly from the electron distribution derivative \(d/d\gamma[N/\gamma^2]\)
- **Smooth spectral transitions** at the characteristic breaks \(\nu_m\) and \(\nu_c\)
- **Pair production** at very high energies (optional, via `include_pp=1`)

The analytic and numeric models agree to \(\sim 10\%\) at optically thin frequencies, with differences arising from the broken power-law approximation in the analytic model. At radio frequencies where SSA matters, the two models use different absorption treatments and can differ by a factor of \(\sim 2\).

## Physical parameters

We use a standard on-axis top-hat jet:

| Parameter | Value | Notes |
|-----------|-------|-------|
| \(E_\mathrm{iso}\) | \(10^{53}\) erg | Isotropic-equivalent energy |
| \(\Gamma_0\) | 300 | Initial Lorentz factor |
| \(\theta_c\) | 0.1 rad | Jet half-opening angle |
| \(n_0\) | 1.0 cm\(^{-3}\) | ISM density |
| \(\varepsilon_e\) | 0.1 | Electron energy fraction |
| \(\varepsilon_B\) | 0.01 | Magnetic energy fraction |
| \(p\) | 2.2 | Electron spectral index |
| \(d\) | 1000 Mpc | Luminosity distance |
| \(z\) | 0.2 | Redshift |

## Computing the models

```python
import numpy as np
from blastwave import FluxDensity_tophat

DAY = 86400.0

P = {
    "Eiso":    1e53,
    "lf":      300.0,
    "theta_c": 0.1,
    "A":       0.0,
    "n0":      1.0,
    "eps_e":   0.1,
    "eps_b":   0.01,
    "p":       2.2,
    "theta_v": 0.0,
    "d":       1000.0,
    "z":       0.2,
}

t = np.geomspace(0.01 * DAY, 1000.0 * DAY, 200)

# Analytic synchrotron (Sari+98 broken power law)
F_sync = {
    "100 GHz": FluxDensity_tophat(t, 1e11 * np.ones_like(t), P,
                                   tmin=1.0, tmax=1e8, model="sync"),
    "1 keV":   FluxDensity_tophat(t, 2.418e17 * np.ones_like(t), P,
                                   tmin=1.0, tmax=1e8, model="sync"),
}

# Chang-Cooper numeric
F_num = {
    "100 GHz": FluxDensity_tophat(t, 1e11 * np.ones_like(t), P,
                                   tmin=1.0, tmax=1e8, model="numeric"),
    "1 keV":   FluxDensity_tophat(t, 2.418e17 * np.ones_like(t), P,
                                   tmin=1.0, tmax=1e8, model="numeric"),
}
```

## Light curves and residuals

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
    "svg.hashsalt": "numeric_comparison",
})

DAY = 86400.0
P = {
    "Eiso": 1e53, "lf": 300.0, "theta_c": 0.1, "A": 0.0, "n0": 1.0,
    "eps_e": 0.1, "eps_b": 0.01, "p": 2.2,
    "theta_v": 0.0, "d": 1000.0, "z": 0.2,
}

t = np.geomspace(0.01 * DAY, 1000.0 * DAY, 200)
tday = t / DAY
kw = dict(tmin=1.0, tmax=1e8)

bands = [
    ("100 GHz", 1e11, "C0"),
    ("Optical R", 4.56e14, "C2"),
    ("1 keV", 2.418e17, "C3"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for name, nu, color in bands:
    nu_arr = nu * np.ones_like(t)
    f_sync = FluxDensity_tophat(t, nu_arr, P, model="sync", **kw)
    f_num  = FluxDensity_tophat(t, nu_arr, P, model="numeric", **kw)

    m = (f_sync > 0) & (f_num > 0)
    ax1.plot(tday[m], f_sync[m], "-", color=color, lw=2, label=f"{name} sync")
    ax1.plot(tday[m], f_num[m], "--", color=color, lw=2, alpha=0.8, label=f"{name} numeric")

    resid = np.full_like(f_sync, np.nan)
    resid[m] = (f_num[m] / f_sync[m] - 1.0) * 100
    ax2.plot(tday, resid, "-", color=color, lw=2, label=name)

ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("Time (days)"); ax1.set_ylabel("Flux density (mJy)")
ax1.set_title("Light Curves")
ax1.legend(fontsize=8, ncol=2)
ax1.tick_params(which="both", direction="in", top=True, right=True)

ax2.axhline(0, color="gray", ls="--", lw=1)
ax2.axhspan(-10, 10, color="gray", alpha=0.08)
ax2.set_xscale("log")
ax2.set_xlabel("Time (days)"); ax2.set_ylabel("Residual (%)")
ax2.set_title("Numeric / Sync $-$ 1")
ax2.legend()
ax2.set_ylim(-50, 80)
ax2.tick_params(which="both", direction="in", top=True, right=True)

plt.suptitle("Numeric (Chang-Cooper) vs Analytic (Sari+98)", fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

The left panel shows light curves at three representative frequencies. Solid lines are the analytic `sync` model; dashed lines are the `numeric` model. At optically thin frequencies (100 GHz near the spectral peak, optical, X-ray), the two models track each other closely across three decades in time.

The right panel shows fractional residuals. The numeric model typically agrees with the analytic Sari+98 formula to within \(\sim 10\text{--}30\%\), with the sign and magnitude of the offset depending on the spectral regime:

- **Near the spectral peak** (\(\nu \sim \nu_m\)): the numeric model can be \(\sim 20\text{--}30\%\) higher because the exact \(F(x)\) kernel is broader than the broken power-law approximation
- **Above the cooling break** (\(\nu \gg \nu_c\)): the numeric model is \(\sim 10\%\) lower, reflecting the difference between the exact spectral integral and the Sari+98 coefficient

## Broadband SED comparison

The spectral energy distribution (SED) highlights differences near the characteristic breaks \(\nu_m\), \(\nu_c\), and the SSA turnover.

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
    "svg.hashsalt": "numeric_sed",
})

DAY = 86400.0
P = {
    "Eiso": 1e53, "lf": 300.0, "theta_c": 0.1, "A": 0.0, "n0": 1.0,
    "eps_e": 0.1, "eps_b": 0.01, "p": 2.2,
    "theta_v": 0.0, "d": 1000.0, "z": 0.2,
}
kw = dict(tmin=1.0, tmax=1e8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, t_epoch, title in [
    (ax1, 1.0 * DAY, "$t = 1$ day (slow cooling)"),
    (ax2, 0.001 * DAY, "$t = 86$ s (fast cooling)"),
]:
    nu_arr = np.geomspace(1e7, 1e19, 300)
    t_arr = t_epoch * np.ones_like(nu_arr)
    f_sync = FluxDensity_tophat(t_arr, nu_arr, P, model="sync", **kw)
    f_num  = FluxDensity_tophat(t_arr, nu_arr, P, model="numeric", **kw)

    m_s = f_sync > 0
    m_n = f_num > 0
    if m_s.any():
        ax.loglog(nu_arr[m_s], (nu_arr * f_sync)[m_s], "-", color="C0", lw=2, label="sync")
    if m_n.any():
        ax.loglog(nu_arr[m_n], (nu_arr * f_num)[m_n], "--", color="C3", lw=2, alpha=0.8, label="numeric")
    ax.set_xlabel(r"Frequency (Hz)")
    ax.set_ylabel(r"$\nu F_\nu$ (mJy Hz)")
    ax.set_title(title)
    ax.legend()
    ax.tick_params(which="both", direction="in", top=True, right=True)

plt.suptitle("Broadband SED Comparison", fontweight="bold")
plt.tight_layout()
buf = io.StringIO()
plt.savefig(buf, format="svg")
svg = buf.getvalue()
print(f'<div class="md-typeset__scrollwrap">{svg[svg.index("<svg"):]}</div>')
plt.close()
```

Key features visible in the SEDs:

- At low frequencies, the numeric model shows SSA self-absorption (the \(\nu^{5/2}\) rise) while the bare `sync` model does not include SSA. The SSA turnover is computed directly from the electron distribution.
- Near the spectral breaks (\(\nu_m\), \(\nu_c\)), the numeric model produces smooth transitions rather than the sharp power-law junctions of the analytic model.
- At high frequencies, both models follow the same \(\nu^{-p/2}\) power law with \(\sim 10\%\) normalization difference.

## Discussion

The `"numeric"` model constructs the steady-state cooled electron distribution analytically on a log-spaced \(\gamma\) grid:

- **Slow cooling** (\(\gamma_c > \gamma_m\)): \(N(\gamma) \propto \gamma^{-p}\) for \(\gamma_m \le \gamma \le \gamma_c\), steepening to \(\gamma^{-(p+1)}\) above \(\gamma_c\)
- **Fast cooling** (\(\gamma_c < \gamma_m\)): \(N(\gamma) \propto \gamma^{-2}\) for \(\gamma_c \le \gamma \le \gamma_m\), steepening to \(\gamma^{-(p+1)}\) above \(\gamma_m\)

The emissivity is then computed by integrating the exact Dermer (2009) synchrotron kernel \(F(\nu/\nu_c(\gamma))\) over this distribution, and SSA is computed from the distribution derivative. This avoids the broken power-law approximations of the analytic model while remaining fast (a single grid evaluation per frequency, no PDE time-stepping).

The \(\sim 10\text{--}30\%\) differences between the models are expected and well-understood:

1. The Sari+98 broken power-law approximation uses asymptotic spectral slopes matched at sharp break frequencies. The exact integral smoothly interpolates between regimes.
2. The pitch-angle averaging factor (`PITCH_ANGLE_AVG` \(\approx 0.72\)) used in the analytic model is an approximate correction; the numeric model computes the perpendicular emissivity, leading to small normalization offsets.
3. SSA in the numeric model is computed self-consistently from the electron distribution, while the analytic SSA models use a Rayleigh-Jeans blackbody approximation.

!!! tip "When to use `model=\"numeric\"`"
    Use the numeric model when:

    - Fitting broadband SEDs across spectral breaks where smooth transitions matter
    - Working at very high energies (\(\gtrsim 10\) keV) where pair production is relevant (set `include_pp=1`)
    - Wanting self-consistent SSA from the electron distribution

    For standard light curve fitting at a single frequency, `model="sync"` or `model="sync_ssa_smooth"` is faster and equally accurate.

## Full script

The complete analysis script is at [`examples/numeric_comparison.py`](https://github.com/nuclear-multimessenger-astronomy/blastwave/blob/main/examples/numeric_comparison.py).
