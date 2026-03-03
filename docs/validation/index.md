# Physics Validation

This section demonstrates each physics feature in blastwave with worked examples and validation plots.

All example scripts are in [`examples/`](https://github.com/nuclear-multimessenger-astronomy/blastwave/tree/main/examples) and can be run independently to reproduce the plots shown here.

## Feature Overview

| Category | Feature | Model / Parameter |
|----------|---------|-------------------|
| **Jets** | TopHat, Gaussian, PowerLaw | `TopHat`, `Gaussian`, `PowerLaw` |
| | Arbitrary viewing angle | `theta_v` in parameter dict |
| | ISM medium | `n0`, `k=0` |
| | Wind medium | `A`, `k=2` |
| **Dynamics** | PDE lateral spreading | `spread_mode="pde"` (default) |
| | ODE lateral spreading | `spread_mode="ode"` |
| | Reverse shock | `include_reverse_shock=True` |
| | Magnetar energy injection | `magnetar_l0`, `magnetar_t0`, `magnetar_q` |
| | Radiative correction | `eps_e`, `eps_b`, `p_fwd` on `Jet()` |
| **Radiation** | Synchrotron + SSA | `model="sync_ssa_smooth"` |
| | SSC + Klein-Nishina | `model="sync_ssc"` |
| | Chang-Cooper solver | `model="numeric"` |
| | Thermal electrons | `model="sync_thermal"` |
| | Deep Newtonian | `model="sync_dnp"` |
| | EBL absorption | `ebl=True` |
| | Pair production | `model="numeric"` |

## Canonical Parameters

Unless noted otherwise, examples on these pages use:

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Isotropic energy | \(E_\mathrm{iso}\) | \(10^{52}\) erg |
| Initial Lorentz factor | \(\Gamma_0\) | 300 |
| Half-opening angle | \(\theta_c\) | 0.1 rad |
| ISM density | \(n_0\) | 1.0 \(\mathrm{cm}^{-3}\) |
| Electron energy fraction | \(\varepsilon_e\) | 0.1 |
| Magnetic energy fraction | \(\varepsilon_B\) | 0.01 |
| Electron spectral index | \(p\) | 2.2 |
| Redshift | \(z\) | 0.1 |
| Luminosity distance | \(d_L\) | 474.33 Mpc |

## Pages

- **[Jets](jets.md)** — Jet structure profiles, viewing angle, circumburst medium
- **[Dynamics](dynamics.md)** — Lateral spreading, reverse shock, energy injection
- **[Radiation](radiation.md)** — Radiation models, SSC, Chang-Cooper, EBL, deep Newtonian
- **[Thermal Electrons](thermal.md)** — MQ21 thermal synchrotron, FM25 full-volume extension
