# Hydrodynamics

blastwave solves the relativistic blast wave deceleration problem on a 1D angular grid (\(\theta\)-cells from 0 to \(\pi\)). Three lateral spreading modes are available, each trading accuracy for speed.

## State variables

All three modes track five quantities per cell:

| Variable | Symbol | Description |
|----------|--------|-------------|
| Swept-up mass | \(M_\mathrm{sw}\) | ISM/wind mass shocked by the forward shock (g/sr) |
| Ejecta mass | \(M_\mathrm{ej}\) | Initial ejecta mass (g/sr) |
| Four-velocity squared | \(\beta\Gamma^2\) | \(= \beta^2 \Gamma^2\), encodes both bulk Lorentz factor and velocity |
| Lateral velocity | \(\beta_\theta\) | Polar spreading speed (PDE mode only) |
| Radius | \(R\) | Forward shock radius (cm) |

## PDE spreading (`spread_mode="pde"`)

The default mode, originally from [jetsimpy](https://github.com/haowang-astro/jetsimpy) (Wang et al. 2024).

**Algorithm:**

- Finite-volume Godunov scheme on a 1D \(\theta\)-grid
- Conservative variables: \([E_b, H_t, M_\mathrm{sw}, M_\mathrm{ej}, R]\)
- RK2 (Heun) time integration with CFL-limited time steps
- Minmod slope limiter for cell-edge reconstruction
- Eigenvalue-based artificial viscosity for shock capturing

**Lateral flux transport:** Energy and mass flow between adjacent \(\theta\)-cells through Godunov fluxes, coupling all cells at each time step. This makes the PDE the most physically accurate mode but also the most expensive: the CFL condition forces \(\Delta t \propto \Delta\theta / v_\theta\), giving \(O(n_\theta^2)\) scaling.

**Output:** ~1000 log-spaced time steps from \(t_\mathrm{min}\) to \(t_\mathrm{max}\).

## ODE spreading (`spread_mode="ode"`)

VegasAfterglow-style per-cell adaptive ODE spreading (Wang, Zhang & Huang 2025).

**Algorithm:**

- Each \(\theta\)-cell evolves independently via adaptive Dormand-Prince RK45
- Primitive variables directly: \([M_\mathrm{sw}, M_\mathrm{ej}, R, \theta_\mathrm{cell}, \beta\Gamma^2]\)
- No lateral flux transport between cells

**Key equations:**

Radial evolution:

\[
\frac{dR}{dt} = \beta_f \cdot c
\]

where \(\beta_f\) is the forward shock velocity derived from \(\beta\Gamma^2\).

Mass sweeping with solid angle correction:

\[
\frac{dM_\mathrm{sw}}{dt} = \rho(R) \cdot R^2 \cdot \frac{dR}{dt} \cdot \frac{1 - \cos\theta_0}{1 - \cos\theta}
\]

The factor \(f = (1 - \cos\theta) / (1 - \cos\theta_0)\) accounts for the changing solid angle as the jet spreads.

Lateral spreading:

\[
\frac{d\theta}{dt} = F(u) \cdot \frac{\beta_f \cdot c}{2\Gamma R}
\]

where \(F(u) = 1/(1 + 7 u \cdot \theta_s)\) is a suppression function that enforces causality at high Lorentz factors, \(u = \sqrt{\beta\Gamma^2}\), and \(\theta_s = \max(\theta_\mathrm{cell}, \theta_c)\).

Energy conservation: \(d(\beta\Gamma^2)/dt\) is derived analytically from the energy equation, avoiding root-finding in the RHS.

**Tophat fast path:** When all cells have identical initial conditions (top-hat jet), the solver runs a single cell and replicates the result.

**Output:** ~150 log-spaced time steps.

**Energy injection:** The ODE mode supports magnetar spin-down injection:

\[
L_\mathrm{inj}(t) = L_0 \left(1 + \frac{t - t_s}{t_0}\right)^{-q}
\]

Multiple injection episodes can be specified (each with independent \(L_0, t_0, q, t_s\)).

## No spreading (`spread_mode="none"`)

Same adaptive RK45 solver as ODE mode but with lateral spreading disabled (\(d\theta/dt = 0\)). Each cell evolves radially only. This is the fastest mode and useful for early-time light curves where spreading has not yet become important.

## Accuracy comparison

ODE and PDE modes agree within ~0.1 dex for top-hat jets and ~0.13 dex for structured (Gaussian) jets across all frequencies and times. The ODE mode becomes faster than PDE at ~33--65 cells.

| Cells | PDE | ODE | No-spread |
|-------|-----|-----|-----------|
| 17 | 0.004s | 0.005s | 0.004s |
| 33 | 0.008s | 0.009s | 0.006s |
| 129 | 0.099s | 0.028s | 0.018s |
| 257 | 0.398s | 0.054s | 0.034s |

*Benchmarks: top-hat jet, on-axis, X-ray (10^18 Hz), 100 time points, single core.*

## Reverse shock

When `include_reverse_shock=True`, a separate ODE is solved per \(\theta\)-cell using the forward shock solution as the driver. The reverse shock propagates back through the ejecta and produces its own synchrotron emission with independent microphysical parameters (\(\varepsilon_{e,\mathrm{RS}}\), \(\varepsilon_{B,\mathrm{RS}}\), \(p_\mathrm{RS}\)).

The ejecta magnetization \(\sigma\) suppresses the reverse shock: higher \(\sigma\) reduces the RS contribution.

```python
jet = blastwave.Jet(
    profiles, nwind, nism,
    include_reverse_shock=True,
    sigma=0.0,          # magnetization
    eps_e_rs=0.1,       # RS electron energy fraction
    eps_b_rs=0.01,      # RS magnetic energy fraction
    p_rs=2.3,           # RS spectral index
)

flux_total = jet.FluxDensity(t, nu, P)         # FS + RS
flux_fs = jet.FluxDensity_forward(t, nu, P)    # FS only
flux_rs = jet.FluxDensity_reverse(t, nu, P)    # RS only
```

## CSM density profile

The circumburst medium density is parameterized as:

\[
n(r) = n_\mathrm{wind} \left(\frac{r}{10^{17}\,\mathrm{cm}}\right)^{-k} + n_\mathrm{ISM}
\]

| Environment | \(k\) | `nwind` | `nism` |
|-------------|-------|---------|--------|
| Constant density (ISM) | 0 | 0 | \(n_0\) |
| Stellar wind | 2 | \(A_*\) | 0 |
| Intermediate | 0 < k < 2 | \(n_w\) | 0 |

The `k` parameter is set on the `Jet` constructor or on `FluxDensity_spherical`.

## References

- Wang, H. et al. (2024). "jetsimpy." *ApJS*, 273(1), 17.
- Wang, Y., Zhang, B., & Huang, B. (2025). "VegasAfterglow." *ApJ*.
