# Forward-Mapping Flux

For on-axis viewing angles (\(\theta_v \approx 0\)), blastwave supports a forward-mapping flux computation that replaces the default EATS adaptive quadrature. This is selected via `flux_method="forward"`.

## Background: EATS vs forward-mapping

The standard approach to computing observer-frame flux from a relativistic blast wave is **inverse mapping** via the Equal Arrival Time Surface (EATS). For each observation time \(t_\mathrm{obs}\) and each angular cell, one must root-find the source-frame time \(t_\mathrm{src}\) satisfying:

\[
t_\mathrm{obs} = t_\mathrm{src} - \frac{R(t_\mathrm{src}) \cos\theta}{c}
\]

This root-finding, combined with adaptive quadrature over \((\theta, \phi)\), dominates the cost of computing multi-frequency light curves.

**Forward-mapping** inverts this: instead of finding \(t_\mathrm{src}\) for a given \(t_\mathrm{obs}\), it pre-computes \(t_\mathrm{obs}\) and the radiation at every hydro grid point, then queries via binary search.

## Algorithm

### Pre-computation

After the hydro solve, for each theta cell \(j\) and source time step \(k\):

1. Compute observer-frame time:
\[
t_\mathrm{obs}[j][k] = t_\mathrm{src}[k] - \frac{R[j][k] \cos\theta_j}{c}
\]

2. Compute spectral luminosity per steradian:
\[
\frac{dL}{d\Omega}[j][k] = I(\nu_\mathrm{src}) \cdot R^2 \cdot \mathcal{D}^3
\]

where \(\mathcal{D}\) is the Doppler factor and \(I(\nu_\mathrm{src})\) is the specific intensity from the chosen radiation model.

### Query

For a desired observation time \(t_\mathrm{obs}\):

1. In each theta cell, binary search the pre-computed \(t_\mathrm{obs}\) array
2. Log-linear interpolate \(dL/d\Omega\)
3. Sum over solid angle:

\[
L = 2\pi \sum_j \frac{dL}{d\Omega}_j(t_\mathrm{obs}) \cdot \left[\cos\theta_{j-1/2} - \cos\theta_{j+1/2}\right]
\]

## Optimizations

- **Cell skipping:** Cells with negligible energy (peak \(\beta\Gamma^2 < 10^{-6} \times \max\)) are excluded
- **Time-grid subsampling:** When the hydro output exceeds 300 time steps (PDE mode produces ~1000), the time grid is subsampled to limit pre-computation cost
- **Off-axis fallback:** For \(\theta_v > 0\), `flux_method="forward"` silently falls back to EATS (forward-mapping assumes azimuthal symmetry)

## Usage

```python
# On-axis: uses forward-mapping
fd = blastwave.FluxDensity_tophat(t, nu, P, flux_method="forward")

# Or via the Jet class
fd = jet.FluxDensity(t, nu, P, flux_method="forward")

# Off-axis: silently falls back to EATS
P_offaxis = {**P, "theta_v": 0.3}
fd = blastwave.FluxDensity_tophat(t, nu, P_offaxis, flux_method="forward")
```

## Accuracy

Forward-mapping agrees with EATS to within:

| Spread mode | Max deviation |
|-------------|--------------|
| PDE | 0.002 dex |
| No-spread | 0.04 dex |
| ODE | 0.10 dex |

Verified across top-hat (0.002 dex), Gaussian (0.002 dex), and power-law (0.004 dex) jet profiles.

## Performance

Forward-mapping reduces flux computation from milliseconds to ~0.1 ms, making the remaining performance gap vs VegasAfterglow (~1.5 ms) entirely hydro solve overhead.

| Method | Total | Hydro | Flux |
|--------|-------|-------|------|
| no-spread + forward | 2.4 ms | 2.3 ms | 0.1 ms |
| no-spread + EATS | 6.6 ms | 2.3 ms | 4.8 ms |
| ODE + forward | 4.1 ms | 4.2 ms | 0.1 ms |
| ODE + EATS | 8.9 ms | 3.5 ms | 3.8 ms |

*17 cells, top-hat, on-axis, X-ray, 100 time points.*

## References

- Wang, Y., Zhang, B., & Huang, B. (2025). "VegasAfterglow." *ApJ*.
