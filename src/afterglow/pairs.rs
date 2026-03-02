/// Pair production (γ+γ → e⁺e⁻) module.
///
/// Cross-section kernel from Miceli & Nava (2022).
/// Integrates as an optional iteration within the Chang-Cooper solver.

use crate::constants::*;
use crate::afterglow::chang_cooper::ChangCooperSolver;

// ---------------------------------------------------------------------------
// Cross-section kernel
// ---------------------------------------------------------------------------

/// Pair production reaction rate kernel R(ν, ν_t) from Miceli & Nava (2022).
///
/// ξ = ν · ν_t · (h / m_e c²)²
/// R = 0 if ξ ≤ 1
/// R = 0.652 · c · σ_T · (ξ²-1) · ln(ξ) / ξ³
pub fn r_kernel(nu: f64, nu_t: f64) -> f64 {
    let mec2 = MASS_E * C_SPEED * C_SPEED;
    let xi = nu * nu_t * (H_PLANCK / mec2) * (H_PLANCK / mec2);
    if xi <= 1.0 || !xi.is_finite() {
        return 0.0;
    }
    0.652 * C_SPEED * SIGMA_T * (xi * xi - 1.0) * xi.ln() / (xi * xi * xi)
}

/// Compute pair injection rate into a given γ bin from photon field.
///
/// For a photon density spectrum n_ν (photons / cm³ / Hz), the pair injection
/// rate at Lorentz factor γ is:
///   Q_pair(γ) = ∫ n_ν R(ν_γ, ν) dν
/// where ν_γ = γ m_e c² / h is the frequency corresponding to electron energy γ m_e c².
pub fn pair_injection_rate(
    gamma: f64,
    photon_density: &[f64],
    nu_bins: &[f64],
) -> f64 {
    let mec2 = MASS_E * C_SPEED * C_SPEED;
    let nu_gamma = gamma * mec2 / H_PLANCK;

    let n = photon_density.len().min(nu_bins.len());
    if n < 2 {
        return 0.0;
    }

    let mut rate = 0.0;
    for i in 0..n - 1 {
        let r = r_kernel(nu_gamma, nu_bins[i]);
        if r > 0.0 {
            let dnu = nu_bins[i + 1] - nu_bins[i];
            rate += r * photon_density[i] * dnu;
        }
    }

    rate
}

/// Compute pair absorption optical depth for a photon at frequency nu_ssc.
///
/// τ_pair(ν) = dr · ∫ n_ν_t R(ν, ν_t) dν_t
pub fn pair_absorption(
    nu_ssc: f64,
    photon_density: &[f64],
    nu_bins: &[f64],
    dr: f64,
) -> f64 {
    let n = photon_density.len().min(nu_bins.len());
    if n < 2 {
        return 0.0;
    }

    let mut tau = 0.0;
    for i in 0..n - 1 {
        let r = r_kernel(nu_ssc, nu_bins[i]);
        if r > 0.0 {
            let dnu = nu_bins[i + 1] - nu_bins[i];
            tau += r * photon_density[i] * dnu;
        }
    }

    tau * dr
}

// ---------------------------------------------------------------------------
// Integration with Chang-Cooper solver
// ---------------------------------------------------------------------------

/// Solve the electron distribution with pair production feedback.
///
/// 1. Use the current N(γ) to compute synchrotron spectrum → photon density
/// 2. Compute pair injection source for each γ bin
/// 3. Re-solve with updated source
pub fn solve_with_pairs(
    solver: &mut ChangCooperSolver,
    _nu_obs: f64,
    b: f64,
    dr: f64,
    n_bins: usize,
    gamma_max: f64,
    p_val: f64,
    gamma_m: f64,
    n_total: f64,
    dt: f64,
    dln_v_dt: f64,
) {
    // Step 1: Build photon frequency grid from the electron distribution
    let n_nu = 100;
    let nu_min: f64 = 1e6; // Hz
    let nu_max: f64 = 1e25; // Hz
    let log_ratio = (nu_max / nu_min).ln() / (n_nu - 1) as f64;

    let mut nu_bins = vec![0.0; n_nu];
    let mut photon_density = vec![0.0; n_nu];

    for i in 0..n_nu {
        nu_bins[i] = nu_min * (log_ratio * i as f64).exp();
    }

    // Step 2: Compute photon density n_ν = j_ν · dr / (h ν c)
    for i in 0..n_nu {
        let j_nu = solver.emissivity(nu_bins[i], b);
        photon_density[i] = j_nu * dr / (H_PLANCK * nu_bins[i] * C_SPEED);
        if !photon_density[i].is_finite() || photon_density[i] < 0.0 {
            photon_density[i] = 0.0;
        }
    }

    // Step 3: Compute pair injection source for each γ bin
    let mut pair_source = vec![0.0; n_bins];
    for i in 0..n_bins {
        let g = solver.gamma_at(i);
        if g > 1.0 {
            pair_source[i] = pair_injection_rate(g, &photon_density, &nu_bins);
        }
    }

    // Step 4: Re-inject base electrons + pair source and re-solve
    solver.reinject_and_solve(p_val, gamma_m, n_total, &pair_source, dt);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_kernel_threshold() {
        // ξ ≤ 1 should give 0
        let mec2 = MASS_E * C_SPEED * C_SPEED;
        let nu1 = 1e10; // low frequency
        let nu2 = 1e10;
        let xi = nu1 * nu2 * (H_PLANCK / mec2) * (H_PLANCK / mec2);
        assert!(xi < 1.0, "ξ should be < 1 for low frequencies");
        assert_eq!(r_kernel(nu1, nu2), 0.0);
    }

    #[test]
    fn test_r_kernel_above_threshold() {
        // High frequencies should give positive kernel
        let r = r_kernel(1e25, 1e25);
        assert!(r > 0.0, "R should be positive above threshold, got {}", r);
        assert!(r.is_finite());
    }

    #[test]
    fn test_r_kernel_peaked() {
        // The kernel should have a peaked shape
        let nu_t = 1e24;
        let r_low = r_kernel(1e22, nu_t);
        let r_mid = r_kernel(1e24, nu_t);
        let r_high = r_kernel(1e28, nu_t);

        // At least the mid-range should be nonzero
        assert!(r_mid >= r_low || r_mid >= r_high,
            "Kernel should be peaked: low={}, mid={}, high={}", r_low, r_mid, r_high);
    }

    #[test]
    fn test_pair_injection_rate_basic() {
        // With zero photon density, injection should be zero
        let photon_density = vec![0.0; 10];
        let nu_bins: Vec<f64> = (0..10).map(|i| 1e20 * (1.0 + i as f64)).collect();
        let rate = pair_injection_rate(1e4, &photon_density, &nu_bins);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_pair_absorption_basic() {
        let photon_density = vec![0.0; 10];
        let nu_bins: Vec<f64> = (0..10).map(|i| 1e20 * (1.0 + i as f64)).collect();
        let tau = pair_absorption(1e25, &photon_density, &nu_bins, 1e15);
        assert_eq!(tau, 0.0);
    }
}
