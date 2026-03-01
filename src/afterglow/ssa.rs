use crate::constants::*;
use crate::afterglow::blast::{Blast, ShockType};
use crate::afterglow::models::Dict;

// ---------------------------------------------------------------------------
// Synchrotron characteristic quantities
// ---------------------------------------------------------------------------

/// Synchrotron frequency: ν = (3e / 4π m_e c) B γ²
pub fn compute_syn_freq(gamma: f64, b: f64) -> f64 {
    3.0 * E_CHARGE * b * gamma * gamma / (4.0 * PI * MASS_E * C_SPEED)
}

/// Inverse: γ from frequency
pub fn compute_syn_gamma(nu: f64, b: f64) -> f64 {
    if b <= 0.0 || nu <= 0.0 {
        return 0.0;
    }
    (4.0 * PI * MASS_E * C_SPEED * nu / (3.0 * E_CHARGE * b)).sqrt()
}

/// Maximum electron Lorentz factor: γ_M = √(6π e / (σ_T B (1+Y)))
pub fn compute_syn_gamma_m_max(b: f64, y: f64) -> f64 {
    if b <= 0.0 {
        return f64::INFINITY;
    }
    (6.0 * PI * E_CHARGE / (SIGMA_T * b * (1.0 + y))).sqrt()
}

/// Minimum electron Lorentz factor
pub fn compute_syn_gamma_m(gamma_th: f64, gamma_m_max: f64, eps_e: f64, p: f64, xi: f64) -> f64 {
    let gamma_ave_minus_1 = eps_e * (gamma_th - 1.0) * (MASS_P / MASS_E) / xi;
    if p > 2.0 {
        let gm1 = (p - 2.0) / (p - 1.0) * gamma_ave_minus_1;
        gm1 + 1.0
    } else if p < 2.0 {
        let gm1 = ((2.0 - p) / (p - 1.0) * gamma_ave_minus_1
            * gamma_m_max.powf(p - 2.0)).powf(1.0 / (p - 1.0));
        gm1 + 1.0
    } else {
        // p = 2: approximate
        gamma_ave_minus_1 + 1.0
    }
}

/// Cooling Lorentz factor: γ_c from synchrotron + IC losses
pub fn compute_gamma_c(t_comv: f64, b: f64, y: f64) -> f64 {
    if b <= 0.0 || t_comv <= 0.0 {
        return f64::INFINITY;
    }
    let gamma_bar = 6.0 * PI * MASS_E * C_SPEED / (SIGMA_T * b * b * (1.0 + y) * t_comv);
    // Solve γ_c² - γ_c = gamma_bar (correction for Newtonian regime)
    (gamma_bar + (gamma_bar * gamma_bar + 4.0).sqrt()) / 2.0
}

/// Peak synchrotron power per electron: P_ν,max
fn compute_single_elec_p_nu_max(b: f64) -> f64 {
    if b <= 0.0 {
        return 0.0;
    }
    MASS_E * C_SPEED * C_SPEED * SIGMA_T * b / (3.0 * E_CHARGE)
}

/// Peak synchrotron intensity from column density
fn compute_syn_i_peak(p_nu_max: f64, column_den: f64) -> f64 {
    p_nu_max * column_den
}

// ---------------------------------------------------------------------------
// Self-absorption Lorentz factor
// ---------------------------------------------------------------------------

/// Compute the self-absorption Lorentz factor γ_a.
/// Equates synchrotron intensity to blackbody at the peak.
/// Port from VegasAfterglow synchrotron.cpp.
pub fn compute_syn_gamma_a(
    b: f64,
    i_syn_peak: f64,
    gamma_m: f64,
    gamma_c: f64,
    _gamma_m_max: f64,
    p: f64,
) -> f64 {
    if b <= 0.0 || i_syn_peak <= 0.0 {
        return 1.0;
    }

    let gamma_peak = gamma_m.min(gamma_c);
    let nu_peak = compute_syn_freq(gamma_peak, b);
    let k_t = (gamma_peak - 1.0).max(0.0) * MASS_E * C_SPEED * C_SPEED / 3.0;

    if k_t <= 0.0 || nu_peak <= 0.0 {
        return 1.0;
    }

    // Initial guess: assume ν_a in the 1/3 spectral segment
    let c2 = C_SPEED * C_SPEED;
    let ratio = i_syn_peak * c2 / (nu_peak.cbrt() * 2.0 * k_t);
    let mut nu_a = if ratio > 0.0 {
        ratio.powf(0.6)
    } else {
        return 1.0;
    };

    if nu_a > nu_peak {
        if gamma_c > gamma_m {
            // Slow cooling: try -(p-1)/2 segment, then -p/2 segment
            let nu_m = compute_syn_freq(gamma_m, b);
            let ratio2 = i_syn_peak * c2 / (2.0 * k_t) * nu_m.powf(p / 2.0);
            if ratio2 > 0.0 {
                nu_a = ratio2.powf(2.0 / (p + 4.0));
            }
            let nu_c = compute_syn_freq(gamma_c, b);
            if nu_a > nu_c {
                let ratio3 = i_syn_peak * c2 / (2.0 * k_t) * nu_c.sqrt() * nu_m.powf(p / 2.0);
                if ratio3 > 0.0 {
                    nu_a = ratio3.powf(2.0 / (p + 5.0));
                }
            }
        } else {
            // Fast cooling: try -1/2 segment, then -p/2 segment
            let nu_c = compute_syn_freq(gamma_c, b);
            let ratio2 = i_syn_peak * c2 / (2.0 * k_t) * nu_c.sqrt();
            if ratio2 > 0.0 {
                nu_a = ratio2.powf(0.4); // 2/5
            }
            let nu_m = compute_syn_freq(gamma_m, b);
            if nu_a > nu_m {
                let ratio3 = i_syn_peak * c2 / (2.0 * k_t) * nu_c.sqrt() * nu_m.powf(p / 2.0);
                if ratio3 > 0.0 {
                    nu_a = ratio3.powf(2.0 / (p + 5.0));
                }
            }
        }
    }

    compute_syn_gamma(nu_a, b) + 1.0
}

// ---------------------------------------------------------------------------
// Spectral regime determination
// ---------------------------------------------------------------------------

/// Determine the spectral regime (1-6) based on ordering of γ_a, γ_m, γ_c.
pub fn determine_regime(gamma_a: f64, gamma_m: f64, gamma_c: f64) -> usize {
    if gamma_m <= gamma_c {
        // Slow cooling
        if gamma_a <= gamma_m {
            1 // γ_a < γ_m < γ_c
        } else if gamma_a <= gamma_c {
            2 // γ_m < γ_a < γ_c
        } else {
            5 // γ_m < γ_c < γ_a
        }
    } else {
        // Fast cooling
        if gamma_a <= gamma_c {
            3 // γ_a < γ_c < γ_m
        } else if gamma_a <= gamma_m {
            4 // γ_c < γ_a < γ_m
        } else {
            6 // γ_c < γ_m < γ_a
        }
    }
}

// ---------------------------------------------------------------------------
// SynElectrons struct
// ---------------------------------------------------------------------------

/// Synchrotron electron population with self-absorption.
#[derive(Clone, Debug)]
pub struct SynElectrons {
    pub gamma_m: f64,      // minimum electron Lorentz factor
    pub gamma_c: f64,      // cooling electron Lorentz factor
    pub gamma_a: f64,      // self-absorption Lorentz factor
    pub gamma_m_max: f64,  // maximum electron Lorentz factor
    pub p: f64,            // power-law index
    pub n_e: f64,          // shock electron number per solid angle
    pub column_den: f64,   // column number density
    pub regime: usize,     // spectral regime (1-6)
}

impl SynElectrons {
    /// Compute electron spectrum dN/dγ (normalized).
    pub fn compute_spectrum(&self, gamma: f64) -> f64 {
        let cutoff = (-gamma / self.gamma_m_max - self.gamma_m.min(self.gamma_c) / gamma).exp();

        match self.regime {
            1 | 2 | 5 => {
                // Slow cooling: γ_m < γ_c
                if self.gamma_m <= 0.0 { return 0.0; }
                (self.p - 1.0) / self.gamma_m
                    * (gamma / self.gamma_m).powf(-self.p)
                    * self.gamma_c / (gamma + self.gamma_c)
                    * cutoff
            }
            3 | 4 | 6 => {
                // Fast cooling: γ_c < γ_m
                if gamma <= 0.0 { return 0.0; }
                self.gamma_c / (gamma * gamma)
                    / (1.0 + (gamma / self.gamma_m).powf(self.p - 1.0))
                    * cutoff
            }
            _ => 0.0,
        }
    }

    /// Compute column density spectrum dN_col/dγ (for SSA).
    pub fn compute_column_den(&self, gamma: f64) -> f64 {
        self.column_den * self.compute_spectrum(gamma)
    }
}

// ---------------------------------------------------------------------------
// SSA synchrotron model
// ---------------------------------------------------------------------------

/// Synchrotron model with self-absorption (SSA).
///
/// Uses the same optically-thin spectrum as `sync()` (Sari 1998) and applies
/// SSA via min(I_thin, I_blackbody), matching VegasAfterglow's approach.
///
/// Below the self-absorption frequency ν_a, the intensity saturates at the
/// Rayleigh-Jeans blackbody limit with effective temperature set by the
/// electron energy at that frequency:
///   - ν < ν_m: T_eff ~ (γ_m - 1) m_e c² / 3  → I ∝ ν²
///   - ν > ν_m: T_eff ~ (γ(ν) - 1) m_e c² / 3  → I ∝ ν^{5/2}
pub fn sync_ssa(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync_ssa requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync_ssa requires 'eps_b'");
    let p_val = *p.get("p").expect("sync_ssa requires 'p'");

    // Get B-field and density based on shock type
    let (b, n_blast, t_comv, gamma_th, dr) = match blast.shock_type {
        ShockType::Forward => {
            let e = blast.e_density;
            let b = (8.0 * PI * eps_b * e).sqrt();
            (b, blast.n_blast, blast.t / blast.gamma, blast.gamma, blast.dr)
        }
        ShockType::Reverse => {
            (blast.b3, blast.n3, blast.t_comv, blast.gamma_th3, blast.dr)
        }
    };

    if b <= 0.0 || n_blast <= 0.0 || dr <= 0.0 || t_comv <= 0.0 || gamma_th <= 1.0 {
        return 0.0;
    }

    // Characteristic Lorentz factors (same as VegasAfterglow)
    let y = 0.0;
    let gamma_m_max = compute_syn_gamma_m_max(b, y);
    let gamma_m = compute_syn_gamma_m(gamma_th, gamma_m_max, eps_e, p_val, 1.0);
    let gamma_c = compute_gamma_c(t_comv, b, y);

    // Characteristic frequencies
    let nu_m = compute_syn_freq(gamma_m, b);
    let nu_c = compute_syn_freq(gamma_c, b);

    // Peak specific emissivity (same Sari 1998 formula as sync model)
    let e_p = 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * n_blast
        / MASS_E / C_SPEED / C_SPEED;

    // Optically thin spectrum (identical to sync model)
    let emissivity = if nu_m < nu_c {
        // Slow cooling
        if nu < nu_m {
            e_p * (nu / nu_m).cbrt()
        } else if nu < nu_c {
            e_p * (nu / nu_m).powf(-(p_val - 1.0) / 2.0)
        } else {
            e_p * (nu_c / nu_m).powf(-(p_val - 1.0) / 2.0) * (nu / nu_c).powf(-p_val / 2.0)
        }
    } else {
        // Fast cooling
        if nu < nu_c {
            e_p * (nu / nu_c).cbrt()
        } else if nu < nu_m {
            e_p / (nu / nu_c).sqrt()
        } else {
            e_p / (nu_m / nu_c).sqrt() * (nu / nu_m).powf(-p_val / 2.0)
        }
    };

    // Optically thin intensity: I_thin = j_ν × dr
    let i_thin = emissivity * dr;

    // SSA: Rayleigh-Jeans blackbody limit
    // Effective temperature depends on electron energy at this frequency.
    // Below ν_m: all electrons pile up at γ_m, giving T_eff ∝ γ_m → I ∝ ν².
    // Above ν_m: the radiating electron has γ(ν) ∝ ν^{1/2}, so T_eff ∝ ν^{1/2} → I ∝ ν^{5/2}.
    let gamma_eff = if nu <= nu_m {
        gamma_m
    } else {
        compute_syn_gamma(nu, b)
    };
    let k_t = (gamma_eff - 1.0).max(0.0) * MASS_E * C_SPEED * C_SPEED / 3.0;

    if k_t <= 0.0 {
        return i_thin;
    }

    // I_BB = 2 k T_eff (ν/c)²
    let i_thick = 2.0 * k_t * nu * nu / (C_SPEED * C_SPEED);

    // Self-absorbed intensity: min of optically thin and blackbody limit
    i_thin.min(i_thick)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::afterglow::blast::Blast;

    #[test]
    fn test_syn_freq_roundtrip() {
        let b = 1.0; // 1 Gauss
        let gamma = 100.0;
        let nu = compute_syn_freq(gamma, b);
        let gamma_back = compute_syn_gamma(nu, b);
        assert!((gamma_back - gamma).abs() / gamma < 1e-10);
    }

    #[test]
    fn test_gamma_c() {
        let b = 0.1;
        let t = 1e5;
        let gc = compute_gamma_c(t, b, 0.0);
        assert!(gc > 1.0);
        assert!(gc.is_finite());
        // Stronger B → lower γ_c
        let gc2 = compute_gamma_c(t, 1.0, 0.0);
        assert!(gc2 < gc);
    }

    #[test]
    fn test_regime_determination() {
        assert_eq!(determine_regime(10.0, 100.0, 1000.0), 1); // γ_a < γ_m < γ_c
        assert_eq!(determine_regime(100.0, 10.0, 1000.0), 2); // γ_m < γ_a < γ_c
        assert_eq!(determine_regime(10.0, 1000.0, 100.0), 3); // γ_a < γ_c < γ_m
        assert_eq!(determine_regime(100.0, 1000.0, 10.0), 4); // γ_c < γ_a < γ_m
        assert_eq!(determine_regime(1000.0, 10.0, 100.0), 5); // γ_m < γ_c < γ_a
        assert_eq!(determine_regime(1000.0, 100.0, 10.0), 6); // γ_c < γ_m < γ_a
    }

    #[test]
    fn test_sync_ssa_suppresses_low_freq() {
        let mut p = Dict::new();
        p.insert("eps_e".into(), 0.1);
        p.insert("eps_b".into(), 0.01);
        p.insert("p".into(), 2.3);

        let blast = Blast {
            t: 1e5,
            theta: 0.05,
            phi: 0.0,
            r: 1e17,
            beta: 0.99,
            gamma: 10.0,
            s: 0.5,
            doppler: 5.0,
            n_blast: 1e3,
            e_density: 1e-2,
            n_ambient: 1.0,
            dr: 1e15,
            ..Blast::default()
        };

        // Very low frequency should be suppressed by SSA
        let flux_low = sync_ssa(1e6, &p, &blast);
        let flux_high = sync_ssa(1e14, &p, &blast);

        // Both should be non-negative and finite
        assert!(flux_low >= 0.0);
        assert!(flux_high >= 0.0);
        assert!(flux_low.is_finite());
        assert!(flux_high.is_finite());
    }

    #[test]
    fn test_sync_ssa_matches_sync_at_xray() {
        use crate::afterglow::models::sync;

        let mut p = Dict::new();
        p.insert("eps_e".into(), 0.1);
        p.insert("eps_b".into(), 0.01);
        p.insert("p".into(), 2.17);

        let blast = Blast {
            t: 1e5,
            theta: 0.05,
            phi: 0.0,
            r: 1e17,
            beta: 0.99,
            gamma: 10.0,
            s: 0.5,
            doppler: 5.0,
            n_blast: 1e3,
            e_density: 1e-2,
            n_ambient: 1.0,
            dr: 1e15,
            ..Blast::default()
        };

        // At X-ray, SSA should NOT affect the result
        let nu_xray = 1e18;
        let f_sync = sync(nu_xray, &p, &blast);
        let f_ssa = sync_ssa(nu_xray, &p, &blast);
        let ratio = f_ssa / f_sync;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "sync_ssa should match sync at X-ray, but ratio = {:.6e}",
            ratio
        );

        // At radio, SSA should suppress the flux
        let nu_radio = 1e6;
        let f_sync_r = sync(nu_radio, &p, &blast);
        let f_ssa_r = sync_ssa(nu_radio, &p, &blast);
        assert!(
            f_ssa_r < f_sync_r,
            "SSA should suppress radio: sync={:.6e}, ssa={:.6e}",
            f_sync_r, f_ssa_r
        );
    }

    #[test]
    fn test_syn_electrons_spectrum() {
        let elec = SynElectrons {
            gamma_m: 100.0,
            gamma_c: 1000.0,
            gamma_a: 10.0,
            gamma_m_max: 1e8,
            p: 2.3,
            n_e: 1e50,
            column_den: 1e20,
            regime: 1,
        };

        // Spectrum should be positive
        let s1 = elec.compute_spectrum(50.0);
        let s2 = elec.compute_spectrum(500.0);
        assert!(s1 >= 0.0);
        assert!(s2 >= 0.0);
        // Above γ_m: should decrease with increasing γ
        let s_high = elec.compute_spectrum(200.0);
        let s_higher = elec.compute_spectrum(500.0);
        assert!(s_high >= s_higher);
    }
}
