use crate::math::root::brentq;

/// Utility class matching C++ Tool.
pub struct Tool {
    nwind: f64,
    nism: f64,
    k: f64,
    rtol: f64,
    cal_level: i32,
    factor: f64,
}

impl Tool {
    pub fn new(nwind: f64, nism: f64, rtol: f64, cal_level: i32) -> Self {
        Tool {
            nwind,
            nism,
            k: 2.0,
            rtol,
            cal_level,
            factor: 2.0,
        }
    }

    pub fn new_with_k(nwind: f64, nism: f64, k: f64, rtol: f64, cal_level: i32) -> Self {
        Tool {
            nwind,
            nism,
            k,
            rtol,
            cal_level,
            factor: 2.0,
        }
    }

    pub fn nwind_nonzero(&self) -> bool {
        self.nwind > 0.0
    }

    pub fn solve_density(&self, r: f64) -> f64 {
        let r_ratio = r / 1e17;
        if self.k == 2.0 {
            self.nwind / (r_ratio * r_ratio) + self.nism
        } else {
            self.nwind / r_ratio.powf(self.k) + self.nism
        }
    }

    /// Integral of n(r)·r² dr from 0 to R, i.e. the swept number of particles per sr.
    /// Used for initial conditions with general k.
    pub fn solve_swept_number(&self, r: f64) -> f64 {
        let ism = self.nism * r * r * r / 3.0;
        if self.nwind == 0.0 {
            return ism;
        }
        let r17 = 1e17_f64;
        let csm = self.nwind * r17.powf(self.k) * r.powf(3.0 - self.k) / (3.0 - self.k);
        csm + ism
    }

    pub fn solve_s(&self, r: f64, beta_gamma_sq: f64) -> f64 {
        let r_ratio = r / 1e17;
        let n_csm = self.nwind / if self.k == 2.0 { r_ratio * r_ratio } else { r_ratio.powf(self.k) };
        let k_eff = if n_csm + self.nism > 0.0 { self.k * n_csm / (n_csm + self.nism) } else { 0.0 };

        match self.cal_level {
            0 => 1.0,
            1 => 0.52935729 - 0.05698377 * k_eff - 0.00158176 * k_eff * k_eff - 0.00939548 * k_eff * k_eff * k_eff,
            2 => {
                let s_bm =
                    0.52935729 - 0.05698377 * k_eff - 0.00158176 * k_eff * k_eff - 0.00939548 * k_eff * k_eff * k_eff;
                let s_st = 1.635 - 0.651 * k_eff;
                (s_st + s_bm * self.factor * beta_gamma_sq)
                    / (1.0 + self.factor * beta_gamma_sq)
            }
            _ => panic!("Hydro: invalid calibration level!"),
        }
    }

    pub fn solve_beta_gamma_sq(&self, msw_eb: f64, mej_eb: f64, r: f64) -> Result<f64, String> {
        let beta_gamma_min_sq = 0.0;
        let gamma_max = 1.0 / (msw_eb + mej_eb);
        let beta_gamma_max_sq = (gamma_max * gamma_max - 1.0) * 1.01;

        let mut f = |u_sq: f64| -> f64 {
            let beta_sq = u_sq / (u_sq + 1.0);
            let gamma = (u_sq + 1.0).sqrt();
            let s = self.solve_s(r, u_sq);
            s * gamma * gamma * (1.0 + beta_sq * beta_sq / 3.0) * msw_eb
                + gamma * ((1.0 - s) * msw_eb + mej_eb)
                - 1.0
        };

        brentq(&mut f, beta_gamma_min_sq, beta_gamma_max_sq, 0.0, self.rtol, 100)
    }

    pub fn minmod(&self, x1: f64, x2: f64) -> f64 {
        if x1 * x2 > 0.0 {
            if x1.abs() < x2.abs() {
                x1
            } else {
                x2
            }
        } else {
            0.0
        }
    }

    /// Binary search for index range in a sorted array.
    pub fn find_index(&self, x_array: &[f64], x: f64) -> (usize, usize) {
        let mut index1 = 0usize;
        let mut index2 = x_array.len() - 1;

        while index2 - index1 > 1 {
            let index_mid = (index1 + index2) / 2;
            if x > x_array[index_mid] {
                index1 = index_mid;
            } else {
                index2 = index_mid;
            }
        }
        (index1, index2)
    }

    pub fn linear(&self, x: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
        if x1 == x2 {
            y1
        } else {
            (y2 - y1) / (x2 - x1) * (x - x1) + y1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> Tool {
        Tool::new(0.0, 1.0, 1e-6, 1)
    }

    #[test]
    fn test_solve_density_ism_only() {
        let t = Tool::new(0.0, 5.0, 1e-6, 1);
        assert!((t.solve_density(1e18) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_density_wind_only() {
        let t = Tool::new(1.0, 0.0, 1e-6, 1);
        // At r = 1e17: nwind / (1e17/1e17)^2 = nwind
        assert!((t.solve_density(1e17) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_density_combined() {
        let t = Tool::new(2.0, 3.0, 1e-6, 1);
        // At r = 1e17: 2/(1)^2 + 3 = 5
        assert!((t.solve_density(1e17) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_s_cal_level_0() {
        let t = Tool::new(0.0, 1.0, 1e-6, 0);
        assert!((t.solve_s(1e18, 100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_s_cal_level_1_ism() {
        // With nwind=0, k = 0, so s = 0.52935729
        let t = Tool::new(0.0, 1.0, 1e-6, 1);
        assert!((t.solve_s(1e18, 100.0) - 0.52935729).abs() < 1e-6);
    }

    #[test]
    fn test_minmod_same_sign() {
        let t = tool();
        assert!((t.minmod(3.0, 5.0) - 3.0).abs() < 1e-10);
        assert!((t.minmod(-5.0, -2.0) - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_minmod_opposite_sign() {
        let t = tool();
        assert!((t.minmod(3.0, -5.0)).abs() < 1e-10);
        assert!((t.minmod(-3.0, 5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_find_index() {
        let t = tool();
        let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(t.find_index(&arr, 2.5), (1, 2));
        assert_eq!(t.find_index(&arr, 4.5), (3, 4));
        assert_eq!(t.find_index(&arr, 1.5), (0, 1));
    }

    #[test]
    fn test_linear_interpolation() {
        let t = tool();
        // Midpoint between (0, 0) and (10, 20) at x=5 should give 10
        assert!((t.linear(5.0, 0.0, 10.0, 0.0, 20.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolation_at_endpoints() {
        let t = tool();
        assert!((t.linear(0.0, 0.0, 10.0, 3.0, 7.0) - 3.0).abs() < 1e-10);
        assert!((t.linear(10.0, 0.0, 10.0, 3.0, 7.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_degenerate() {
        let t = tool();
        // x1 == x2: should return y1
        assert!((t.linear(5.0, 5.0, 5.0, 3.0, 7.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_beta_gamma_sq() {
        // For a relativistic case, verify the root satisfies energy conservation
        let t = Tool::new(0.0, 1.0, 1e-6, 1);
        let msw_eb = 0.01;
        let mej_eb = 0.001;
        let r = 1e18;
        let u_sq = t.solve_beta_gamma_sq(msw_eb, mej_eb, r).unwrap();
        assert!(u_sq > 0.0);
        assert!(u_sq.is_finite());

        // Verify the root satisfies the equation f(u_sq) ~ 0
        let beta_sq = u_sq / (u_sq + 1.0);
        let gamma = (u_sq + 1.0).sqrt();
        let s = t.solve_s(r, u_sq);
        let residual = s * gamma * gamma * (1.0 + beta_sq * beta_sq / 3.0) * msw_eb
            + gamma * ((1.0 - s) * msw_eb + mej_eb)
            - 1.0;
        assert!(residual.abs() < 1e-5);
    }

    #[test]
    fn test_solve_density_k2_matches_original() {
        // new_with_k(k=2) should give identical results to new()
        let t_old = Tool::new(1.0, 3.0, 1e-6, 1);
        let t_new = Tool::new_with_k(1.0, 3.0, 2.0, 1e-6, 1);
        for &r in &[1e16, 1e17, 1e18, 1e19] {
            assert!((t_old.solve_density(r) - t_new.solve_density(r)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_solve_density_general_k() {
        // k=0: n(r) = nwind * (r/1e17)^0 + nism = nwind + nism (constant)
        let t = Tool::new_with_k(2.0, 3.0, 0.0, 1e-6, 1);
        assert!((t.solve_density(1e16) - 5.0).abs() < 1e-10);
        assert!((t.solve_density(1e18) - 5.0).abs() < 1e-10);

        // k=1: n(r) = nwind / (r/1e17) + nism
        let t = Tool::new_with_k(2.0, 0.0, 1.0, 1e-6, 1);
        // At r=1e17: nwind/(1) = 2
        assert!((t.solve_density(1e17) - 2.0).abs() < 1e-10);
        // At r=2e17: nwind/(2) = 1
        assert!((t.solve_density(2e17) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_swept_number() {
        // ISM only: integral = nism * R^3 / 3
        let t = Tool::new_with_k(0.0, 1.0, 2.0, 1e-6, 1);
        let r = 1e18;
        let expected = 1.0 * r * r * r / 3.0;
        assert!((t.solve_swept_number(r) - expected).abs() / expected < 1e-10);

        // Wind only (k=2): integral = nwind * (1e17)^2 * r
        let t = Tool::new_with_k(1.0, 0.0, 2.0, 1e-6, 1);
        let r = 1e18;
        let expected = 1.0 * (1e17_f64).powi(2) * r / (3.0 - 2.0);
        assert!((t.solve_swept_number(r) - expected).abs() / expected < 1e-10);
    }
}
