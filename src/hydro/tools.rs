use crate::math::root::brentq;

/// Utility class matching C++ Tool.
pub struct Tool {
    nwind: f64,
    nism: f64,
    rtol: f64,
    cal_level: i32,
    factor: f64,
}

impl Tool {
    pub fn new(nwind: f64, nism: f64, rtol: f64, cal_level: i32) -> Self {
        Tool {
            nwind,
            nism,
            rtol,
            cal_level,
            factor: 2.0,
        }
    }

    pub fn solve_density(&self, r: f64) -> f64 {
        let r17 = r / 1e17;
        self.nwind / (r17 * r17) + self.nism
    }

    pub fn solve_s(&self, r: f64, beta_gamma_sq: f64) -> f64 {
        let r17 = r / 1e17;
        let k = 2.0 * self.nwind / (self.nwind + self.nism * r17 * r17);

        match self.cal_level {
            0 => 1.0,
            1 => 0.52935729 - 0.05698377 * k - 0.00158176 * k * k - 0.00939548 * k * k * k,
            2 => {
                let s_bm =
                    0.52935729 - 0.05698377 * k - 0.00158176 * k * k - 0.00939548 * k * k * k;
                let s_st = 1.635 - 0.651 * k;
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
