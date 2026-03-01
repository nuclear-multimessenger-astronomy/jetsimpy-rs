use crate::constants::*;
use crate::hydro::tools::Tool;
use crate::afterglow::blast::Blast;

/// Equal Arrival Time Surface solver.
pub struct EATS {
    ntheta: usize,
    nt: usize,
    tmin: f64,
    tmax: f64,
    theta_min: f64,
    theta_max: f64,
}

impl EATS {
    pub fn new(theta: &[f64], ts: &[f64]) -> Self {
        EATS {
            ntheta: theta.len(),
            nt: ts.len(),
            tmin: ts[0],
            tmax: *ts.last().unwrap(),
            theta_min: theta[0],
            theta_max: *theta.last().unwrap(),
        }
    }

    fn find_theta_index(&self, theta: f64, theta_data: &[f64], tool: &Tool) -> (usize, usize) {
        if theta < 0.0 || theta > PI {
            panic!("EATS: theta outside bounds.");
        } else if theta < self.theta_min {
            (0, 0)
        } else if theta > self.theta_max {
            (self.ntheta - 1, self.ntheta - 1)
        } else {
            tool.find_index(theta_data, theta)
        }
    }

    fn find_time_index(
        &self,
        mu: f64,
        tobs_z: f64,
        theta_index: usize,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
    ) -> Option<(usize, usize)> {
        let f_at = |t_index: usize| -> f64 {
            let r = y_data[4][theta_index][t_index];
            let t = t_data[t_index];
            t - r * mu / C_SPEED - tobs_z
        };

        if f_at(0) > 0.0 {
            return Some((0, 0));
        }
        if f_at(self.nt - 1) < 0.0 {
            return None; // observing time exceeds PDE max
        }

        let mut idx1 = 0;
        let mut idx2 = self.nt - 1;
        while idx2 - idx1 > 1 {
            let mid = (idx1 + idx2) / 2;
            if f_at(mid) > 0.0 {
                idx2 = mid;
            } else {
                idx1 = mid;
            }
        }
        Some((idx1, idx2))
    }

    fn solve_t(
        &self,
        mu: f64,
        tobs_z: f64,
        theta_index: usize,
        t_idx1: usize,
        t_idx2: usize,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
    ) -> f64 {
        if t_idx1 == t_idx2 {
            let r = y_data[4][theta_index][t_idx1];
            tobs_z / (1.0 - r / (C_SPEED * self.tmin) * mu)
        } else {
            let t1 = t_data[t_idx1];
            let t2 = t_data[t_idx2];
            let r1 = y_data[4][theta_index][t_idx1];
            let r2 = y_data[4][theta_index][t_idx2];
            let slope = (r2 - r1) / (t2 - t1);
            (tobs_z + (r1 - slope * t1) * mu / C_SPEED) / (1.0 - slope * mu / C_SPEED)
        }
    }

    fn solve_primitive(
        &self,
        mu: f64,
        tobs_z: f64,
        theta_index: usize,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        tool: &Tool,
    ) -> Option<[f64; 6]> {
        let (t_idx1, t_idx2) = self.find_time_index(mu, tobs_z, theta_index, y_data, t_data)?;
        let t = self.solve_t(mu, tobs_z, theta_index, t_idx1, t_idx2, y_data, t_data);

        let mut val = [0.0f64; 6];
        for i in 0..5 {
            val[i] = tool.linear(
                t,
                t_data[t_idx1],
                t_data[t_idx2],
                y_data[i][theta_index][t_idx1],
                y_data[i][theta_index][t_idx2],
            );
        }
        val[5] = t;
        Some(val)
    }

    pub fn derive_blast(
        &self,
        theta: f64,
        phi: f64,
        theta_v: f64,
        val: &[f64; 6],
        tool: &Tool,
        blast: &mut Blast,
    ) {
        let msw = val[0];
        let beta_gamma_sq = val[2];
        let beta_th = val[3];
        let r = val[4];

        blast.t = val[5];
        blast.theta = theta;
        blast.phi = phi;
        blast.r = r;

        blast.gamma = (beta_gamma_sq + 1.0).sqrt();
        blast.beta = beta_gamma_sq.sqrt() / blast.gamma;
        blast.beta_th = beta_th;
        blast.beta_r = if beta_th <= blast.beta {
            (blast.beta * blast.beta - beta_th * beta_th).sqrt()
        } else {
            0.0
        };
        blast.beta_f = 4.0 * blast.beta * (beta_gamma_sq + 1.0) / (4.0 * beta_gamma_sq + 3.0);
        blast.gamma_f = (4.0 * beta_gamma_sq + 3.0) / (8.0 * beta_gamma_sq + 9.0).sqrt();
        blast.s = tool.solve_s(r, beta_gamma_sq);

        // angles
        let nr = blast.beta_r / blast.beta;
        let nth = beta_th / blast.beta;
        let mu_beta = (nr * theta.sin() * phi.cos() + nth * theta.cos() * phi.cos())
            * theta_v.sin()
            + (nr * theta.cos() - nth * theta.sin()) * theta_v.cos();
        let _mu_r = theta.cos() * theta_v.cos() + theta.sin() * phi.cos() * theta_v.sin();

        blast.doppler = if blast.gamma > 1e3 {
            1.0 / blast.gamma
                / (1.0 - mu_beta + 0.5 / blast.gamma / blast.gamma * mu_beta)
        } else {
            1.0 / blast.gamma / (1.0 - blast.beta * mu_beta)
        };
        blast.cos_theta_beta = (mu_beta - blast.beta) / (1.0 - blast.beta * mu_beta);

        // thermodynamic properties
        blast.n_ambient = tool.solve_density(r);
        blast.n_blast = 4.0 * blast.gamma * blast.n_ambient;
        blast.e_density =
            (blast.gamma - 1.0) * blast.n_blast * MASS_P * C_SPEED * C_SPEED * blast.s;
        blast.pressure =
            4.0 / 3.0 * beta_gamma_sq * blast.n_ambient * MASS_P * C_SPEED * C_SPEED * blast.s;
        blast.dr = msw / r / r / blast.n_blast / MASS_P;
    }

    /// Type 1: per-theta-cell EATS (for steep gradients).
    /// Returns false if observing time exceeds PDE max (blast left as default).
    pub fn solve_blast_type1(
        &self,
        tobs_z: f64,
        theta: f64,
        phi: f64,
        theta_v: f64,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
        blast: &mut Blast,
    ) -> bool {
        let (theta_idx1, theta_idx2) = self.find_theta_index(theta, theta_data, tool);

        let val;
        if theta_idx1 == theta_idx2 {
            // Near poles
            let mu_l = theta_data[theta_idx1].cos() * theta_v.cos()
                + theta_data[theta_idx1].sin() * phi.cos() * theta_v.sin();
            let val_l = match self.solve_primitive(mu_l, tobs_z, theta_idx1, y_data, t_data, tool) {
                Some(v) => v,
                None => return false,
            };

            let mu_r = theta_data[theta_idx1].cos() * theta_v.cos()
                + theta_data[theta_idx1].sin() * (phi + PI).cos() * theta_v.sin();
            let val_r = match self.solve_primitive(mu_r, tobs_z, theta_idx2, y_data, t_data, tool) {
                Some(v) => v,
                None => return false,
            };

            let mirror = if theta_idx1 == 0 {
                -theta_data[theta_idx1]
            } else {
                2.0 * PI - theta_data[theta_idx1]
            };

            let mut v = [0.0f64; 6];
            for i in 0..6 {
                v[i] = tool.linear(theta, theta_data[theta_idx1], mirror, val_l[i], val_r[i]);
            }
            val = v;
        } else {
            let mu_l = theta_data[theta_idx1].cos() * theta_v.cos()
                + theta_data[theta_idx1].sin() * phi.cos() * theta_v.sin();
            let val_l = match self.solve_primitive(mu_l, tobs_z, theta_idx1, y_data, t_data, tool) {
                Some(v) => v,
                None => return false,
            };

            let mu_r = theta_data[theta_idx2].cos() * theta_v.cos()
                + theta_data[theta_idx2].sin() * phi.cos() * theta_v.sin();
            let val_r = match self.solve_primitive(mu_r, tobs_z, theta_idx2, y_data, t_data, tool) {
                Some(v) => v,
                None => return false,
            };

            let mut v = [0.0f64; 6];
            for i in 0..6 {
                v[i] = tool.linear(
                    theta,
                    theta_data[theta_idx1],
                    theta_data[theta_idx2],
                    val_l[i],
                    val_r[i],
                );
            }
            val = v;
        };

        self.derive_blast(theta, phi, theta_v, &val, tool, blast);
        true
    }

    fn solve_interpolated_eats(
        &self,
        mu: f64,
        tobs_z: f64,
        theta: f64,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> Option<(f64, usize, usize)> {
        let (theta_idx1, theta_idx2) = self.find_theta_index(theta, theta_data, tool);

        let f_at = |t_index: usize| -> f64 {
            let r1 = y_data[4][theta_idx1][t_index];
            let r2 = y_data[4][theta_idx2][t_index];
            let r = if theta_idx1 == theta_idx2 {
                r1
            } else {
                (r2 - r1) / (theta_data[theta_idx2] - theta_data[theta_idx1])
                    * (theta - theta_data[theta_idx1])
                    + r1
            };
            t_data[t_index] - r * mu / C_SPEED - tobs_z
        };

        let (t_idx1, t_idx2);
        if f_at(0) > 0.0 {
            t_idx1 = 0;
            t_idx2 = 0;
        } else if f_at(self.nt - 1) < 0.0 {
            return None; // observing time exceeds PDE max
        } else {
            let mut i1 = 0;
            let mut i2 = self.nt - 1;
            while i2 - i1 > 1 {
                let mid = (i1 + i2) / 2;
                if f_at(mid) > 0.0 {
                    i2 = mid;
                } else {
                    i1 = mid;
                }
            }
            t_idx1 = i1;
            t_idx2 = i2;
        }

        let t = if t_idx1 == t_idx2 {
            t_data[t_idx1]
        } else {
            let r1 = tool.linear(
                theta,
                theta_data[theta_idx1],
                theta_data[theta_idx2],
                y_data[4][theta_idx1][t_idx1],
                y_data[4][theta_idx2][t_idx1],
            );
            let r2 = tool.linear(
                theta,
                theta_data[theta_idx1],
                theta_data[theta_idx2],
                y_data[4][theta_idx1][t_idx2],
                y_data[4][theta_idx2][t_idx2],
            );
            let slope = (r2 - r1) / (t_data[t_idx2] - t_data[t_idx1]);
            (tobs_z - mu / C_SPEED * (slope * t_data[t_idx1] - r1)) / (1.0 - mu * slope / C_SPEED)
        };

        Some((t, t_idx1, t_idx2))
    }

    /// Type 2: interpolated EATS (for smooth regions).
    /// Returns false if observing time exceeds PDE max (blast left as default).
    pub fn solve_blast_type2(
        &self,
        tobs_z: f64,
        theta: f64,
        phi: f64,
        theta_v: f64,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
        blast: &mut Blast,
    ) -> bool {
        let (theta_idx1, theta_idx2) = self.find_theta_index(theta, theta_data, tool);
        let mu = theta.cos() * theta_v.cos() + theta.sin() * phi.cos() * theta_v.sin();

        let (t, t_idx1, t_idx2) = match self.solve_interpolated_eats(
            mu, tobs_z, theta, y_data, t_data, theta_data, tool,
        ) {
            Some(v) => v,
            None => return false,
        };

        let mut val = [0.0f64; 6];
        for i in 0..5 {
            let y1 = tool.linear(
                theta,
                theta_data[theta_idx1],
                theta_data[theta_idx2],
                y_data[i][theta_idx1][t_idx1],
                y_data[i][theta_idx2][t_idx1],
            );
            let y2 = tool.linear(
                theta,
                theta_data[theta_idx1],
                theta_data[theta_idx2],
                y_data[i][theta_idx1][t_idx2],
                y_data[i][theta_idx2][t_idx2],
            );
            val[i] = tool.linear(t, t_data[t_idx1], t_data[t_idx2], y1, y2);
        }
        val[5] = t;

        self.derive_blast(theta, phi, theta_v, &val, tool, blast);
        true
    }

    /// Hybrid auto-selector.
    /// Returns false if observing time exceeds PDE max (blast left as default).
    pub fn solve_blast(
        &self,
        tobs_z: f64,
        theta: f64,
        phi: f64,
        theta_v: f64,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
        blast: &mut Blast,
    ) -> bool {
        let (theta_idx1, theta_idx2) = self.find_theta_index(theta, theta_data, tool);

        if theta_idx1 == theta_idx2 {
            self.solve_blast_type2(tobs_z, theta, phi, theta_v, y_data, t_data, theta_data, tool, blast)
        } else {
            let bg1 = y_data[2][theta_idx1][0];
            let bg2 = y_data[2][theta_idx2][0];
            if bg1.min(bg2) * 10.0 < bg1.max(bg2) {
                self.solve_blast_type1(tobs_z, theta, phi, theta_v, y_data, t_data, theta_data, tool, blast)
            } else {
                self.solve_blast_type2(tobs_z, theta, phi, theta_v, y_data, t_data, theta_data, tool, blast)
            }
        }
    }

    /// Solve blast for reverse shock.
    /// Uses forward shock EATS (same arrival time surface) but populates
    /// RS-specific fields from rs_data.
    pub fn solve_blast_reverse(
        &self,
        tobs_z: f64,
        theta: f64,
        phi: f64,
        theta_v: f64,
        y_data: &[Vec<Vec<f64>>],  // forward shock data
        rs_data: &[Vec<Vec<f64>>], // reverse shock data [NVAR_RS][ntheta][nt]
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
        blast: &mut Blast,
    ) -> bool {
        // First solve the forward EATS to get time and geometry
        if !self.solve_blast(tobs_z, theta, phi, theta_v, y_data, t_data, theta_data, tool, blast) {
            return false;
        }

        // Now interpolate RS variables at the same (t, theta) point
        let t = blast.t;
        let (theta_idx1, theta_idx2) = self.find_theta_index(theta, theta_data, tool);

        // Find time indices
        let t_idx = tool.find_index(t_data, t.min(t_data[t_data.len() - 1]).max(t_data[0]));

        // Interpolate RS state: [0]=Gamma, [1]=r_rs, [2]=m3, [3]=x3, [4]=u3_th,
        //                       [5]=t_comv, [6]=gamma_th3, [7]=b3, [8]=n3,
        //                       [9]=gamma34, [10]=n4
        let interp_rs = |var: usize| -> f64 {
            if var >= rs_data.len() { return 0.0; }
            let y11 = rs_data[var][theta_idx1][t_idx.0];
            let y12 = if theta_idx2 < rs_data[var].len() { rs_data[var][theta_idx2][t_idx.0] } else { y11 };
            let y21 = rs_data[var][theta_idx1][t_idx.1];
            let y22 = if theta_idx2 < rs_data[var].len() { rs_data[var][theta_idx2][t_idx.1] } else { y21 };

            let y1 = if theta_idx1 == theta_idx2 { y11 } else {
                tool.linear(theta, theta_data[theta_idx1], theta_data[theta_idx2], y11, y12)
            };
            let y2 = if theta_idx1 == theta_idx2 { y21 } else {
                tool.linear(theta, theta_data[theta_idx1], theta_data[theta_idx2], y21, y22)
            };
            if t_idx.0 == t_idx.1 { y1 } else {
                tool.linear(t, t_data[t_idx.0], t_data[t_idx.1], y1, y2)
            }
        };

        blast.shock_type = crate::afterglow::blast::ShockType::Reverse;
        blast.gamma_th3 = interp_rs(6);
        blast.b3 = interp_rs(7);
        blast.n3 = interp_rs(8);
        blast.t_comv = interp_rs(5);
        blast.gamma34 = interp_rs(9);
        blast.n4_upstream = interp_rs(10);

        // RS-specific thermodynamic quantities
        let m3 = interp_rs(2);
        let x3 = interp_rs(3);
        let u3_th = interp_rs(4);

        // Number density and energy density in region 3
        blast.n_blast = blast.n3;
        if blast.gamma_th3 > 1.0 {
            blast.e_density = (blast.gamma_th3 - 1.0) * blast.n3 * MASS_P * C_SPEED * C_SPEED;
        } else {
            blast.e_density = 0.0;
        }

        // Shell width = x3 (comoving) / Γ (to lab frame approx)
        if blast.n3 > 0.0 && blast.r > 0.0 {
            blast.dr = m3 / (blast.r * blast.r * blast.n3 * MASS_P);
        } else {
            blast.dr = 0.0;
        }

        true
    }

    /// Simply solve EATS time. Returns NaN if observing time exceeds PDE max.
    pub fn solve_eats(
        &self,
        tobs_z: f64,
        theta: f64,
        phi: f64,
        theta_v: f64,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> f64 {
        let mu = theta.cos() * theta_v.cos() + theta.sin() * phi.cos() * theta_v.sin();
        match self.solve_interpolated_eats(mu, tobs_z, theta, y_data, t_data, theta_data, tool) {
            Some((t, _, _)) => t,
            None => f64::NAN,
        }
    }
}
