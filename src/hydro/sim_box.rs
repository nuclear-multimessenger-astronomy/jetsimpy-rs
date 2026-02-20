use crate::constants::*;
use crate::hydro::config::JetConfig;
use crate::hydro::tools::Tool;

/// PDE solver for jet hydrodynamics.
pub struct SimBox {
    // tools
    tool: Tool,

    // configuration
    tmin: f64,
    tmax: f64,
    cfl: f64,
    spread: bool,

    // mesh
    ntheta: usize,
    theta: Vec<f64>,
    theta_edge: Vec<f64>,

    // conserved variables
    eb: Vec<f64>,
    ht: Vec<f64>,
    msw: Vec<f64>,
    mej: Vec<f64>,
    r: Vec<f64>,

    // primitive variables
    beta_gamma_sq: Vec<f64>,
    beta_th: Vec<f64>,
    psw: Vec<f64>,
    hb: Vec<f64>,
    s: Vec<f64>,

    // convenient variables
    beta: Vec<f64>,
    gamma: Vec<f64>,

    // eigenvalues
    eigenvalues: Vec<f64>,
    alpha_r: Vec<f64>,

    // slope: [5][ntheta]
    slope: Vec<Vec<f64>>,
    r_slope_l: Vec<f64>,
    r_slope_r: Vec<f64>,

    // numerical flux: [4][ntheta+1]
    numerical_flux: Vec<Vec<f64>>,
    dr_dt: Vec<f64>,

    // dy/dt: [5][ntheta]
    dy_dt: Vec<Vec<f64>>,

    // PDE solution: [5][ntheta][nt]
    pub ys: Vec<Vec<Vec<f64>>>,
    pub ts: Vec<f64>,
}

impl SimBox {
    pub fn new(config: &JetConfig) -> Self {
        let tool = Tool::new(config.nwind, config.nism, config.rtol, config.cal_level);
        let ntheta = config.eb.len();

        let mut theta = vec![0.0; ntheta];
        for i in 0..ntheta {
            theta[i] = (config.theta_edge[i] + config.theta_edge[i + 1]) / 2.0;
        }

        let mut sb = SimBox {
            tool,
            tmin: config.tmin,
            tmax: config.tmax,
            cfl: config.cfl,
            spread: config.spread,
            ntheta,
            theta,
            theta_edge: config.theta_edge.clone(),
            eb: config.eb.clone(),
            ht: config.ht.clone(),
            msw: config.msw.clone(),
            mej: config.mej.clone(),
            r: config.r.clone(),
            beta_gamma_sq: vec![0.0; ntheta],
            beta_th: vec![0.0; ntheta],
            psw: vec![0.0; ntheta],
            hb: vec![0.0; ntheta],
            s: vec![0.0; ntheta],
            beta: vec![0.0; ntheta],
            gamma: vec![0.0; ntheta],
            eigenvalues: vec![0.0; ntheta],
            alpha_r: vec![0.0; ntheta],
            slope: vec![vec![0.0; ntheta]; 5],
            r_slope_l: vec![0.0; ntheta],
            r_slope_r: vec![0.0; ntheta],
            numerical_flux: vec![vec![0.0; ntheta + 1]; 4],
            dr_dt: vec![0.0; ntheta],
            dy_dt: vec![vec![0.0; ntheta]; 5],
            ys: Vec::new(),
            ts: Vec::new(),
        };

        sb.solve_primitive();
        sb.solve_eigen();
        sb
    }

    pub fn get_theta(&self) -> &Vec<f64> {
        &self.theta
    }

    pub fn tool(&self) -> &Tool {
        &self.tool
    }

    pub fn solve_pde(&mut self) {
        if self.spread {
            self.solve_spread();
        } else {
            self.solve_no_spread();
        }
    }

    fn solve_primitive(&mut self) {
        for i in 0..self.ntheta {
            self.beta_gamma_sq[i] = self
                .tool
                .solve_beta_gamma_sq(self.msw[i] / self.eb[i], self.mej[i] / self.eb[i], self.r[i])
                .unwrap_or_else(|e| panic!("Hydro Primitive solver: {}", e));

            self.s[i] = self.tool.solve_s(self.r[i], self.beta_gamma_sq[i]);
            self.gamma[i] = (self.beta_gamma_sq[i] + 1.0).sqrt();
            self.beta[i] = (self.beta_gamma_sq[i] / (self.beta_gamma_sq[i] + 1.0)).sqrt();
            self.psw[i] = self.s[i] * self.beta[i] * self.beta[i] * self.msw[i] / 3.0;
            self.hb[i] = self.eb[i] + self.psw[i];
            self.beta_th[i] = self.ht[i] / self.hb[i];
        }
    }

    fn solve_eigen(&mut self) {
        for i in 0..self.ntheta {
            let g = self.gamma[i];
            let b = self.beta[i];
            let a_val = 2.0 * self.s[i] / 3.0 * self.msw[i] * (4.0 * g * g * g * g - 1.0)
                + ((1.0 - self.s[i]) * self.msw[i] + self.mej[i]) * g * g * g;
            let d_psw_d_eb = 2.0 * self.s[i] / 3.0 * self.msw[i] / a_val;
            let d_psw_d_msw = self.s[i] * b * b / 3.0
                - 2.0 * self.s[i] / 3.0 * (self.eb[i] - g * self.mej[i]) / a_val;
            let d_psw_d_mej = -2.0 * self.s[i] / 3.0 * g * self.msw[i] / a_val;

            let b_val =
                self.mej[i] / self.hb[i] * d_psw_d_mej + self.msw[i] / self.hb[i] * d_psw_d_msw;
            let bt = self.beta_th[i];
            let c_val = ((1.0 - bt * bt) * (d_psw_d_eb + b_val) + bt * bt / 4.0 * b_val * b_val)
                .sqrt();
            let alpha1 = bt.abs();
            let alpha2 = (bt * (1.0 - b_val / 2.0) + c_val).abs();
            let alpha3 = (bt * (1.0 - b_val / 2.0) - c_val).abs();

            self.eigenvalues[i] = alpha1.max(alpha2).max(alpha3) * C_SPEED / self.r[i];
            self.alpha_r[i] = bt.abs() / self.r[i];
        }
    }

    fn solve_slope(&mut self) {
        // For msw, mej, beta_gamma_sq, beta_th (indices 0..4)
        for j in 0..4usize {
            for i in 0..self.ntheta {
                let index1 = if i > 0 { i - 1 } else { 0 };
                let index2 = if i < self.ntheta - 1 { i + 1 } else { self.ntheta - 1 };

                let var = match j {
                    0 => &self.msw,
                    1 => &self.mej,
                    2 => &self.beta_gamma_sq,
                    3 => &self.beta_th,
                    _ => unreachable!(),
                };

                let diff1 = var[i] - var[index1];
                let diff2 = var[index2] - var[i];
                let slope1 = if i == index1 {
                    0.0
                } else {
                    diff1 / (self.theta[i] - self.theta[index1])
                };
                let slope2 = if i == index2 {
                    0.0
                } else {
                    diff2 / (self.theta[index2] - self.theta[i])
                };
                self.slope[j][i] = self.tool.minmod(slope1, slope2);
            }
        }

        // R slopes
        for i in 0..self.ntheta {
            let index1 = if i > 0 { i - 1 } else { 0 };
            let index2 = if i < self.ntheta - 1 { i + 1 } else { self.ntheta - 1 };
            let diff1 = self.r[i] - self.r[index1];
            let diff2 = self.r[index2] - self.r[i];
            self.r_slope_l[i] = if i == index1 {
                0.0
            } else {
                diff1 / (self.theta[i] - self.theta[index1])
            };
            self.r_slope_r[i] = if i == index2 {
                0.0
            } else {
                diff2 / (self.theta[index2] - self.theta[i])
            };
            self.slope[4][i] = self.tool.minmod(self.r_slope_l[i], self.r_slope_r[i]);
        }
    }

    fn solve_numerical_flux(&mut self) {
        for i in 1..self.ntheta {
            // Reconstruct primitive variables at face i
            let vars: [&Vec<f64>; 5] = [
                &self.msw,
                &self.mej,
                &self.beta_gamma_sq,
                &self.beta_th,
                &self.r,
            ];

            let mut var_l = [0.0f64; 5];
            let mut var_r = [0.0f64; 5];

            for j in 0..5 {
                var_l[j] =
                    vars[j][i - 1] + self.slope[j][i - 1] * (self.theta_edge[i] - self.theta[i - 1]);
                var_r[j] = vars[j][i] + self.slope[j][i] * (self.theta_edge[i] - self.theta[i]);
            }

            // Aliases
            let msw_l = var_l[0];
            let mej_l = var_l[1];
            let bg_sq_l = var_l[2];
            let bt_l = var_l[3];
            let r_l = var_l[4];

            let msw_r = var_r[0];
            let mej_r = var_r[1];
            let bg_sq_r = var_r[2];
            let bt_r = var_r[3];
            let r_r = var_r[4];

            let s_l = self.tool.solve_s(r_l, bg_sq_l);
            let s_r = self.tool.solve_s(r_r, bg_sq_r);

            // Left-biased conserved
            let eb_l = s_l
                * (1.0 + bg_sq_l * bg_sq_l / (bg_sq_l + 1.0) / (bg_sq_l + 1.0) / 3.0)
                * (bg_sq_l + 1.0)
                * msw_l
                + (1.0 - s_l) * (bg_sq_l + 1.0).sqrt() * msw_l
                + (bg_sq_l + 1.0).sqrt() * mej_l;
            let psw_l = s_l * bg_sq_l / (bg_sq_l + 1.0) * msw_l / 3.0;
            let ht_l = (eb_l + psw_l) * bt_l;

            // Right-biased conserved
            let eb_r = s_r
                * (1.0 + bg_sq_r * bg_sq_r / (bg_sq_r + 1.0) / (bg_sq_r + 1.0) / 3.0)
                * (bg_sq_r + 1.0)
                * msw_r
                + (1.0 - s_r) * (bg_sq_r + 1.0).sqrt() * msw_r
                + (bg_sq_r + 1.0).sqrt() * mej_r;
            let psw_r = s_r * bg_sq_r / (bg_sq_r + 1.0) * msw_r / 3.0;
            let ht_r = (eb_r + psw_r) * bt_r;

            // Physical flux
            let fl = [
                ht_l / r_l * C_SPEED,
                (ht_l * bt_l + psw_l) / r_l * C_SPEED,
                msw_l * bt_l / r_l * C_SPEED,
                mej_l * bt_l / r_l * C_SPEED,
            ];
            let fr = [
                ht_r / r_r * C_SPEED,
                (ht_r * bt_r + psw_r) / r_r * C_SPEED,
                msw_r * bt_r / r_r * C_SPEED,
                mej_r * bt_r / r_r * C_SPEED,
            ];

            // Viscosity
            let il = if i >= 2 { i - 2 } else { 0 };
            let ir = if i + 1 < self.ntheta { i + 1 } else { self.ntheta - 1 };
            let alpha = self.eigenvalues[il..ir]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            let sin_edge = self.theta_edge[i].sin();
            self.numerical_flux[0][i] =
                0.5 * (fl[0] + fr[0] - alpha * (eb_r - eb_l)) * sin_edge;
            self.numerical_flux[1][i] =
                0.5 * (fl[1] + fr[1] - alpha * (ht_r - ht_l)) * sin_edge;
            self.numerical_flux[2][i] =
                0.5 * (fl[2] + fr[2] - alpha * (msw_r - msw_l)) * sin_edge;
            self.numerical_flux[3][i] =
                0.5 * (fl[3] + fr[3] - alpha * (mej_r - mej_l)) * sin_edge;
        }
    }

    fn solve_delta_t(&self) -> f64 {
        let mut delta_t_min = f64::MAX;
        for i in 0..self.ntheta {
            let omega = self.beta[i] * C_SPEED / self.r[i];
            let omega_all = self.eigenvalues[i] + 0.05 * omega;
            let delta_t = self.cfl * (self.theta_edge[i + 1] - self.theta_edge[i]) / omega_all;
            delta_t_min = delta_t_min.min(delta_t);
        }
        delta_t_min
    }

    fn solve_dy_dt(&mut self) {
        for i in 0..self.ntheta {
            let beta_f = 4.0 * self.beta[i] * (self.beta_gamma_sq[i] + 1.0)
                / (4.0 * self.beta_gamma_sq[i] + 3.0);
            let beta_r = if self.beta_th[i] <= self.beta[i] {
                (self.beta[i] * self.beta[i] - self.beta_th[i] * self.beta_th[i]).sqrt()
            } else {
                0.0
            };
            let vol =
                self.theta_edge[i].cos() - self.theta_edge[i + 1].cos();

            // dR/dt
            let il = if i > 0 { i - 1 } else { 0 };
            let ir = if i < self.ntheta - 1 { i + 1 } else { self.ntheta - 1 };
            let alpha = self.alpha_r[il].max(self.alpha_r[i]).max(self.alpha_r[ir]);
            self.dy_dt[4][i] = (beta_f - self.slope[4][i] * self.beta_th[i] / self.r[i]
                + 0.5 * alpha * (self.r_slope_r[i] - self.r_slope_l[i]))
                * C_SPEED;

            let rho = self.tool.solve_density(self.r[i]) * MASS_P;
            self.dy_dt[0][i] = (self.numerical_flux[0][i] - self.numerical_flux[0][i + 1]) / vol
                + self.dy_dt[4][i] * rho * self.r[i] * self.r[i];
            self.dy_dt[1][i] = (self.numerical_flux[1][i] - self.numerical_flux[1][i + 1]) / vol
                + (self.theta[i].cos() / self.theta[i].sin() * self.psw[i]
                    - self.ht[i] * beta_r)
                    * C_SPEED
                    / self.r[i];
            self.dy_dt[2][i] = (self.numerical_flux[2][i] - self.numerical_flux[2][i + 1]) / vol
                + self.dy_dt[4][i] * rho * self.r[i] * self.r[i];
            self.dy_dt[3][i] = (self.numerical_flux[3][i] - self.numerical_flux[3][i + 1]) / vol;
        }
    }

    fn solve_dy_dt_no_spread(&mut self) {
        for i in 0..self.ntheta {
            let beta_f = 4.0 * self.beta[i] * (self.beta_gamma_sq[i] + 1.0)
                / (4.0 * self.beta_gamma_sq[i] + 3.0);

            self.dy_dt[4][i] = beta_f * C_SPEED;

            let rho = self.tool.solve_density(self.r[i]) * MASS_P;
            self.dy_dt[0][i] = self.dy_dt[4][i] * rho * self.r[i] * self.r[i];
            self.dy_dt[1][i] = 0.0;
            self.dy_dt[2][i] = self.dy_dt[4][i] * rho * self.r[i] * self.r[i];
            self.dy_dt[3][i] = 0.0;
        }
    }

    fn one_step_rk2(&mut self, dt: f64) {
        // Copy initial state
        let conserved_ini: [Vec<f64>; 5] = [
            self.eb.clone(),
            self.ht.clone(),
            self.msw.clone(),
            self.mej.clone(),
            self.r.clone(),
        ];

        // Step 1
        self.solve_slope();
        self.solve_numerical_flux();
        self.solve_dy_dt();

        for j in 0..self.ntheta {
            self.eb[j] += dt * self.dy_dt[0][j];
            self.ht[j] += dt * self.dy_dt[1][j];
            self.msw[j] += dt * self.dy_dt[2][j];
            self.mej[j] += dt * self.dy_dt[3][j];
            self.r[j] += dt * self.dy_dt[4][j];
        }
        self.solve_primitive();
        self.solve_eigen();

        // Step 2
        self.solve_slope();
        self.solve_numerical_flux();
        self.solve_dy_dt();

        for j in 0..self.ntheta {
            self.eb[j] = 0.5 * conserved_ini[0][j] + 0.5 * self.eb[j] + 0.5 * dt * self.dy_dt[0][j];
            self.ht[j] = 0.5 * conserved_ini[1][j] + 0.5 * self.ht[j] + 0.5 * dt * self.dy_dt[1][j];
            self.msw[j] = 0.5 * conserved_ini[2][j] + 0.5 * self.msw[j] + 0.5 * dt * self.dy_dt[2][j];
            self.mej[j] = 0.5 * conserved_ini[3][j] + 0.5 * self.mej[j] + 0.5 * dt * self.dy_dt[3][j];
            self.r[j] = 0.5 * conserved_ini[4][j] + 0.5 * self.r[j] + 0.5 * dt * self.dy_dt[4][j];
        }
        self.solve_primitive();
        self.solve_eigen();
    }

    fn one_step_rk45(&mut self, dt: &mut f64, rtol: f64) -> bool {
        let n = self.ntheta;
        let conserved_ini: [Vec<f64>; 5] = [
            self.eb.clone(),
            self.ht.clone(),
            self.msw.clone(),
            self.mej.clone(),
            self.r.clone(),
        ];

        let mut k1 = vec![vec![0.0; n]; 5];
        let mut k2 = vec![vec![0.0; n]; 5];
        let mut k3 = vec![vec![0.0; n]; 5];
        let mut k4 = vec![vec![0.0; n]; 5];
        let mut k5 = vec![vec![0.0; n]; 5];
        let mut k6 = vec![vec![0.0; n]; 5];

        let dt_val = *dt;

        // Helper macro to set conserved from expression
        macro_rules! set_conserved {
            ($j:expr, $expr:expr) => {
                let vals: [f64; 5] = $expr;
                self.eb[$j] = vals[0];
                self.ht[$j] = vals[1];
                self.msw[$j] = vals[2];
                self.mej[$j] = vals[3];
                self.r[$j] = vals[4];
            };
        }

        // Step 1
        self.solve_dy_dt_no_spread();
        for i in 0..5 {
            for j in 0..n {
                k1[i][j] = dt_val * self.dy_dt[i][j];
            }
        }
        for j in 0..n {
            set_conserved!(j, [
                conserved_ini[0][j] + k1[0][j] * 2.0 / 9.0,
                conserved_ini[1][j] + k1[1][j] * 2.0 / 9.0,
                conserved_ini[2][j] + k1[2][j] * 2.0 / 9.0,
                conserved_ini[3][j] + k1[3][j] * 2.0 / 9.0,
                conserved_ini[4][j] + k1[4][j] * 2.0 / 9.0
            ]);
        }
        self.solve_primitive();

        // Step 2
        self.solve_dy_dt_no_spread();
        for i in 0..5 {
            for j in 0..n {
                k2[i][j] = dt_val * self.dy_dt[i][j];
            }
        }
        for j in 0..n {
            set_conserved!(j, [
                conserved_ini[0][j] + k1[0][j] / 12.0 + k2[0][j] / 4.0,
                conserved_ini[1][j] + k1[1][j] / 12.0 + k2[1][j] / 4.0,
                conserved_ini[2][j] + k1[2][j] / 12.0 + k2[2][j] / 4.0,
                conserved_ini[3][j] + k1[3][j] / 12.0 + k2[3][j] / 4.0,
                conserved_ini[4][j] + k1[4][j] / 12.0 + k2[4][j] / 4.0
            ]);
        }
        self.solve_primitive();

        // Step 3
        self.solve_dy_dt_no_spread();
        for i in 0..5 {
            for j in 0..n {
                k3[i][j] = dt_val * self.dy_dt[i][j];
            }
        }
        for j in 0..n {
            set_conserved!(j, [
                conserved_ini[0][j] + k1[0][j] * 69.0 / 128.0 - k2[0][j] * 243.0 / 128.0 + k3[0][j] * 135.0 / 64.0,
                conserved_ini[1][j] + k1[1][j] * 69.0 / 128.0 - k2[1][j] * 243.0 / 128.0 + k3[1][j] * 135.0 / 64.0,
                conserved_ini[2][j] + k1[2][j] * 69.0 / 128.0 - k2[2][j] * 243.0 / 128.0 + k3[2][j] * 135.0 / 64.0,
                conserved_ini[3][j] + k1[3][j] * 69.0 / 128.0 - k2[3][j] * 243.0 / 128.0 + k3[3][j] * 135.0 / 64.0,
                conserved_ini[4][j] + k1[4][j] * 69.0 / 128.0 - k2[4][j] * 243.0 / 128.0 + k3[4][j] * 135.0 / 64.0
            ]);
        }
        self.solve_primitive();

        // Step 4
        self.solve_dy_dt_no_spread();
        for i in 0..5 {
            for j in 0..n {
                k4[i][j] = dt_val * self.dy_dt[i][j];
            }
        }
        for j in 0..n {
            set_conserved!(j, [
                conserved_ini[0][j] - k1[0][j] * 17.0 / 12.0 + k2[0][j] * 27.0 / 4.0 - k3[0][j] * 27.0 / 5.0 + k4[0][j] * 16.0 / 15.0,
                conserved_ini[1][j] - k1[1][j] * 17.0 / 12.0 + k2[1][j] * 27.0 / 4.0 - k3[1][j] * 27.0 / 5.0 + k4[1][j] * 16.0 / 15.0,
                conserved_ini[2][j] - k1[2][j] * 17.0 / 12.0 + k2[2][j] * 27.0 / 4.0 - k3[2][j] * 27.0 / 5.0 + k4[2][j] * 16.0 / 15.0,
                conserved_ini[3][j] - k1[3][j] * 17.0 / 12.0 + k2[3][j] * 27.0 / 4.0 - k3[3][j] * 27.0 / 5.0 + k4[3][j] * 16.0 / 15.0,
                conserved_ini[4][j] - k1[4][j] * 17.0 / 12.0 + k2[4][j] * 27.0 / 4.0 - k3[4][j] * 27.0 / 5.0 + k4[4][j] * 16.0 / 15.0
            ]);
        }
        self.solve_primitive();

        // Step 5
        self.solve_dy_dt_no_spread();
        for i in 0..5 {
            for j in 0..n {
                k5[i][j] = dt_val * self.dy_dt[i][j];
            }
        }
        for j in 0..n {
            set_conserved!(j, [
                conserved_ini[0][j] + k1[0][j] * 65.0 / 432.0 - k2[0][j] * 5.0 / 16.0 + k3[0][j] * 13.0 / 16.0 + k4[0][j] * 4.0 / 27.0 + k5[0][j] * 5.0 / 144.0,
                conserved_ini[1][j] + k1[1][j] * 65.0 / 432.0 - k2[1][j] * 5.0 / 16.0 + k3[1][j] * 13.0 / 16.0 + k4[1][j] * 4.0 / 27.0 + k5[1][j] * 5.0 / 144.0,
                conserved_ini[2][j] + k1[2][j] * 65.0 / 432.0 - k2[2][j] * 5.0 / 16.0 + k3[2][j] * 13.0 / 16.0 + k4[2][j] * 4.0 / 27.0 + k5[2][j] * 5.0 / 144.0,
                conserved_ini[3][j] + k1[3][j] * 65.0 / 432.0 - k2[3][j] * 5.0 / 16.0 + k3[3][j] * 13.0 / 16.0 + k4[3][j] * 4.0 / 27.0 + k5[3][j] * 5.0 / 144.0,
                conserved_ini[4][j] + k1[4][j] * 65.0 / 432.0 - k2[4][j] * 5.0 / 16.0 + k3[4][j] * 13.0 / 16.0 + k4[4][j] * 4.0 / 27.0 + k5[4][j] * 5.0 / 144.0
            ]);
        }
        self.solve_primitive();

        // Step 6
        self.solve_dy_dt_no_spread();
        for i in 0..5 {
            for j in 0..n {
                k6[i][j] = dt_val * self.dy_dt[i][j];
            }
        }
        for j in 0..n {
            set_conserved!(j, [
                conserved_ini[0][j] + k1[0][j] * 47.0 / 450.0 + k3[0][j] * 12.0 / 25.0 + k4[0][j] * 32.0 / 225.0 + k5[0][j] / 30.0 + k6[0][j] * 6.0 / 25.0,
                conserved_ini[1][j] + k1[1][j] * 47.0 / 450.0 + k3[1][j] * 12.0 / 25.0 + k4[1][j] * 32.0 / 225.0 + k5[1][j] / 30.0 + k6[1][j] * 6.0 / 25.0,
                conserved_ini[2][j] + k1[2][j] * 47.0 / 450.0 + k3[2][j] * 12.0 / 25.0 + k4[2][j] * 32.0 / 225.0 + k5[2][j] / 30.0 + k6[2][j] * 6.0 / 25.0,
                conserved_ini[3][j] + k1[3][j] * 47.0 / 450.0 + k3[3][j] * 12.0 / 25.0 + k4[3][j] * 32.0 / 225.0 + k5[3][j] / 30.0 + k6[3][j] * 6.0 / 25.0,
                conserved_ini[4][j] + k1[4][j] * 47.0 / 450.0 + k3[4][j] * 12.0 / 25.0 + k4[4][j] * 32.0 / 225.0 + k5[4][j] / 30.0 + k6[4][j] * 6.0 / 25.0
            ]);
        }

        // Error estimate
        let mut rerror = 0.0f64;
        for i in 0..5 {
            let conserved = match i {
                0 => &self.eb,
                1 => &self.ht,
                2 => &self.msw,
                3 => &self.mej,
                4 => &self.r,
                _ => unreachable!(),
            };
            for j in 0..n {
                let error = (k1[i][j] / 150.0 - k3[i][j] * 3.0 / 100.0
                    + k4[i][j] * 16.0 / 75.0
                    + k5[i][j] / 20.0
                    - k6[i][j] * 6.0 / 25.0)
                    .abs();
                rerror = rerror.max(error / conserved[j].abs());
            }
        }

        let succeeded = rerror < rtol;
        if !succeeded {
            // Roll back
            for j in 0..n {
                self.eb[j] = conserved_ini[0][j];
                self.ht[j] = conserved_ini[1][j];
                self.msw[j] = conserved_ini[2][j];
                self.mej[j] = conserved_ini[3][j];
                self.r[j] = conserved_ini[4][j];
            }
        }
        self.solve_primitive();

        // Update dt
        let boost_factor = (0.9 * (rtol / rerror).powf(0.2)).min(1.5);
        *dt *= boost_factor;

        succeeded
    }

    fn save_primitives(&mut self) {
        let primitives: [&Vec<f64>; 5] = [
            &self.msw,
            &self.mej,
            &self.beta_gamma_sq,
            &self.beta_th,
            &self.r,
        ];
        for i in 0..5 {
            for j in 0..self.ntheta {
                self.ys[i][j].push(primitives[i][j]);
            }
        }
    }

    fn init_solution(&mut self) {
        self.ts.push(self.tmin);
        self.ys = vec![vec![Vec::new(); self.ntheta]; 5];
        self.save_primitives();
    }

    fn solve_spread(&mut self) {
        self.init_solution();

        let mut t = self.tmin;
        while t < self.tmax {
            let dt = self.solve_delta_t().min(self.tmax - t + 1e-6);
            self.one_step_rk2(dt);
            t += dt;

            self.ts.push(t);
            self.save_primitives();
        }
    }

    fn solve_no_spread(&mut self) {
        self.init_solution();

        let mut delta_t = 1.0;
        let mut t = self.tmin;
        while t < self.tmax {
            let mut dt = delta_t;
            let succeeded = self.one_step_rk45(&mut dt, 1e-6);

            if succeeded {
                t += delta_t;
                delta_t = dt;

                self.ts.push(t);
                self.save_primitives();
            } else {
                delta_t = dt;
            }
        }
    }
}
