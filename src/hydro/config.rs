/// Spreading mode for the hydro solver.
#[derive(Clone, Copy, PartialEq)]
pub enum SpreadMode {
    /// No lateral spreading (cells evolve independently, no θ evolution).
    None,
    /// Per-cell ODE spreading (VegasAfterglow-style, adaptive RK45).
    Ode,
    /// Finite-volume PDE spreading (CFL-limited RK2).
    Pde,
}

/// Simulation configuration, matching C++ JetConfig.
#[derive(Clone)]
pub struct JetConfig {
    pub theta_edge: Vec<f64>,
    pub eb: Vec<f64>,
    pub ht: Vec<f64>,
    pub msw: Vec<f64>,
    pub mej: Vec<f64>,
    pub r: Vec<f64>,
    pub nwind: f64,
    pub nism: f64,
    pub tmin: f64,
    pub tmax: f64,
    pub rtol: f64,
    pub cfl: f64,
    pub spread: bool,
    pub spread_mode: SpreadMode,
    pub theta_c: f64,
    pub cal_level: i32,

    // Reverse shock parameters
    pub include_reverse_shock: bool,
    pub sigma: f64,           // magnetization parameter (0 = unmagnetized)
    pub eps_e_rs: f64,        // electron energy fraction (reverse shock)
    pub eps_b_rs: f64,        // magnetic field energy fraction (reverse shock)
    pub p_rs: f64,            // electron spectral index (reverse shock)

    // Energy injection parameters
    pub t0_injection: f64,    // characteristic injection time (0 = no injection)
    pub l_injection: f64,     // injection luminosity (erg/s/sr)
    pub m_dot_injection: f64, // mass injection rate (g/s/sr)
}

impl Default for JetConfig {
    fn default() -> Self {
        JetConfig {
            theta_edge: Vec::new(),
            eb: Vec::new(),
            ht: Vec::new(),
            msw: Vec::new(),
            mej: Vec::new(),
            r: Vec::new(),
            nwind: 0.0,
            nism: 0.0,
            tmin: 10.0,
            tmax: 1e10,
            rtol: 1e-6,
            cfl: 0.9,
            spread: true,
            spread_mode: SpreadMode::Pde,
            theta_c: 0.1,
            cal_level: 1,
            include_reverse_shock: false,
            sigma: 0.0,
            eps_e_rs: 0.1,
            eps_b_rs: 0.01,
            p_rs: 2.3,
            t0_injection: 0.0,
            l_injection: 0.0,
            m_dot_injection: 0.0,
        }
    }
}
