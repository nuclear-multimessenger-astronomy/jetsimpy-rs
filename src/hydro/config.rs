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
    pub cal_level: i32,
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
            cal_level: 1,
        }
    }
}
