/// Blast fluid element properties.
#[derive(Clone, Default)]
pub struct Blast {
    // coordinate values (burster frame)
    pub t: f64,
    pub theta: f64,
    pub phi: f64,
    pub r: f64,

    // blast velocity (burster frame)
    pub beta: f64,
    pub gamma: f64,
    pub beta_th: f64,
    pub beta_r: f64,
    pub beta_f: f64,
    pub gamma_f: f64,
    pub s: f64,

    // angles (burster frame)
    pub doppler: f64,
    pub cos_theta_beta: f64,

    // thermodynamic values (comoving frame)
    pub n_blast: f64,
    pub e_density: f64,
    pub pressure: f64,
    pub n_ambient: f64,
    pub dr: f64,
}
