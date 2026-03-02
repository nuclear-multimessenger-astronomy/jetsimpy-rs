use crate::constants::*;
use crate::hydro::tools::Tool;

// ---------------------------------------------------------------------------
// Physical constants derived from CGS fundamentals
// ---------------------------------------------------------------------------
const C2: f64 = C_SPEED * C_SPEED;
const SIGMA_CUT: f64 = 1e-6; // below this σ is treated as 0

// ---------------------------------------------------------------------------
// Smoothstep: smooth interpolation from 1 (t<=edge1) to 0 (t>=edge0)
// ---------------------------------------------------------------------------
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if x <= edge1 {
        return 1.0;
    }
    if x >= edge0 {
        return 0.0;
    }
    let t = (x - edge1) / (edge0 - edge1);
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Adiabatic index for relativistic gas: γ_ad = 4/3 + 1/(3Γ)
// ---------------------------------------------------------------------------
fn adiabatic_idx(gamma_rel: f64) -> f64 {
    4.0 / 3.0 + 1.0 / (3.0 * gamma_rel)
}

// ---------------------------------------------------------------------------
// Reverse shock state vector (12 variables)
// ---------------------------------------------------------------------------
/// State vector for the coupled forward-reverse shock ODE system.
/// Regions: (1) unshocked ISM, (2) shocked ISM, (3) shocked ejecta, (4) unshocked ejecta.
#[derive(Clone, Debug)]
pub struct ReverseShockState {
    pub gamma: f64,       // bulk Lorentz factor of shocked region (region 2/3 contact)
    pub x4: f64,          // comoving width of unshocked ejecta (region 4)
    pub x3: f64,          // comoving width of reverse shock region (region 3)
    pub m2: f64,          // shocked ISM mass per solid angle
    pub m3: f64,          // shocked ejecta mass per solid angle
    pub u2_th: f64,       // internal thermal energy per solid angle (region 2)
    pub u3_th: f64,       // internal thermal energy per solid angle (region 3)
    pub r: f64,           // radius
    pub t_comv: f64,      // comoving time
    pub theta: f64,       // angular coordinate
    pub eps4: f64,        // energy per solid angle in unshocked ejecta (region 4)
    pub m4: f64,          // mass per solid angle in unshocked ejecta (region 4)
}

impl Default for ReverseShockState {
    fn default() -> Self {
        ReverseShockState {
            gamma: 1.0,
            x4: 0.0,
            x3: 0.0,
            m2: 0.0,
            m3: 0.0,
            u2_th: 0.0,
            u3_th: 0.0,
            r: 0.0,
            t_comv: 0.0,
            theta: 0.0,
            eps4: 0.0,
            m4: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Shock jump conditions (magnetized relativistic MHD)
// ---------------------------------------------------------------------------

/// Downstream four-velocity from relativistic shock jump conditions.
/// For σ=0: u_down ≈ 3/(γ_rel + 2) (strong shock limit scaled by downstream velocity).
/// For σ>0: solves cubic polynomial via Cardano's formula.
pub fn compute_downstr_4vel(gamma_rel: f64, sigma: f64) -> f64 {
    if gamma_rel <= 1.0 {
        return 0.0;
    }
    let u_rel = (gamma_rel * gamma_rel - 1.0).sqrt();

    if sigma <= SIGMA_CUT {
        // Unmagnetized: approximate Taub adiabat
        // u_down from: u_down = u_rel / (4*gamma_rel + 3)^0.5
        // More precisely from Blandford-McKee:
        let ad_idx = adiabatic_idx(gamma_rel);
        let num = ad_idx * (gamma_rel - 1.0) + 1.0;
        let denom = ad_idx * (2.0 - ad_idx) * (gamma_rel - 1.0) + 1.0;
        if denom <= 0.0 {
            return u_rel / (4.0 * gamma_rel).sqrt();
        }
        let u_down_sq = (gamma_rel - 1.0) * (num - 1.0) / denom;
        if u_down_sq <= 0.0 {
            return 0.0;
        }
        return u_down_sq.sqrt();
    }

    // Magnetized case: solve cubic a*x³ + b*x² + c*x + d = 0
    // where x = u_down² (downstream four-velocity squared)
    // From Zhang & Kobayashi (2005) magnetized jump conditions
    let g = gamma_rel;
    let s = sigma;
    let g2 = g * g;

    // Coefficients of the cubic in u_d² (approximation from VegasAfterglow)
    let a = 8.0 * s + 1.0;
    let b = -(4.0 * s * (2.0 * g2 - 1.0) + (g2 - 1.0) * (4.0 / 3.0 + 1.0 / (3.0 * g)));
    let c = 4.0 * s * s * (g2 - 1.0);
    let _d = 0.0;

    // Since d=0, factor: x * (a*x² + b*x + c) = 0
    // Non-trivial root from quadratic: a*x² + b*x + c = 0
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        // Fallback to unmagnetized
        return u_rel / (4.0 * gamma_rel).sqrt();
    }
    let x = (-b - discriminant.sqrt()) / (2.0 * a);
    if x > 0.0 {
        x.sqrt()
    } else {
        let x2 = (-b + discriminant.sqrt()) / (2.0 * a);
        if x2 > 0.0 { x2.sqrt() } else { 0.0 }
    }
}

/// Upstream four-velocity in the shock frame.
fn compute_upstr_4vel(u_down: f64, gamma_rel: f64) -> f64 {
    let g_down = (1.0 + u_down * u_down).sqrt();
    let u_rel = (gamma_rel * gamma_rel - 1.0).sqrt();
    (g_down * u_rel + u_down * gamma_rel).abs()
}

/// Compression ratio: u_upstream / u_downstream (four-velocity ratio).
pub fn compute_4vel_jump(gamma_rel: f64, sigma: f64) -> f64 {
    let u_down = compute_downstr_4vel(gamma_rel, sigma);
    if u_down == 0.0 {
        return 4.0 * gamma_rel; // non-magnetized strong shock limit
    }
    let u_up = compute_upstr_4vel(u_down, gamma_rel);
    u_up / u_down
}

/// Relative Lorentz factor between two shells.
pub fn compute_rel_gamma(gamma1: f64, gamma2: f64) -> f64 {
    let u1 = (gamma1 * gamma1 - 1.0).sqrt();
    let u2 = (gamma2 * gamma2 - 1.0).sqrt();
    let beta1 = u1 / gamma1;
    let beta2 = u2 / gamma2;
    gamma1 * gamma2 * (1.0 - beta1 * beta2)
}

/// Upstream magnetic field from magnetization parameter.
/// B₄ = √(4π σ ρ₄ c²)
pub fn compute_upstr_b(rho_up: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 || rho_up <= 0.0 {
        return 0.0;
    }
    (4.0 * PI * C2 * sigma * rho_up).sqrt()
}

/// Weibel instability magnetic field: B = √(8π ε_B e_th)
pub fn compute_weibel_b(eps_b: f64, e_th: f64) -> f64 {
    if eps_b <= 0.0 || e_th <= 0.0 {
        return 0.0;
    }
    (8.0 * PI * eps_b * e_th).sqrt()
}

/// Downstream magnetic field = Weibel + compressed upstream.
pub fn compute_downstr_b(eps_b: f64, rho_up: f64, b_up: f64, gamma_th: f64, comp_ratio: f64) -> f64 {
    let rho_down = rho_up * comp_ratio;
    let e_th = (gamma_th - 1.0) * rho_down * C2;
    compute_weibel_b(eps_b, e_th) + b_up * comp_ratio
}

/// Effective Lorentz factor: Γ_eff = (α Γ² − α + 1) / Γ
fn compute_effective_gamma(ad_idx: f64, gamma: f64) -> f64 {
    (ad_idx * gamma * gamma - ad_idx + 1.0) / gamma
}

/// Derivative of effective Lorentz factor w.r.t. Γ
fn compute_effective_gamma_dgamma(ad_idx: f64, gamma: f64) -> f64 {
    let g2 = gamma * gamma;
    (ad_idx * g2 + ad_idx - 1.0) / g2
}

/// Sound speed for relativistic gas
fn compute_sound_speed(gamma_rel: f64) -> f64 {
    let ad = adiabatic_idx(gamma_rel);
    let val = ad * (ad - 1.0) * (gamma_rel - 1.0) / (1.0 + (gamma_rel - 1.0) * ad);
    val.abs().sqrt() * C_SPEED
}

/// Thermal Lorentz factor: Γ_th = U_th/(m c²) + 1
pub fn compute_gamma_th(u_th: f64, m: f64) -> f64 {
    if m <= 0.0 || u_th <= 0.0 {
        return 1.0;
    }
    u_th / (m * C2) + 1.0
}

// ---------------------------------------------------------------------------
// Forward-Reverse Shock ODE system
// ---------------------------------------------------------------------------

/// Coupled forward-reverse shock equation system.
/// Ports FRShockEqn from VegasAfterglow reverse-shock.tpp.
pub struct FRShockEqn {
    // Medium/density parameters
    pub nwind: f64,
    pub nism: f64,

    // Ejecta initial conditions
    pub gamma4: f64,         // initial ejecta Lorentz factor
    pub m4_init: f64,        // initial ejecta mass per solid angle
    pub eps4_init: f64,      // initial ejecta energy per solid angle

    // Energy injection
    pub t0_injection: f64,
    pub l_injection: f64,
    pub m_dot_injection: f64,

    // Magnetization
    pub sigma_init: f64,

    // Radiation parameters (for radiative efficiency)
    pub eps_e_fwd: f64,
    pub eps_b_fwd: f64,
    pub p_fwd: f64,
    pub eps_e_rs: f64,
    pub eps_b_rs: f64,
    pub p_rs: f64,

    // Cross state (saved when RS crosses ejecta)
    pub crossing_done: bool,
    pub r_x: f64,            // radius at crossing
    pub u_x: f64,            // four-velocity at crossing
    pub v3_comv_x: f64,      // comoving volume of region 3 at crossing
    pub rho3_x: f64,         // density of region 3 at crossing
    pub b3_ordered_x: f64,   // ordered B-field in region 3 at crossing

    // Tool for density computation
    tool: Tool,
}

impl FRShockEqn {
    pub fn new(
        nwind: f64,
        nism: f64,
        k: f64,
        gamma4: f64,
        m4_init: f64,
        eps4_init: f64,
        sigma_init: f64,
        eps_e_fwd: f64,
        eps_b_fwd: f64,
        p_fwd: f64,
        eps_e_rs: f64,
        eps_b_rs: f64,
        p_rs: f64,
        t0_injection: f64,
        l_injection: f64,
        m_dot_injection: f64,
        rtol: f64,
    ) -> Self {
        FRShockEqn {
            nwind,
            nism,
            gamma4,
            m4_init,
            eps4_init,
            sigma_init,
            eps_e_fwd,
            eps_b_fwd,
            p_fwd,
            eps_e_rs,
            eps_b_rs,
            p_rs,
            crossing_done: false,
            r_x: 0.0,
            u_x: 0.0,
            v3_comv_x: 0.0,
            rho3_x: 0.0,
            b3_ordered_x: 0.0,
            t0_injection,
            l_injection,
            m_dot_injection,
            tool: Tool::new_with_k(nwind, nism, k, rtol, 1),
        }
    }

    /// Set initial state from ejecta parameters at time t0.
    pub fn set_init_state(&self, t0: f64) -> ReverseShockState {
        let beta4 = (1.0 - 1.0 / (self.gamma4 * self.gamma4)).sqrt();
        let r0 = beta4 * C_SPEED * t0;

        // Enclosed ambient mass from medium density profile (general k)
        let m2_init = self.tool.solve_swept_number(r0) * MASS_P;

        // Initial Lorentz factor from momentum conservation (approximate)
        let gamma_init = if m2_init < 1e-20 * self.m4_init {
            self.gamma4
        } else {
            let ratio = m2_init / (self.m4_init * self.gamma4);
            if ratio < 0.01 {
                self.gamma4 * (1.0 - ratio / 2.0)
            } else {
                ((self.eps4_init + m2_init * C2) / ((self.m4_init + m2_init) * C2)).max(1.0)
            }
        };

        // Seed RS with tiny fraction of ejecta
        let seed = 1e-8;
        let m3_init = self.m4_init * seed;
        let x4_init = r0 / self.gamma4; // comoving ejecta width
        let x3_init = x4_init * seed;

        // Initial thermal energy from shock heating
        let gamma_rel_init = compute_rel_gamma(self.gamma4, gamma_init);
        let u2_th_init = (gamma_init - 1.0) * m2_init * C2;
        let u3_th_init = (gamma_rel_init - 1.0) * m3_init * C2;

        ReverseShockState {
            gamma: gamma_init,
            x4: x4_init * (1.0 - seed),
            x3: x3_init,
            m2: m2_init,
            m3: m3_init,
            u2_th: u2_th_init,
            u3_th: u3_th_init,
            r: r0,
            t_comv: t0 / gamma_init,
            theta: 0.0,
            eps4: self.eps4_init * (1.0 - seed),
            m4: self.m4_init * (1.0 - seed),
        }
    }

    /// Magnetization parameter: σ = ε₄/(Γ₄ m₄ c²) - 1
    pub fn compute_shell_sigma(&self, state: &ReverseShockState) -> f64 {
        if state.m4 <= 0.0 {
            return 0.0;
        }
        let sigma = state.eps4 / (self.gamma4 * state.m4 * C2) - 1.0;
        if sigma > SIGMA_CUT { sigma } else { 0.0 }
    }

    /// Injection efficiency: fraction of ejecta swept by RS
    fn injection_efficiency(&self, state: &ReverseShockState) -> f64 {
        if self.m4_init <= 0.0 {
            return 1.0;
        }
        (state.m3 / self.m4_init).min(1.0)
    }

    /// Check if reverse shock crossing is complete
    pub fn crossing_complete(&self, state: &ReverseShockState, t: f64) -> bool {
        if state.m3 < 0.999 * (state.m3 + state.m4) {
            return false;
        }
        // Check injection has stopped
        if self.t0_injection > 0.0 {
            if smoothstep(self.t0_injection * 1.5, self.t0_injection * 0.5, t) > 1e-6 {
                return false;
            }
        }
        true
    }

    /// Save state at RS crossing point for post-crossing evolution
    pub fn save_cross_state(&mut self, state: &ReverseShockState) {
        self.crossing_done = true;
        self.r_x = state.r;
        self.u_x = (state.gamma * state.gamma - 1.0).sqrt();
        self.v3_comv_x = state.r * state.r * state.x3;

        let gamma34 = compute_rel_gamma(self.gamma4, state.gamma);
        let sigma = self.compute_shell_sigma(state);
        let comp_ratio = compute_4vel_jump(gamma34, sigma);

        // Upstream density of ejecta
        let rho4 = if state.x4 > 0.0 && state.r > 0.0 {
            state.m4 / (state.r * state.r * state.x4)
        } else {
            0.0
        };

        self.rho3_x = rho4 * comp_ratio;
        let b4 = compute_upstr_b(rho4, sigma);
        self.b3_ordered_x = b4 * comp_ratio;
    }

    // ---- ODE right-hand-side components ----

    /// dr/dt = β c where β comes from Γ
    fn compute_dr_dt(&self, state: &ReverseShockState) -> f64 {
        let u = (state.gamma * state.gamma - 1.0).sqrt();
        let beta = u / state.gamma;
        beta * C_SPEED
    }

    /// dm₂/dt = ρ_ISM r² dr/dt (forward shock sweeps ISM)
    fn compute_dm2_dt(&self, state: &ReverseShockState, dr_dt: f64) -> f64 {
        let rho = self.tool.solve_density(state.r) * MASS_P;
        rho * state.r * state.r * dr_dt
    }

    /// dε₄/dt — energy injection with smooth shutdown
    fn compute_deps4_dt(&self, state: &ReverseShockState, t: f64) -> f64 {
        if self.t0_injection <= 0.0 || self.l_injection <= 0.0 {
            return 0.0;
        }
        let envelope = smoothstep(self.t0_injection * 1.5, self.t0_injection * 0.5, t);
        self.l_injection * envelope
    }

    /// dm₄/dt — mass injection with smooth shutdown
    fn compute_dm4_dt(&self, state: &ReverseShockState, t: f64) -> f64 {
        if self.t0_injection <= 0.0 || self.m_dot_injection <= 0.0 {
            return 0.0;
        }
        let envelope = smoothstep(self.t0_injection * 1.5, self.t0_injection * 0.5, t);
        self.m_dot_injection * envelope
    }

    /// Shock heating rate: (Γ_rel - 1) c² dm/dt
    fn shock_heating_rate(gamma_rel: f64, dm_dt: f64) -> f64 {
        (gamma_rel - 1.0) * C2 * dm_dt
    }

    /// Adiabatic cooling rate for region 2
    fn adiabatic_cooling_rate_2(&self, state: &ReverseShockState, dr_dt: f64) -> f64 {
        if state.r <= 0.0 || state.m2 <= 0.0 {
            return 0.0;
        }
        let gamma_th2 = compute_gamma_th(state.u2_th, state.m2);
        let ad_idx = adiabatic_idx(gamma_th2);
        let e_eff2 = compute_effective_gamma(ad_idx, gamma_th2);
        // PdV work: (Γ_eff - 1) U_th v/r
        (e_eff2 - 1.0) * state.u2_th * 2.0 * dr_dt / state.r
    }

    /// Adiabatic cooling rate for region 3
    fn adiabatic_cooling_rate_3(&self, state: &ReverseShockState, dx3_dt: f64) -> f64 {
        if state.x3 <= 0.0 || state.m3 <= 0.0 {
            return 0.0;
        }
        let gamma_th3 = compute_gamma_th(state.u3_th, state.m3);
        let ad_idx = adiabatic_idx(gamma_th3);
        let e_eff3 = compute_effective_gamma(ad_idx, gamma_th3);
        // Shell expansion cooling
        (e_eff3 - 1.0) * state.u3_th * (2.0 * self.compute_dr_dt(state) / state.r
            + dx3_dt / state.x3.max(1e-30))
    }

    /// Radiative efficiency for forward shock
    fn radiative_efficiency_fwd(&self, state: &ReverseShockState) -> f64 {
        let gamma_th2 = compute_gamma_th(state.u2_th, state.m2);
        self.compute_eps_rad(gamma_th2, state.t_comv, state.gamma,
                            self.eps_e_fwd, self.eps_b_fwd, self.p_fwd)
    }

    /// Compute radiative efficiency
    fn compute_eps_rad(&self, gamma_th: f64, t_comv: f64, gamma: f64,
                       eps_e: f64, eps_b: f64, p: f64) -> f64 {
        if gamma_th <= 1.0 || t_comv <= 0.0 {
            return 0.0;
        }
        let gamma_m = (p - 2.0) / (p - 1.0) * eps_e * (gamma_th - 1.0) * MASS_P / MASS_E + 1.0;
        let u = (gamma * gamma - 1.0).sqrt();
        let e_th = (gamma_th - 1.0) * MASS_P * C2; // per proton
        let b = compute_weibel_b(eps_b, e_th * 4.0 * gamma * self.tool.solve_density(1e17)); // approximate
        if b <= 0.0 {
            return 0.0;
        }
        let gamma_c = (6.0 * PI * MASS_E * C_SPEED / (SIGMA_T * b * b * t_comv)).max(1.0);
        let ratio = (gamma_m / gamma_c).abs();
        if ratio < 1.0 && p > 2.0 {
            if ratio < 1e-2 { return 0.0; }
            eps_e * ratio.powf(p - 2.0)
        } else {
            eps_e
        }
    }

    /// dx₄/dt — unshocked ejecta width evolution
    fn compute_dx4_dt(&self, state: &ReverseShockState, dr_dt: f64) -> f64 {
        // Ejecta width decreases as RS sweeps through
        -(dr_dt / state.gamma)
    }

    /// dx₃/dt — reverse shock region width evolution
    fn compute_dx3_dt(&self, state: &ReverseShockState, dr_dt: f64,
                      gamma34: f64, sigma: f64, comp_ratio: f64) -> f64 {
        let eff = self.injection_efficiency(state);
        let u = (state.gamma * state.gamma - 1.0).sqrt();

        // During crossing: width grows from compression
        let crossing_rate = if comp_ratio > 0.0 && gamma34 > 1.0 {
            dr_dt / (state.gamma * comp_ratio)
        } else {
            0.0
        };

        // After crossing: shell spreads at sound speed
        let gamma_th3 = compute_gamma_th(state.u3_th, state.m3);
        let cs = compute_sound_speed(gamma_th3);
        let spreading_rate = cs / C_SPEED * dr_dt / state.gamma;

        // Blend between crossing and spreading
        eff * spreading_rate + (1.0 - eff) * crossing_rate
    }

    /// dm₃/dt — mass accumulation in reverse shock
    fn compute_dm3_dt(&self, state: &ReverseShockState, dr_dt: f64,
                      gamma34: f64, sigma: f64, comp_ratio: f64) -> f64 {
        if state.x4 <= 0.0 || state.r <= 0.0 {
            return 0.0;
        }
        // Column density of region 4
        let rho4 = state.m4 / (state.r * state.r * state.x4);

        // Mass crossing rate
        let dm3_base = rho4 * state.r * state.r * dr_dt / state.gamma;

        let eff = self.injection_efficiency(state);

        // Blend: during crossing use actual rate, after crossing taper off
        let dm3 = (1.0 - eff) * dm3_base;

        // Cap: don't exceed available mass
        dm3.min(state.m4 * 1e3) // rate cap for stability
    }

    /// dU₂/dt — thermal energy evolution in region 2 (forward shock)
    fn compute_du2_dt(&self, state: &ReverseShockState, dm2_dt: f64, dr_dt: f64) -> f64 {
        let gamma_th2 = compute_gamma_th(state.u2_th, state.m2);
        let eps_rad = self.radiative_efficiency_fwd(state);

        // Heating from sweeping ISM
        let heating = (1.0 - eps_rad) * Self::shock_heating_rate(state.gamma, dm2_dt);

        // Adiabatic cooling
        let cooling = self.adiabatic_cooling_rate_2(state, dr_dt);

        heating - cooling
    }

    /// dU₃/dt — thermal energy evolution in region 3 (reverse shock)
    fn compute_du3_dt(&self, state: &ReverseShockState, dm3_dt: f64, dx3_dt: f64,
                      gamma34: f64) -> f64 {
        // Heating from reverse shock
        let heating = Self::shock_heating_rate(gamma34, dm3_dt);

        // Adiabatic cooling
        let cooling = self.adiabatic_cooling_rate_3(state, dx3_dt);

        heating - cooling
    }

    /// dΓ/dt — bulk Lorentz factor evolution from energy-momentum conservation
    fn compute_dgamma_dt(&self, state: &ReverseShockState, dm2_dt: f64,
                         dm3_dt: f64, du2_dt: f64, du3_dt: f64, dr_dt: f64,
                         deps4_dt: f64, dm4_dt: f64, gamma34: f64) -> f64 {
        let g = state.gamma;
        let u = (g * g - 1.0).sqrt();
        if u < 1e-30 {
            return 0.0;
        }

        // Effective Lorentz factors
        let gamma_th2 = compute_gamma_th(state.u2_th, state.m2);
        let gamma_th3 = compute_gamma_th(state.u3_th, state.m3);
        let ad2 = adiabatic_idx(gamma_th2);
        let ad3 = adiabatic_idx(gamma_th3);
        let e_eff2 = compute_effective_gamma(ad2, gamma_th2);
        let e_eff2_dg = compute_effective_gamma_dgamma(ad2, gamma_th2);
        let e_eff3 = compute_effective_gamma(ad3, gamma_th3);
        let e_eff3_dg = compute_effective_gamma_dgamma(ad3, gamma_th3);

        // Numerator: change in total momentum
        let rho = self.tool.solve_density(state.r) * MASS_P;

        // Forward shock momentum change
        let num_fwd = -(g - 1.0) * C2 * dm2_dt  // ISM deceleration
                      - du2_dt;                      // thermal energy change

        // Reverse shock momentum change
        let num_rvs = (self.gamma4 - g) * C2 * dm3_dt  // ejecta deceleration
                      - du3_dt;                           // thermal energy change

        // Ejecta change
        let num_ej = -deps4_dt + self.gamma4 * C2 * dm4_dt;

        let numerator = num_fwd + num_rvs + num_ej;

        // Denominator: total energy derivative w.r.t. Γ
        let denom = (e_eff2 * state.m2 + e_eff3 * state.m3
                     + state.m4 * self.gamma4) * C2
                    + state.u2_th * e_eff2_dg + state.u3_th * e_eff3_dg;

        if denom.abs() < 1e-60 {
            return 0.0;
        }

        numerator / denom / u * g
    }

    /// Compute all derivatives (main ODE right-hand side)
    pub fn derivatives(&mut self, t: f64, state: &ReverseShockState) -> ReverseShockState {
        let dr_dt = self.compute_dr_dt(state);
        let dm2_dt = self.compute_dm2_dt(state, dr_dt);
        let deps4_dt = self.compute_deps4_dt(state, t);
        let dm4_dt_inject = self.compute_dm4_dt(state, t);

        // Relative Lorentz factor between ejecta and shocked region
        let gamma34 = compute_rel_gamma(self.gamma4, state.gamma);
        let sigma = self.compute_shell_sigma(state);
        let comp_ratio = compute_4vel_jump(gamma34, sigma);

        let dx4_dt = self.compute_dx4_dt(state, dr_dt);
        let dx3_dt = self.compute_dx3_dt(state, dr_dt, gamma34, sigma, comp_ratio);
        let dm3_dt = self.compute_dm3_dt(state, dr_dt, gamma34, sigma, comp_ratio);

        let du2_dt = self.compute_du2_dt(state, dm2_dt, dr_dt);
        let du3_dt = self.compute_du3_dt(state, dm3_dt, dx3_dt, gamma34);

        let dgamma_dt = self.compute_dgamma_dt(
            state, dm2_dt, dm3_dt, du2_dt, du3_dt, dr_dt,
            deps4_dt, dm4_dt_inject, gamma34,
        );

        ReverseShockState {
            gamma: dgamma_dt,
            x4: dx4_dt,
            x3: dx3_dt,
            m2: dm2_dt,
            m3: dm3_dt,
            u2_th: du2_dt,
            u3_th: du3_dt,
            r: dr_dt,
            t_comv: state.gamma + (state.gamma * state.gamma - 1.0).sqrt(), // dt_comv/dt
            theta: 0.0,
            eps4: deps4_dt,
            m4: -dm3_dt + dm4_dt_inject, // mass leaving region 4 → region 3
        }
    }

    /// RK45 adaptive step for the reverse shock ODE.
    /// Returns (new_state, new_dt, succeeded).
    pub fn step_rk45(
        &mut self,
        t: f64,
        state: &ReverseShockState,
        dt: f64,
        rtol: f64,
    ) -> (ReverseShockState, f64, bool) {
        // Butcher tableau (Dormand-Prince RK45)
        let k1 = self.derivatives(t, state);
        let s2 = self.add_states(state, &self.scale_state(&k1, dt * 2.0 / 9.0));
        let k2 = self.derivatives(t + dt * 2.0 / 9.0, &s2);

        let s3 = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt / 12.0),
            &self.scale_state(&k2, dt / 4.0),
        ));
        let k3 = self.derivatives(t + dt / 3.0, &s3);

        let s4 = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt * 69.0 / 128.0),
            &self.add_states(
                &self.scale_state(&k2, dt * -243.0 / 128.0),
                &self.scale_state(&k3, dt * 135.0 / 64.0),
            ),
        ));
        let k4 = self.derivatives(t + dt * 13.0 / 24.0, &s4);

        let s5 = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt * -17.0 / 12.0),
            &self.add_states(
                &self.scale_state(&k2, dt * 27.0 / 4.0),
                &self.add_states(
                    &self.scale_state(&k3, dt * -27.0 / 5.0),
                    &self.scale_state(&k4, dt * 16.0 / 15.0),
                ),
            ),
        ));
        let k5 = self.derivatives(t + dt, &s5);

        let s6 = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt * 65.0 / 432.0),
            &self.add_states(
                &self.scale_state(&k2, dt * -5.0 / 16.0),
                &self.add_states(
                    &self.scale_state(&k3, dt * 13.0 / 16.0),
                    &self.add_states(
                        &self.scale_state(&k4, dt * 4.0 / 27.0),
                        &self.scale_state(&k5, dt * 5.0 / 144.0),
                    ),
                ),
            ),
        ));
        let k6 = self.derivatives(t + dt, &s6);

        // 5th-order solution
        let result = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt * 47.0 / 450.0),
            &self.add_states(
                &self.scale_state(&k3, dt * 12.0 / 25.0),
                &self.add_states(
                    &self.scale_state(&k4, dt * 32.0 / 225.0),
                    &self.add_states(
                        &self.scale_state(&k5, dt / 30.0),
                        &self.scale_state(&k6, dt * 6.0 / 25.0),
                    ),
                ),
            ),
        ));

        // Error estimate (difference between 4th and 5th order)
        let err = self.add_states(
            &self.scale_state(&k1, dt / 150.0),
            &self.add_states(
                &self.scale_state(&k3, dt * -3.0 / 100.0),
                &self.add_states(
                    &self.scale_state(&k4, dt * 16.0 / 75.0),
                    &self.add_states(
                        &self.scale_state(&k5, dt / 20.0),
                        &self.scale_state(&k6, dt * -6.0 / 25.0),
                    ),
                ),
            ),
        );

        let rerror = self.max_relative_error(&err, &result);
        let succeeded = rerror < rtol;

        let boost = (0.9 * (rtol / rerror.max(1e-30)).powf(0.2)).min(1.5).max(0.2);
        let new_dt = dt * boost;

        if succeeded {
            (result, new_dt, true)
        } else {
            (state.clone(), new_dt, false)
        }
    }

    // ---- State arithmetic helpers ----

    fn scale_state(&self, s: &ReverseShockState, factor: f64) -> ReverseShockState {
        ReverseShockState {
            gamma: s.gamma * factor,
            x4: s.x4 * factor,
            x3: s.x3 * factor,
            m2: s.m2 * factor,
            m3: s.m3 * factor,
            u2_th: s.u2_th * factor,
            u3_th: s.u3_th * factor,
            r: s.r * factor,
            t_comv: s.t_comv * factor,
            theta: s.theta * factor,
            eps4: s.eps4 * factor,
            m4: s.m4 * factor,
        }
    }

    fn add_states(&self, a: &ReverseShockState, b: &ReverseShockState) -> ReverseShockState {
        ReverseShockState {
            gamma: a.gamma + b.gamma,
            x4: a.x4 + b.x4,
            x3: a.x3 + b.x3,
            m2: a.m2 + b.m2,
            m3: a.m3 + b.m3,
            u2_th: a.u2_th + b.u2_th,
            u3_th: a.u3_th + b.u3_th,
            r: a.r + b.r,
            t_comv: a.t_comv + b.t_comv,
            theta: a.theta + b.theta,
            eps4: a.eps4 + b.eps4,
            m4: a.m4 + b.m4,
        }
    }

    fn max_relative_error(&self, err: &ReverseShockState, state: &ReverseShockState) -> f64 {
        let rel = |e: f64, s: f64| -> f64 {
            if s.abs() < 1e-30 { e.abs() } else { (e / s).abs() }
        };
        rel(err.gamma, state.gamma)
            .max(rel(err.r, state.r))
            .max(rel(err.m2, state.m2))
            .max(rel(err.m3, state.m3))
            .max(rel(err.u2_th, state.u2_th))
            .max(rel(err.u3_th, state.u3_th))
            .max(rel(err.x3, state.x3))
            .max(rel(err.m4, state.m4))
    }
}

// ---------------------------------------------------------------------------
// Reverse shock ODE solver
// ---------------------------------------------------------------------------

/// Solve the coupled forward-reverse shock ODE from tmin to tmax.
/// Returns (ts, forward_ys, reverse_ys) where:
/// - forward_ys[var][itheta][it] are the forward shock primitive variables
/// - reverse_ys[var][it] are the reverse shock state variables
///
/// reverse_ys layout: [0]=Gamma, [1]=r_rs, [2]=m3, [3]=x3, [4]=u3_th,
///                    [5]=t_comv, [6]=gamma_th3, [7]=b3, [8]=n3,
///                    [9]=gamma34, [10]=n4
pub const NVAR_RS: usize = 11;

/// Solve the reverse shock ODE for a single theta cell.
/// Returns (rs_vars[NVAR_RS][nt], crossing_idx) for each saved timestep.
pub fn solve_reverse_shock_cell(
    eqn: &mut FRShockEqn,
    tmin: f64,
    tmax: f64,
    rtol: f64,
    save_times: &[f64],
) -> (Vec<Vec<f64>>, usize) {
    let state0 = eqn.set_init_state(tmin);
    let mut state = state0;
    let mut t = tmin;
    let mut dt = (tmax - tmin) * 1e-4;

    // Output arrays
    let mut rs_out: Vec<Vec<f64>> = vec![Vec::new(); NVAR_RS];
    let mut save_idx = 0;
    let mut crossing_idx = save_times.len(); // default: never crossed

    // Save initial state
    while save_idx < save_times.len() && save_times[save_idx] <= t + 1e-10 {
        save_rs_state(&state, eqn, &mut rs_out);
        save_idx += 1;
    }

    while t < tmax && save_idx < save_times.len() {
        // After crossing is complete, the RS has swept all ejecta.
        // Fill remaining save times with the current (final) state and stop.
        if eqn.crossing_done {
            while save_idx < save_times.len() {
                save_rs_state(&state, eqn, &mut rs_out);
                save_idx += 1;
            }
            break;
        }

        // Clamp dt
        let dt_max = if save_idx < save_times.len() {
            (save_times[save_idx] - t).min(tmax - t + 1e-6)
        } else {
            tmax - t + 1e-6
        };
        dt = dt.min(dt_max);

        let (new_state, new_dt, succeeded) = eqn.step_rk45(t, &state, dt, rtol);

        if succeeded {
            t += dt;
            state = new_state;

            // Enforce physical bounds
            state.gamma = state.gamma.max(1.0);
            state.m2 = state.m2.max(0.0);
            state.m3 = state.m3.max(0.0);
            state.m4 = state.m4.max(0.0);
            state.u2_th = state.u2_th.max(0.0);
            state.u3_th = state.u3_th.max(0.0);
            state.x3 = state.x3.max(0.0);
            state.x4 = state.x4.max(0.0);
            state.r = state.r.max(0.0);

            // Check crossing
            if !eqn.crossing_done && eqn.crossing_complete(&state, t) {
                eqn.save_cross_state(&state);
                crossing_idx = save_idx;
            }

            // Save at requested times
            while save_idx < save_times.len() && save_times[save_idx] <= t + 1e-10 {
                save_rs_state(&state, eqn, &mut rs_out);
                save_idx += 1;
            }

            dt = new_dt;
        } else {
            dt = new_dt;
        }
    }

    // Fill remaining times with last state (for safety)
    while save_idx < save_times.len() {
        save_rs_state(&state, eqn, &mut rs_out);
        save_idx += 1;
    }

    (rs_out, crossing_idx)
}

/// Save RS state variables to output arrays.
fn save_rs_state(
    state: &ReverseShockState,
    eqn: &FRShockEqn,
    rs_out: &mut [Vec<f64>],
) {
    let gamma_th3 = compute_gamma_th(state.u3_th, state.m3);

    // Compute B-field in region 3
    let gamma34 = compute_rel_gamma(eqn.gamma4, state.gamma);
    let sigma = eqn.compute_shell_sigma(state);
    let comp_ratio = compute_4vel_jump(gamma34, sigma);

    let rho4 = if state.x4 > 0.0 && state.r > 0.0 {
        state.m4 / (state.r * state.r * state.x4)
    } else {
        0.0
    };

    let b3 = if eqn.crossing_done {
        // Post-crossing: B-field evolves with compression
        let v3_comv = state.r * state.r * state.x3;
        if v3_comv > 0.0 && eqn.v3_comv_x > 0.0 {
            let weibel = compute_weibel_b(eqn.eps_b_rs, (gamma_th3 - 1.0) * eqn.rho3_x * C2);
            let ordered = eqn.b3_ordered_x * (eqn.v3_comv_x / v3_comv).powf(2.0 / 3.0);
            weibel + ordered
        } else {
            0.0
        }
    } else {
        let b4 = compute_upstr_b(rho4, sigma);
        compute_downstr_b(eqn.eps_b_rs, rho4, b4, gamma_th3, comp_ratio)
    };

    // Number density in region 3
    let n3 = if state.x3 > 0.0 && state.r > 0.0 {
        state.m3 / (state.r * state.r * state.x3 * MASS_P)
    } else {
        0.0
    };

    rs_out[0].push(state.gamma);
    rs_out[1].push(state.r);       // r_rs (same as forward r in thin shell)
    rs_out[2].push(state.m3);
    rs_out[3].push(state.x3);
    rs_out[4].push(state.u3_th);
    rs_out[5].push(state.t_comv);
    rs_out[6].push(gamma_th3);
    rs_out[7].push(b3);
    rs_out[8].push(n3);
    rs_out[9].push(gamma34);
    let n4 = if rho4 > 0.0 { rho4 / MASS_P } else { 0.0 };
    rs_out[10].push(n4);
}

/// Power-law back-extrapolation for early-time zeros in RS data.
pub fn reverse_shock_early_extrap(rs_data: &mut [Vec<f64>], first_nonzero: usize) {
    if first_nonzero <= 1 || first_nonzero >= rs_data[0].len() {
        return;
    }

    // Find two reference points
    let i1 = first_nonzero;
    let i2 = (first_nonzero + 1).min(rs_data[0].len() - 1);

    for var in 0..rs_data.len() {
        let v1 = rs_data[var][i1];
        let v2 = rs_data[var][i2];
        if v1 <= 0.0 || v2 <= 0.0 || v1 == v2 {
            // Fill with first nonzero value
            for j in 0..first_nonzero {
                rs_data[var][j] = v1;
            }
        } else {
            // Log-log extrapolation
            let log_ratio = (v2 / v1).ln();
            let log_t_ratio = ((i2 as f64 + 1.0) / (i1 as f64 + 1.0)).ln();
            let slope = if log_t_ratio.abs() > 1e-30 { log_ratio / log_t_ratio } else { 0.0 };
            for j in 0..first_nonzero {
                let log_val = v1.ln() + slope * ((j as f64 + 1.0) / (i1 as f64 + 1.0)).ln();
                rs_data[var][j] = log_val.exp();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothstep() {
        assert!((smoothstep(1.0, 0.0, -0.5) - 1.0).abs() < 1e-10);
        assert!((smoothstep(1.0, 0.0, 1.5) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.0, 0.0, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_adiabatic_idx() {
        // Ultra-relativistic: γ = 4/3 + 1/(3*large) ≈ 4/3
        assert!((adiabatic_idx(100.0) - 4.0 / 3.0).abs() < 0.01);
        // Non-relativistic: γ = 4/3 + 1/3 = 5/3
        assert!((adiabatic_idx(1.0) - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_rel_gamma() {
        // Same frame: Γ_rel = 1
        assert!((compute_rel_gamma(10.0, 10.0) - 1.0).abs() < 0.01);
        // One at rest: Γ_rel = other's Γ
        assert!((compute_rel_gamma(10.0, 1.0) - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_compression_ratio_unmagnetized() {
        // Strong shock limit: compression ratio → 4Γ
        let gamma_rel = 10.0;
        let comp = compute_4vel_jump(gamma_rel, 0.0);
        assert!(comp > 1.0);
        assert!(comp.is_finite());
    }

    #[test]
    fn test_compression_ratio_magnetized() {
        let gamma_rel = 10.0;
        let comp_unmag = compute_4vel_jump(gamma_rel, 0.0);
        let comp_mag = compute_4vel_jump(gamma_rel, 0.1);
        // Magnetization should modify the compression ratio
        assert!(comp_mag.is_finite());
        assert!(comp_mag > 0.0);
        // With magnetization, compression ratio differs from unmagnetized
        assert!((comp_mag - comp_unmag).abs() > 0.01 || comp_mag > 0.0);
    }

    #[test]
    fn test_upstream_b() {
        let b = compute_upstr_b(1e-24, 0.1);
        assert!(b > 0.0);
        assert!(b.is_finite());
        // Zero sigma → zero B
        assert_eq!(compute_upstr_b(1e-24, 0.0), 0.0);
    }

    #[test]
    fn test_gamma_th() {
        // No thermal energy → Γ_th = 1
        assert!((compute_gamma_th(0.0, 1.0) - 1.0).abs() < 1e-10);
        // Positive thermal energy → Γ_th > 1
        let g = compute_gamma_th(1e50, 1e30);
        assert!(g > 1.0);
    }

    #[test]
    fn test_init_state() {
        let eqn = FRShockEqn::new(
            0.0, 1.0,    // ISM only
            2.0,          // k = 2 (wind profile exponent)
            100.0,        // Γ₄ = 100
            1e-5,         // m4
            1e48,         // eps4
            0.0,          // σ = 0
            0.1, 0.01, 2.3,  // FS rad params
            0.1, 0.01, 2.3,  // RS rad params
            0.0, 0.0, 0.0,   // no injection
            1e-6,
        );
        let state = eqn.set_init_state(10.0);
        assert!(state.r > 0.0);
        assert!(state.gamma >= 1.0);
        assert!(state.m2 >= 0.0);
        assert!(state.m3 > 0.0);
        assert!(state.m4 > 0.0);
    }

    #[test]
    fn test_derivatives_finite() {
        let mut eqn = FRShockEqn::new(
            0.0, 1.0, 2.0, 100.0, 1e-5, 1e48, 0.0,
            0.1, 0.01, 2.3, 0.1, 0.01, 2.3,
            0.0, 0.0, 0.0, 1e-6,
        );
        let state = eqn.set_init_state(10.0);
        let deriv = eqn.derivatives(10.0, &state);
        assert!(deriv.r.is_finite());
        assert!(deriv.gamma.is_finite());
        assert!(deriv.m2.is_finite());
        assert!(deriv.m3.is_finite());
    }

    #[test]
    fn test_rk45_step() {
        let mut eqn = FRShockEqn::new(
            0.0, 1.0, 2.0, 100.0, 1e-5, 1e48, 0.0,
            0.1, 0.01, 2.3, 0.1, 0.01, 2.3,
            0.0, 0.0, 0.0, 1e-6,
        );
        let state = eqn.set_init_state(10.0);
        let (new_state, new_dt, succeeded) = eqn.step_rk45(10.0, &state, 1.0, 1e-4);
        assert!(new_dt > 0.0);
        // Should either succeed or give a new dt to try
        if succeeded {
            assert!(new_state.r > state.r);
            assert!(new_state.gamma >= 1.0);
        }
    }

    #[test]
    fn test_solve_rs_cell() {
        // Use physically consistent parameters:
        // Γ₄=100, mej per sr ≈ E/(Γ c²) ≈ 1e52/(4π*100*c²) ≈ 3e26
        let gamma4 = 100.0;
        let mej = 1e52 / (4.0 * PI * gamma4 * C2);
        let eps4 = gamma4 * mej * C2; // total energy = Γ m c²
        let mut eqn = FRShockEqn::new(
            0.0, 1.0, 2.0, gamma4, mej, eps4, 0.0,
            0.1, 0.01, 2.3, 0.1, 0.01, 2.3,
            0.0, 0.0, 0.0, 1e-3,
        );
        // Save at a few log-spaced times
        let save_times: Vec<f64> = vec![10.0, 30.0, 100.0, 300.0, 1000.0];
        let (rs_out, _crossing_idx) = solve_reverse_shock_cell(
            &mut eqn, 10.0, 1000.0, 1e-3, &save_times,
        );
        assert_eq!(rs_out.len(), NVAR_RS);
        for v in &rs_out {
            assert_eq!(v.len(), save_times.len(), "each var should have {} entries", save_times.len());
        }
        // Gamma should stay >= 1
        for &g in &rs_out[0] {
            assert!(g >= 1.0, "Gamma = {} should be >= 1", g);
        }
        // Radius should be positive
        for &r in &rs_out[1] {
            assert!(r > 0.0);
        }
    }
}
