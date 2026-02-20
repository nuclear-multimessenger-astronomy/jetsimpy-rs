use std::collections::HashMap;
use crate::constants::*;
use crate::afterglow::blast::Blast;

pub type Dict = HashMap<String, f64>;
pub type RadiationModel = fn(nu: f64, p: &Dict, blast: &Blast) -> f64;

/// Standard synchrotron model (Sari 1998).
pub fn sync(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync requires 'eps_b'");
    let p_val = *p.get("p").expect("sync requires 'p'");

    let n_blast = blast.n_blast;
    let t = blast.t;
    let gamma = blast.gamma;
    let e = blast.e_density;

    let gamma_m = (p_val - 2.0) / (p_val - 1.0) * (eps_e * MASS_P / MASS_E * (gamma - 1.0));
    let b = (8.0 * PI * eps_b * e).sqrt();
    let gamma_c = 6.0 * PI * MASS_E * gamma * C_SPEED / SIGMA_T / b / b / t;
    let nu_m = 3.0 * E_CHARGE * b * gamma_m * gamma_m / 4.0 / PI / C_SPEED / MASS_E;
    let nu_c = 3.0 * E_CHARGE * b * gamma_c * gamma_c / 4.0 / PI / C_SPEED / MASS_E;
    let e_p = 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * n_blast
        / MASS_E
        / C_SPEED
        / C_SPEED;

    let emissivity = if nu_m < nu_c {
        if nu < nu_m {
            e_p * (nu / nu_m).cbrt()
        } else if nu < nu_c {
            e_p * (nu / nu_m).powf(-(p_val - 1.0) / 2.0)
        } else {
            e_p * (nu_c / nu_m).powf(-(p_val - 1.0) / 2.0) * (nu / nu_c).powf(-p_val / 2.0)
        }
    } else {
        if nu < nu_c {
            e_p * (nu / nu_c).cbrt()
        } else if nu < nu_m {
            e_p / (nu / nu_c).sqrt()
        } else {
            e_p / (nu_m / nu_c).sqrt() * (nu / nu_m).powf(-p_val / 2.0)
        }
    };

    emissivity * blast.dr
}

/// Synchrotron with deep Newtonian phase correction.
pub fn sync_dnp(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync_dnp requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync_dnp requires 'eps_b'");
    let p_val = *p.get("p").expect("sync_dnp requires 'p'");

    let n_blast = blast.n_blast;
    let t = blast.t;
    let gamma = blast.gamma;
    let e = blast.e_density;

    let mut gamma_m = (p_val - 2.0) / (p_val - 1.0) * eps_e * MASS_P / MASS_E * (gamma - 1.0);
    let mut f = 1.0;
    if gamma_m <= 1.0 {
        f = (p_val - 2.0) / (p_val - 1.0) * eps_e * MASS_P / MASS_E * (gamma - 1.0);
        gamma_m = 1.0;
    }
    let b = (8.0 * PI * eps_b * e).sqrt();
    let gamma_c = 6.0 * PI * MASS_E * gamma * C_SPEED / SIGMA_T / b / b / t;
    let nu_m = 3.0 * E_CHARGE * b * gamma_m * gamma_m / 4.0 / PI / C_SPEED / MASS_E;
    let nu_c = 3.0 * E_CHARGE * b * gamma_c * gamma_c / 4.0 / PI / C_SPEED / MASS_E;
    let e_p = 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * f * n_blast
        / MASS_E
        / C_SPEED
        / C_SPEED;

    let emissivity = if nu_m < nu_c {
        if nu < nu_m {
            e_p * (nu / nu_m).cbrt()
        } else if nu < nu_c {
            e_p * (nu / nu_m).powf(-(p_val - 1.0) / 2.0)
        } else {
            e_p * (nu_c / nu_m).powf(-(p_val - 1.0) / 2.0) * (nu / nu_c).powf(-p_val / 2.0)
        }
    } else {
        if nu < nu_c {
            e_p * (nu / nu_c).cbrt()
        } else if nu < nu_m {
            e_p / (nu / nu_c).sqrt()
        } else {
            e_p / (nu_m / nu_c).sqrt() * (nu / nu_m).powf(-p_val / 2.0)
        }
    };

    emissivity * blast.dr
}

/// Weighted average model: offset.
pub fn avg_offset(_nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let theta_v = *p.get("theta_v").unwrap();
    let x_tilde = -blast.theta.sin() * blast.phi.cos() * theta_v.cos()
        + blast.theta.cos() * theta_v.sin();
    x_tilde * blast.r
}

/// Weighted average model: sigma_x.
pub fn avg_sigma_x(_nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let theta_v = *p.get("theta_v").unwrap();
    let x_tilde = -blast.theta.sin() * blast.phi.cos() * theta_v.cos()
        + blast.theta.cos() * theta_v.sin();
    x_tilde * blast.r * x_tilde * blast.r
}

/// Weighted average model: sigma_y.
pub fn avg_sigma_y(_nu: f64, _p: &Dict, blast: &Blast) -> f64 {
    let y = blast.theta.sin() * blast.phi.sin();
    y * blast.r * y * blast.r
}

/// Look up a built-in radiation model by name.
pub fn get_radiation_model(name: &str) -> Option<RadiationModel> {
    match name {
        "sync" => Some(sync),
        "sync_dnp" => Some(sync_dnp),
        _ => None,
    }
}

/// Look up a built-in average model by name.
pub fn get_avg_model(name: &str) -> Option<RadiationModel> {
    match name {
        "offset" => Some(avg_offset),
        "sigma_x" => Some(avg_sigma_x),
        "sigma_y" => Some(avg_sigma_y),
        _ => None,
    }
}
