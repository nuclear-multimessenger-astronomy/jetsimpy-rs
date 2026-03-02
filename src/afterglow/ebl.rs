/// EBL (Extragalactic Background Light) absorption.
///
/// Franceschini & Rodighiero (2018) optical depth table.
/// Applies exp(-τ) attenuation to observed flux at high photon energies.

/// 56 energy bins in TeV from Franceschini & Rodighiero (2018) Table 4 erratum.
const EBL_ENERGIES_TEV: [f64; 56] = [
    0.00520, 0.00631, 0.00767, 0.00932, 0.01132, 0.01375, 0.01671, 0.02030,
    0.02466, 0.02997, 0.03641, 0.04423, 0.05374, 0.06529, 0.07932, 0.09636,
    0.11708, 0.14224, 0.17281, 0.20995, 0.25507, 0.30989, 0.37650, 0.45742,
    0.55573, 0.67516, 0.82027, 0.99657, 1.21076, 1.47098, 1.78712, 2.17122,
    2.63787, 3.20481, 3.89360, 4.73042, 5.74710, 6.98230, 8.48296, 10.30610,
    12.52110, 15.21220, 18.48170, 22.45390, 27.27980, 33.14290, 40.26610,
    48.92030, 59.43440, 72.20830, 87.72760, 106.58200, 129.48900, 157.31900,
    191.13100, 232.21000,
];

/// 10 redshift bins.
const EBL_REDSHIFTS: [f64; 10] = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

/// Optical depth τ(E, z). Rows = energy bins, columns = redshift bins.
/// NaN entries (complete absorption) stored as f64::INFINITY so exp(-∞) = 0.
const EBL_TAU: [[f64; 10]; 56] = [
    // E=0.00520 TeV
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-05, 0.00098, 0.0025],
    // E=0.00631
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00041, 0.00384, 0.00706],
    // E=0.00767
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2e-05, 0.00238, 0.01031, 0.01614],
    // E=0.00932
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00052, 0.0077, 0.02253, 0.03325],
    // E=0.01132
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00269, 0.01862, 0.04384, 0.06308],
    // E=0.01375
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0002, 0.00834, 0.03824, 0.07863, 0.10985],
    // E=0.01671
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.00144, 0.0201, 0.07046, 0.13069, 0.17634],
    // E=0.02030
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.00526, 0.04148, 0.11902, 0.20242, 0.26437],
    // E=0.02466
    [0.0, 0.0, 0.0, 0.0, 0.00026, 0.01411, 0.0764, 0.18647, 0.29553, 0.37522],
    // E=0.02997
    [0.0, 0.0, 0.0, 6e-05, 0.00168, 0.03132, 0.12823, 0.27492, 0.41267, 0.51353],
    // E=0.03641
    [0.0, 0.0, 0.0, 0.00075, 0.00574, 0.06037, 0.19977, 0.38853, 0.56127, 0.68797],
    // E=0.04423
    [0.0, 1e-05, 0.00013, 0.0032, 0.01442, 0.10498, 0.29586, 0.53788, 0.75593, 0.91357],
    // E=0.05374
    [4e-05, 0.00015, 0.0009, 0.00881, 0.03012, 0.17, 0.42821, 0.74227, 1.01767, 1.2124],
    // E=0.06529
    [0.00017, 0.00059, 0.00279, 0.01897, 0.05553, 0.26482, 0.61799, 1.02783, 1.37303, 1.60979],
    // E=0.07932
    [0.00044, 0.00146, 0.00625, 0.03557, 0.09497, 0.40704, 0.89424, 1.42494, 1.85166, 2.13489],
    // E=0.09636
    [0.0009, 0.00292, 0.01196, 0.06187, 0.15657, 0.62209, 1.28834, 1.96569, 2.48455, 2.81862],
    // E=0.11708
    [0.00164, 0.0053, 0.02132, 0.10376, 0.25361, 0.93967, 1.83323, 2.68098, 3.30228, 3.69168],
    // E=0.14224
    [0.00288, 0.00923, 0.03649, 0.17004, 0.40341, 1.38927, 2.56014, 3.59959, 4.33198, 4.78158],
    // E=0.17281
    [0.00485, 0.01543, 0.0603, 0.2716, 0.62492, 1.99735, 3.49334, 4.74466, 5.59439, 6.10519],
    // E=0.20995
    [0.00785, 0.02488, 0.09631, 0.42039, 0.93622, 2.78417, 4.65017, 6.12472, 7.0919, 7.66666],
    // E=0.25507
    [0.01228, 0.03875, 0.14856, 0.62743, 1.35031, 3.75962, 6.02682, 7.72167, 8.80131, 9.44462],
    // E=0.30989
    [0.01849, 0.05813, 0.22045, 0.89889, 1.87116, 4.9114, 7.58655, 9.48565, 10.6756, 11.3903],
    // E=0.37650
    [0.02671, 0.08371, 0.31387, 1.23409, 2.49233, 6.20194, 9.26186, 11.3434, 12.6433, 13.4434],
    // E=0.45742
    [0.03695, 0.11541, 0.42716, 1.62395, 3.19434, 7.56774, 10.966, 13.2189, 14.6472, 15.5431],
    // E=0.55573
    [0.04889, 0.15213, 0.55633, 2.05458, 3.94206, 8.93147, 12.6258, 15.0605, 16.6643, 17.7073],
    // E=0.67516
    [0.06195, 0.19214, 0.69563, 2.50232, 4.69382, 10.2273, 14.2003, 16.903, 18.7701, 20.0347],
    // E=0.82027
    [0.07565, 0.23399, 0.8387, 2.9418, 5.40766, 11.4154, 15.7408, 18.8822, 21.1241, 22.7128],
    // E=0.99657
    [0.08899, 0.27442, 0.97492, 3.34713, 6.05113, 12.5223, 17.399, 21.1526, 23.9144, 25.9438],
    // E=1.21076
    [0.10138, 0.31194, 1.09952, 3.70442, 6.61381, 13.64, 19.3428, 23.8785, 27.3181, 29.8841],
    // E=1.47098
    [0.11221, 0.3444, 1.20615, 4.00661, 7.11168, 14.9226, 21.7192, 27.2691, 31.5636, 34.8158],
    // E=1.78712
    [0.1214, 0.37194, 1.29529, 4.26781, 7.60245, 16.5345, 24.7027, 31.5446, 37.0071, 41.2633],
    // E=2.17122
    [0.12901, 0.3947, 1.37052, 4.52208, 8.16693, 18.5987, 28.5221, 37.193, 44.2853, 50.0201],
    // E=2.63787
    [0.13601, 0.41568, 1.44294, 4.8208, 8.91296, 21.2999, 33.6172, 44.8184, 54.3879, 62.3975],
    // E=3.20481
    [0.14377, 0.43949, 1.53147, 5.22872, 9.95027, 24.9165, 40.6821, 55.7377, 69.2046, 80.6985],
    // E=3.89360
    [0.15419, 0.47162, 1.65382, 5.81803, 11.3558, 29.901, 50.8562, 71.9941, 91.3676, 108.244],
    // E=4.73042
    [0.16911, 0.51756, 1.83234, 6.62927, 13.2598, 37.152, 66.2602, 96.9755, 125.369, 150.666],
    // E=5.74710
    [0.19065, 0.58408, 2.08465, 7.71468, 15.8888, 47.9635, 90.2827, 135.775, 177.7, 218.111],
    // E=6.98230
    [0.21876, 0.67023, 2.41055, 9.20728, 19.6965, 64.8492, 128.281, 195.456, 257.948, 338.714],
    // E=8.48296
    [0.25607, 0.78567, 2.85808, 11.3604, 25.5316, 92.0801, 187.29, 285.133, 384.066, 621.861],
    // E=10.30610
    [0.30752, 0.94942, 3.50961, 14.703, 34.7408, 135.263, 275.826, 415.094, 615.195, 1451.43],
    // E=12.52110
    [0.38586, 1.19616, 4.50601, 20.0272, 49.645, 200.724, 399.269, 608.09, 1175.933, 3855.73],
    // E=15.21220
    [0.51166, 1.59762, 6.14335, 28.6023, 73.4936, 293.144, 563.797, 952.984, 2741.434, 9943.42],
    // E=18.48170
    [0.71277, 2.23537, 8.72773, 42.356, 109.697, 414.81, 786.935, 1776.48, 6841.312, f64::INFINITY],
    // E=22.45390
    [1.03624, 3.25844, 12.901, 63.5006, 160.938, 565.925, 1156.41, 3963.61, f64::INFINITY, f64::INFINITY],
    // E=27.27980
    [1.55756, 4.9016, 19.4448, 93.6672, 227.793, 749.334, 2000.75, 9319.26, f64::INFINITY, f64::INFINITY],
    // E=33.14290
    [2.3373, 7.33367, 28.8458, 133.104, 308.522, 1013.7, 4181.95, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=40.26610
    [3.40014, 10.6261, 41.1443, 180.335, 400.08, 1548.18, 9404.14, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=48.92030
    [4.72388, 14.6948, 55.8835, 233.223, 505.738, 2907.47, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=59.43440
    [6.2396, 19.311, 72.197, 290.373, 660.765, 6283.0, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=72.20830
    [7.82189, 24.1678, 89.2541, 364.018, 1014.58, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=87.72760
    [9.50683, 29.3591, 109.08, 522.73, 1997.66, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=106.58200
    [11.8962, 37.2086, 147.522, 996.47, 4510.25, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=129.48900
    [18.3001, 59.3783, 268.848, 2318.77, 9990.42, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=157.31900
    [39.6796, 132.463, 642.223, 5410.71, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=191.13100
    [99.778, 333.153, 1579.26, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    // E=232.21000
    [235.983, 777.945, 3498.49, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
];

/// Convert frequency in Hz to energy in TeV.
/// E_TeV = h*ν / (1 TeV) = ν / 2.418e26
const HZ_TO_TEV: f64 = 1.0 / 2.418e26;

/// Compute EBL optical depth τ(ν, z) via bilinear interpolation in (ln E, z).
///
/// Returns 0 if ν is below the table minimum or z < 0.01.
/// Clamps at table boundaries (no extrapolation beyond the grid).
pub fn ebl_tau(nu_hz: f64, z: f64) -> f64 {
    let e_tev = nu_hz * HZ_TO_TEV;

    // Below table minimum energy or minimum redshift: transparent
    if e_tev < EBL_ENERGIES_TEV[0] || z < EBL_REDSHIFTS[0] {
        return 0.0;
    }

    let n_e = EBL_ENERGIES_TEV.len();
    let n_z = EBL_REDSHIFTS.len();

    // Find energy index (in ln space)
    let ln_e = e_tev.ln();
    let mut ie = n_e - 2; // default to last interval
    for i in 0..n_e - 1 {
        if ln_e < EBL_ENERGIES_TEV[i + 1].ln() {
            ie = i;
            break;
        }
    }

    // Find redshift index
    let mut iz = n_z - 2;
    for i in 0..n_z - 1 {
        if z < EBL_REDSHIFTS[i + 1] {
            iz = i;
            break;
        }
    }

    // Bilinear interpolation weights
    let ln_e0 = EBL_ENERGIES_TEV[ie].ln();
    let ln_e1 = EBL_ENERGIES_TEV[ie + 1].ln();
    let fe = (ln_e - ln_e0) / (ln_e1 - ln_e0);
    let fe = fe.clamp(0.0, 1.0);

    let z0 = EBL_REDSHIFTS[iz];
    let z1 = EBL_REDSHIFTS[iz + 1];
    let fz = (z - z0) / (z1 - z0);
    let fz = fz.clamp(0.0, 1.0);

    // Bilinear interpolation
    let t00 = EBL_TAU[ie][iz];
    let t10 = EBL_TAU[ie + 1][iz];
    let t01 = EBL_TAU[ie][iz + 1];
    let t11 = EBL_TAU[ie + 1][iz + 1];

    // If any corner is INFINITY (NaN-filled entry), return INFINITY
    if t00.is_infinite() || t10.is_infinite() || t01.is_infinite() || t11.is_infinite() {
        return f64::INFINITY;
    }

    (1.0 - fe) * (1.0 - fz) * t00
        + fe * (1.0 - fz) * t10
        + (1.0 - fe) * fz * t01
        + fe * fz * t11
}

/// Apply EBL absorption: returns flux * exp(-τ).
pub fn ebl_absorb(flux: f64, nu_hz: f64, z: f64) -> f64 {
    flux * (-ebl_tau(nu_hz, z)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_below_threshold() {
        // X-ray frequencies are far below the EBL threshold
        let tau = ebl_tau(1e18, 1.0);
        assert_eq!(tau, 0.0, "X-ray should be transparent to EBL");
    }

    #[test]
    fn test_zero_low_redshift() {
        // Below minimum redshift
        let tau = ebl_tau(1e27, 0.005);
        assert_eq!(tau, 0.0);
    }

    #[test]
    fn test_known_table_value() {
        // E=0.17281 TeV at z=0.5 should give τ≈0.62492
        let nu = 0.17281 / HZ_TO_TEV; // convert TeV to Hz
        let tau = ebl_tau(nu, 0.5);
        assert!((tau - 0.62492).abs() < 0.01,
            "Expected τ≈0.62492, got {}", tau);
    }

    #[test]
    fn test_known_table_value_2() {
        // E=18.4817 TeV at z=1.0 should give τ≈414.81
        let nu = 18.4817 / HZ_TO_TEV;
        let tau = ebl_tau(nu, 1.0);
        assert!((tau - 414.81).abs() < 1.0,
            "Expected τ≈414.81, got {}", tau);
    }

    #[test]
    fn test_monotonic_in_energy() {
        // At fixed z, τ should increase with energy
        let z = 0.5;
        let tau_low = ebl_tau(0.05 / HZ_TO_TEV, z);
        let tau_high = ebl_tau(1.0 / HZ_TO_TEV, z);
        assert!(tau_high > tau_low,
            "τ should increase with energy: low={}, high={}", tau_low, tau_high);
    }

    #[test]
    fn test_monotonic_in_redshift() {
        // At fixed energy, τ should increase with z
        let nu = 1.0 / HZ_TO_TEV;
        let tau_low = ebl_tau(nu, 0.1);
        let tau_high = ebl_tau(nu, 1.0);
        assert!(tau_high > tau_low,
            "τ should increase with z: low={}, high={}", tau_low, tau_high);
    }

    #[test]
    fn test_full_absorption_high_energy() {
        // Very high energy at high z should give very large τ (complete absorption)
        let nu = 200.0 / HZ_TO_TEV;
        let tau = ebl_tau(nu, 2.0);
        assert!(tau > 100.0, "Very high energy should be fully absorbed, τ={}", tau);
    }

    #[test]
    fn test_ebl_absorb() {
        let flux = 1.0;
        let absorbed = ebl_absorb(flux, 1e18, 1.0);
        assert_eq!(absorbed, flux, "X-ray flux should be unchanged by EBL");

        let absorbed_tev = ebl_absorb(flux, 1.0 / HZ_TO_TEV, 1.0);
        assert!(absorbed_tev < flux, "TeV flux should be absorbed");
    }

    #[test]
    fn test_interpolation_between_grid_points() {
        // Interpolated value should be between bracketing grid values
        let z = 0.2; // between 0.1 and 0.3
        let nu = 0.5 / HZ_TO_TEV;
        let tau = ebl_tau(nu, z);
        let tau_low = ebl_tau(nu, 0.1);
        let tau_high = ebl_tau(nu, 0.3);
        assert!(tau >= tau_low && tau <= tau_high,
            "Interpolated τ={} should be between {:.4} and {:.4}", tau, tau_low, tau_high);
    }
}
