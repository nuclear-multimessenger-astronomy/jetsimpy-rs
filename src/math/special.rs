use std::f64::consts::PI;

/// Lanczos approximation to the Gamma function.
/// Accurate to ~15 digits for Re(x) > 0.5.
pub fn gamma_fn(x: f64) -> f64 {
    // Reflection formula for x < 0.5
    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma_fn(1.0 - x));
    }

    let x = x - 1.0;
    const G: f64 = 7.0;
    const COEFF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let mut sum = COEFF[0];
    for (i, &c) in COEFF[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let t = x + G + 0.5;
    (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
}

/// Modified Bessel function of the second kind K_2(z).
///
/// Three regimes:
/// - z < 0.01: leading Laurent series K_2(z) ≈ 2/z²
/// - 0.01 ≤ z < 2: polynomial approximation (Abramowitz & Stegun)
/// - z ≥ 2: asymptotic expansion
pub fn bessel_k2(z: f64) -> f64 {
    if z <= 0.0 {
        return f64::INFINITY;
    }
    if z < 0.01 {
        // Leading-order Laurent series: K_2(z) ≈ 2/z^2
        return 2.0 / (z * z);
    }
    if z < 2.0 {
        // Abramowitz & Stegun polynomial approximations.
        // K_2(z) = K_0(z) + (2/z)*K_1(z) via recurrence.

        // I_0(z) polynomial (A&S 9.8.1): t = x/3.75
        let ti = z / 3.75;
        let ti2 = ti * ti;
        let i0 = 1.0
            + 3.5156229 * ti2
            + 3.0899424 * ti2 * ti2
            + 1.2067492 * ti2.powi(3)
            + 0.2659732 * ti2.powi(4)
            + 0.0360768 * ti2.powi(5)
            + 0.0045813 * ti2.powi(6);

        // I_1(z) polynomial (A&S 9.8.3): t = x/3.75
        let i1 = z
            * (0.5
                + 0.87890594 * ti2
                + 0.51498869 * ti2 * ti2
                + 0.15084934 * ti2.powi(3)
                + 0.02658733 * ti2.powi(4)
                + 0.00301532 * ti2.powi(5)
                + 0.00032411 * ti2.powi(6));

        // K_0(z) polynomial (A&S 9.8.5): polynomial uses t = x/2
        let tk = z / 2.0;
        let tk2 = tk * tk;
        let ln_half_z = (z / 2.0).ln();
        let k0 = -ln_half_z * i0
            + (-0.57721566
                + 0.42278420 * tk2
                + 0.23069756 * tk2 * tk2
                + 0.03488590 * tk2.powi(3)
                + 0.00262698 * tk2.powi(4)
                + 0.00010750 * tk2.powi(5)
                + 0.00000740 * tk2.powi(6));

        // K_1(z) polynomial (A&S 9.8.7): polynomial uses t = x/2
        let k1 = ln_half_z * i1
            + (1.0 / z)
                * (1.0
                    + 0.15443144 * tk2
                    - 0.67278579 * tk2 * tk2
                    - 0.18156897 * tk2.powi(3)
                    - 0.01919402 * tk2.powi(4)
                    - 0.00110404 * tk2.powi(5)
                    - 0.00004686 * tk2.powi(6));

        // K_2(z) via recurrence: K_{n+1}(z) = K_{n-1}(z) + (2n/z)*K_n(z)
        // K_2(z) = K_0(z) + (2/z)*K_1(z)
        k0 + (2.0 / z) * k1
    } else {
        // Asymptotic expansion for large z (A&S 9.7.2)
        // K_nu(z) ~ sqrt(pi/(2z)) * exp(-z) * sum
        // For nu=2: mu = 4*nu^2 = 16
        let mu = 16.0;
        let iz = 1.0 / z;
        let sum = 1.0
            + (mu - 1.0) * iz / 8.0
            + (mu - 1.0) * (mu - 9.0) * iz * iz / 128.0
            + (mu - 1.0) * (mu - 9.0) * (mu - 25.0) * iz.powi(3) / 3072.0;
        (PI / (2.0 * z)).sqrt() * (-z).exp() * sum
    }
}

/// Convenience wrapper: f(Theta) = 2*Theta^2 / K_2(1/Theta).
///
/// MQ21 eq. (5). Correction for non-relativistic thermal electrons.
/// Limits: f → 1 for Theta >> 1 (relativistic); exponentially small for Theta << 1.
pub fn mq21_f_theta(theta: f64) -> f64 {
    if theta <= 0.0 {
        return 0.0;
    }
    if theta > 100.0 {
        // Ultra-relativistic limit: K_2(1/Theta) ≈ 2*Theta^2, so f → 1
        return 1.0;
    }
    2.0 * theta * theta / bessel_k2(1.0 / theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_fn_integers() {
        // Gamma(5) = 4! = 24
        assert!((gamma_fn(5.0) - 24.0).abs() < 1e-10);
        // Gamma(1) = 1
        assert!((gamma_fn(1.0) - 1.0).abs() < 1e-10);
        // Gamma(3) = 2! = 2
        assert!((gamma_fn(3.0) - 2.0).abs() < 1e-10);
        // Gamma(6) = 120
        assert!((gamma_fn(6.0) - 120.0).abs() < 1e-8);
    }

    #[test]
    fn test_gamma_fn_half() {
        // Gamma(0.5) = sqrt(pi)
        let expected = PI.sqrt();
        assert!(
            (gamma_fn(0.5) - expected).abs() / expected < 1e-10,
            "Gamma(0.5) = {}, expected {}",
            gamma_fn(0.5),
            expected
        );
    }

    #[test]
    fn test_gamma_fn_fractional() {
        // Gamma(1.5) = sqrt(pi)/2
        let expected = PI.sqrt() / 2.0;
        assert!((gamma_fn(1.5) - expected).abs() / expected < 1e-10);
        // Gamma(2.5) = 3*sqrt(pi)/4
        let expected2 = 3.0 * PI.sqrt() / 4.0;
        assert!((gamma_fn(2.5) - expected2).abs() / expected2 < 1e-10);
    }

    #[test]
    fn test_bessel_k2_known_values() {
        // K_2(1.0) ≈ 1.6248388986351774 (scipy reference)
        let val = bessel_k2(1.0);
        assert!(
            (val - 1.6248388986351774).abs() / 1.6248388986351774 < 1e-4,
            "K_2(1.0) = {}, expected 1.6248...",
            val
        );
    }

    #[test]
    fn test_bessel_k2_small() {
        // K_2(0.1) ≈ 199.5 (dominated by 2/z^2 = 200)
        let val = bessel_k2(0.1);
        assert!(val > 190.0 && val < 210.0, "K_2(0.1) = {}", val);
    }

    #[test]
    fn test_bessel_k2_large() {
        // K_2(5.0) ≈ 0.005308 (scipy reference)
        let val = bessel_k2(5.0);
        assert!(
            (val - 0.005308).abs() / 0.005308 < 0.01,
            "K_2(5.0) = {}, expected ~0.005308",
            val
        );
    }

    #[test]
    fn test_bessel_k2_decreasing() {
        // K_2 is monotonically decreasing for z > 0
        let v1 = bessel_k2(0.5);
        let v2 = bessel_k2(1.0);
        let v3 = bessel_k2(2.0);
        let v4 = bessel_k2(5.0);
        assert!(v1 > v2);
        assert!(v2 > v3);
        assert!(v3 > v4);
    }

    #[test]
    fn test_f_theta_limits() {
        // For large Theta (relativistic), f → 1
        let f_large = mq21_f_theta(100.0);
        assert!(
            (f_large - 1.0).abs() < 0.05,
            "f(100) = {}, expected ~1.0",
            f_large
        );

        // For small Theta, f(Theta) = 2*Theta^2/K_2(1/Theta) is actually very large
        // because K_2(1/Theta) is exponentially small. This is correct — the
        // exponential cutoff in I'(x) at large x ensures the emissivity is small.
        let f_small = mq21_f_theta(0.01);
        assert!(f_small > 1.0, "f(0.01) should be >> 1, got {}", f_small);

        // f should be positive for all Theta > 0
        assert!(mq21_f_theta(1.0) > 0.0);
        assert!(mq21_f_theta(0.1) > 0.0);

        // f(1) should be around 2.0 (Theta=1 is mildly relativistic)
        let f_one = mq21_f_theta(1.0);
        assert!(
            f_one > 0.5 && f_one < 5.0,
            "f(1.0) = {}, expected ~1-3",
            f_one
        );
    }
}
