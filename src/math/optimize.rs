/// Golden section search for minimization. Ported from C++ jetsimpy.
pub fn golden_section<F: FnMut(f64) -> f64>(
    f: &mut F,
    xx1: f64,
    xx2: f64,
    atol: f64,
    max_iter: usize,
) -> f64 {
    let r: f64 = 0.618;

    let mut x1 = xx1;
    let mut x2 = xx2;
    let mut xm1 = (x1 + r * x2) / (1.0 + r);
    let mut f1 = f(x1);
    let mut fm1 = f(xm1);
    let mut f2 = f(x2);

    // bracket the range
    let mut n = 0;
    while n < 10 {
        if fm1 > f1 && fm1 <= f2 {
            x2 = xm1;
            f2 = fm1;
            xm1 = (x1 + r * x2) / (1.0 + r);
            fm1 = f(xm1);
            n += 1;
        } else if fm1 <= f1 && fm1 > f2 {
            x1 = xm1;
            f1 = fm1;
            xm1 = (x1 + r * x2) / (1.0 + r);
            fm1 = f(xm1);
            n += 1;
        } else if fm1 <= f1 && fm1 <= f2 {
            break;
        } else {
            if f1 < f2 {
                x2 = xm1;
                f2 = fm1;
                xm1 = (x1 + r * x2) / (1.0 + r);
                fm1 = f(xm1);
                n += 1;
            } else {
                x1 = xm1;
                f1 = fm1;
                xm1 = (x1 + r * x2) / (1.0 + r);
                fm1 = f(xm1);
                n += 1;
            }
        }
    }

    // if bracketing failed
    if n >= 10 {
        return if fm1 > f1 { x1 } else { x2 };
    }

    // other side
    let mut xm2 = (r * x1 + x2) / (1.0 + r);
    let mut fm2 = f(xm2);

    // loop
    n = 0;
    while (xm1 - xm2).abs() > atol && n < max_iter {
        if fm1 < fm2 {
            x2 = xm2;
            #[allow(unused_assignments)]
            { f2 = fm2; }
            xm2 = xm1;
            fm2 = fm1;
            xm1 = (x1 + r * x2) / (1.0 + r);
            fm1 = f(xm1);
        } else {
            x1 = xm1;
            #[allow(unused_assignments)]
            { f1 = fm1; }
            xm1 = xm2;
            fm1 = fm2;
            xm2 = (r * x1 + x2) / (1.0 + r);
            fm2 = f(xm2);
        }
        n += 1;
    }

    (xm1 + xm2) / 2.0
}

/// Minimization with initial sampling on a range, then golden section refinement.
pub fn minimization_range<F: FnMut(f64) -> f64>(
    f: &mut F,
    x1: f64,
    x2: f64,
    n_opt: usize,
    atol: f64,
    type_: &str,
    max_iter: usize,
) -> Result<f64, String> {
    let mut x_ini = vec![0.0; n_opt];
    let mut f_ini = vec![0.0; n_opt];

    if type_ == "linear" {
        for i in 0..n_opt {
            x_ini[i] = x1 + i as f64 * (x2 - x1) / (n_opt - 1) as f64;
            f_ini[i] = f(x_ini[i]);
        }
    } else if type_ == "log" {
        let asinh_x1 = x1.asinh();
        let asinh_x2 = x2.asinh();
        for i in 0..n_opt {
            x_ini[i] = asinh_x1 + i as f64 * (asinh_x2 - asinh_x1) / (n_opt - 1) as f64;
            x_ini[i] = x_ini[i].sinh();
            f_ini[i] = f(x_ini[i]);
        }
    } else {
        return Err(format!("No such optimization method: '{}'", type_));
    }

    // find minimum
    let argmin = f_ini
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    let x_peak = if argmin == 0 {
        golden_section(f, x_ini[0], x_ini[1], atol, max_iter)
    } else if argmin == n_opt - 1 {
        golden_section(f, x_ini[n_opt - 2], x_ini[n_opt - 1], atol, max_iter)
    } else {
        golden_section(f, x_ini[argmin - 1], x_ini[argmin + 1], atol, max_iter)
    };

    Ok(x_peak)
}

/// Minimization given initial sampling points, then golden section refinement.
pub fn minimization<F: FnMut(f64) -> f64>(f: &mut F, x_ini: &[f64], atol: f64, max_iter: usize) -> f64 {
    let n = x_ini.len();
    let f_ini: Vec<f64> = x_ini.iter().map(|&x| f(x)).collect();

    let argmin = f_ini
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    if argmin == 0 {
        golden_section(f, x_ini[0], x_ini[1], atol, max_iter)
    } else if argmin == n - 1 {
        golden_section(f, x_ini[n - 2], x_ini[n - 1], atol, max_iter)
    } else {
        golden_section(f, x_ini[argmin - 1], x_ini[argmin + 1], atol, max_iter)
    }
}
