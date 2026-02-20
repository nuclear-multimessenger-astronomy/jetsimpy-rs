/// Brent's method root-finding. Ported from SciPy via C++ jetsimpy.
pub fn brentq<F: FnMut(f64) -> f64>(
    f: &mut F,
    xa: f64,
    xb: f64,
    xtol: f64,
    rtol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut xpre = xa;
    let mut xcur = xb;
    let mut xblk: f64 = 0.0;
    let mut fblk: f64 = 0.0;
    let mut spre: f64 = 0.0;
    let mut scur: f64 = 0.0;

    let mut fpre = f(xpre);
    let mut fcur = f(xcur);

    if fpre.abs() == 0.0 {
        return Ok(xpre);
    }
    if fcur.abs() == 0.0 {
        return Ok(xcur);
    }
    if fpre * fcur > 0.0 {
        return Err("f(a) and f(b) must have different signs!".into());
    }

    for _ in 0..max_iter {
        if fpre != 0.0 && fcur != 0.0 && (fpre.is_sign_negative() != fcur.is_sign_negative()) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;

        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry;
            if xpre == xblk {
                // interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                // extrapolate
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }

    Err(format!(
        "Solver doesn't converge after {} iterations!",
        max_iter
    ))
}
