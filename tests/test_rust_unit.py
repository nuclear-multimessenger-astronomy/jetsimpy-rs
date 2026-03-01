"""
Unit tests for the jetsimpy-rs Rust extension and Python wrappers.

These tests verify basic functionality without requiring the original jetsimpy.
"""

import numpy as np
import pytest

from jetsimpy_rs import Jet, TopHat, ForwardJetRes


# Shared minimal parameters
P = dict(
    Eiso=1e52,
    lf=300,
    theta_c=0.1,
    n0=1,
    A=0,
    eps_e=0.1,
    eps_b=0.01,
    p=2.17,
    theta_v=0.0,
    d=474.33,
    z=0.1,
)


@pytest.fixture(scope="module")
def jet():
    """Create a solved Jet instance shared across tests in this module."""
    j = Jet(
        TopHat(P["theta_c"], P["Eiso"], lf0=P["lf"]),
        P["A"],
        P["n0"],
        tmin=10.0,
        tmax=1e10,
        grid=ForwardJetRes(P["theta_c"], 129),
        tail=True,
        spread=True,
        cal_level=1,
    )
    return j


def test_jet_creation(jet):
    """Verify that Jet creation and solveJet complete without error."""
    assert jet is not None


def test_pde_data_shapes(jet):
    """Verify t_pde, y_pde, theta_pde return arrays with expected shapes."""
    t = jet.t_pde
    y = jet.y_pde
    theta = jet.theta_pde

    assert t.ndim == 1
    assert t.shape[0] > 0

    assert theta.ndim == 1
    assert theta.shape[0] > 0

    # y_pde shape should be [5, ntheta, nt]
    assert y.ndim == 3
    assert y.shape[0] == 5
    assert y.shape[1] == theta.shape[0]
    assert y.shape[2] == t.shape[0]


def test_interpolation(jet):
    """Verify interpolation methods return finite values."""
    t = jet.t_pde
    theta = jet.theta_pde

    # Pick a time in the middle of the PDE solution range
    t_mid = t[len(t) // 2]
    th_mid = theta[len(theta) // 4]

    bg = jet.beta_gamma(t_mid, th_mid)
    assert np.isfinite(bg), f"beta_gamma returned non-finite: {bg}"

    r = jet.R(t_mid, th_mid)
    assert np.isfinite(r), f"R returned non-finite: {r}"
    assert r > 0, f"R should be positive, got {r}"

    msw = jet.dMsw_dOmega(t_mid, th_mid)
    assert np.isfinite(msw), f"dMsw_dOmega returned non-finite: {msw}"


def test_interpolation_vectorized(jet):
    """Verify interpolation works with array inputs."""
    t = jet.t_pde
    theta = jet.theta_pde

    t_arr = np.array([t[len(t) // 4], t[len(t) // 2], t[3 * len(t) // 4]])
    th_arr = np.full_like(t_arr, theta[len(theta) // 4])

    bg = jet.beta_gamma(t_arr, th_arr)
    assert bg.shape == (3,)
    assert np.all(np.isfinite(bg))


# ─── Forward-mapping flux method tests ───

P_FWD = dict(
    Eiso=1e52, lf=300, theta_c=0.1, n0=1, A=0,
    eps_e=0.1, eps_b=0.01, p=2.17,
    theta_v=0.0, d=474.33, z=0.1, s=6,
)
T_FWD = np.logspace(-1, 2, 30) * 86400
NU_FWD = 1e18


def test_forward_flux_positive_and_finite():
    """Forward-mapping should produce positive, finite flux values."""
    from jetsimpy_rs import FluxDensity_tophat
    fd = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD, flux_method="forward")
    assert fd.shape == T_FWD.shape
    assert np.all(np.isfinite(fd)), "Forward flux contains non-finite values"
    assert np.all(fd > 0), "Forward flux contains non-positive values"


def test_forward_agrees_with_eats():
    """Forward-mapping should agree with EATS within 0.1 dex for on-axis tophat."""
    from jetsimpy_rs import FluxDensity_tophat
    fd_eats = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD)
    fd_fwd = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"Forward vs EATS max deviation: {max_diff:.4f} dex (limit 0.1)"


def test_forward_gaussian():
    """Forward-mapping should work for Gaussian jet profile."""
    from jetsimpy_rs import FluxDensity_gaussian
    fd_eats = FluxDensity_gaussian(T_FWD, NU_FWD, P_FWD)
    fd_fwd = FluxDensity_gaussian(T_FWD, NU_FWD, P_FWD, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"Gaussian forward vs EATS max deviation: {max_diff:.4f} dex"


def test_forward_powerlaw():
    """Forward-mapping should work for power-law jet profile."""
    from jetsimpy_rs import FluxDensity_powerlaw
    fd_eats = FluxDensity_powerlaw(T_FWD, NU_FWD, P_FWD)
    fd_fwd = FluxDensity_powerlaw(T_FWD, NU_FWD, P_FWD, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"Power-law forward vs EATS max deviation: {max_diff:.4f} dex"


def test_forward_offaxis_fallback():
    """Off-axis should silently fall back to EATS (no error, same result)."""
    from jetsimpy_rs import FluxDensity_tophat
    P_off = {**P_FWD, "theta_v": 0.3}
    t_off = np.logspace(0, 2.5, 10) * 86400

    fd_eats = FluxDensity_tophat(t_off, NU_FWD, P_off)
    fd_fwd = FluxDensity_tophat(t_off, NU_FWD, P_off, flux_method="forward")

    # Should be identical (both use EATS for off-axis)
    np.testing.assert_allclose(fd_fwd, fd_eats, rtol=1e-10,
                               err_msg="Off-axis forward should fall back to EATS identically")


def test_forward_ode_spread():
    """Forward-mapping should work with ODE spreading mode."""
    from jetsimpy_rs import Jet, TopHat, ForwardJetRes
    jet = Jet(
        TopHat(P_FWD["theta_c"], P_FWD["Eiso"], lf0=P_FWD["lf"]),
        P_FWD["A"], P_FWD["n0"],
        grid=ForwardJetRes(P_FWD["theta_c"], 129),
        spread_mode="ode", tail=True,
    )
    fd_eats = jet.FluxDensity(T_FWD, NU_FWD, P_FWD)
    fd_fwd = jet.FluxDensity(T_FWD, NU_FWD, P_FWD, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.2, f"ODE forward vs EATS max deviation: {max_diff:.4f} dex"


def test_forward_no_spread():
    """Forward-mapping should work with no-spread mode."""
    from jetsimpy_rs import FluxDensity_tophat
    fd_eats = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD, spread=False)
    fd_fwd = FluxDensity_tophat(T_FWD, NU_FWD, P_FWD, spread=False, flux_method="forward")

    mask = (fd_eats > 0) & (fd_fwd > 0)
    log_diff = np.abs(np.log10(fd_fwd[mask]) - np.log10(fd_eats[mask]))
    max_diff = np.max(log_diff)
    assert max_diff < 0.1, f"No-spread forward vs EATS max deviation: {max_diff:.4f} dex"
