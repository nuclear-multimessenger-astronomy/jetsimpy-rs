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
