"""
Comparison tests between blastwave and VegasAfterglow.

Each test runs both implementations with identical parameters and asserts
that the median flux ratio is within tolerance. blastwave uses the
sync_ssa_smooth model with eps_rad correction enabled for best agreement.

Requires VegasAfterglow to be installed; tests are skipped otherwise.
"""

import numpy as np
import pytest

VA = pytest.importorskip("VegasAfterglow")

import blastwave

# ── Constants ──
MPC_cm = 3.0856775814913673e24
mJy_cgs = 1e-26

# Shared parameters
theta_c = 0.1
E_iso = 1e52
Gamma0 = 300.0
n0 = 1.0
eps_e = 0.1
eps_B = 0.01
p_val = 2.2
theta_v = 0.0
d_Mpc = 474.33
z = 0.1

P_BW = dict(
    Eiso=E_iso, lf=Gamma0, theta_c=theta_c,
    n0=n0, A=0.0, eps_e=eps_e, eps_b=eps_B, p=p_val,
    theta_v=theta_v, d=d_Mpc, z=z, s=6,
)

# Sparse time array (0.01 to 1000 days)
T_SECONDS = np.logspace(-2, 3, 30) * 86400
NU_XRAY = 1e18
NU_RADIO = 1e9


def _median_ratio(bw, va):
    """Median flux ratio, ignoring near-zero values."""
    mask = (np.abs(va) > 1e-30) & (np.abs(bw) > 1e-30) & np.isfinite(va) & np.isfinite(bw)
    if mask.sum() == 0:
        return 1.0
    return np.median(bw[mask] / va[mask])


def _va_model(jet, medium, fwd_rad, rvs_rad=None):
    """Build a VegasAfterglow Model."""
    return VA.Model(
        jet=jet,
        medium=medium,
        observer=VA.Observer(lumi_dist=d_Mpc * MPC_cm, z=z, theta_obs=theta_v),
        fwd_rad=fwd_rad,
        rvs_rad=rvs_rad,
    )


# ── Forward shock: tophat, no spreading ──

@pytest.fixture(scope="module")
def fs_results():
    """Forward shock comparison at X-ray and radio (no spreading)."""
    # VegasAfterglow
    model_v = _va_model(
        jet=VA.TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0, spreading=False),
        medium=VA.ISM(n_ism=n0),
        fwd_rad=VA.Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val),
    )
    freqs = np.array([NU_RADIO, NU_XRAY])
    rv = model_v.flux_density_grid(T_SECONDS, freqs)
    va_radio = np.array(rv.total[0]) / mJy_cgs
    va_xray = np.array(rv.total[1]) / mJy_cgs

    # blastwave
    jet_bw = blastwave.Jet(
        blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
        0.0, n0, tmin=10.0, tmax=1e10,
        grid=blastwave.ForwardJetRes(theta_c, 129),
        spread=False,
        eps_e=eps_e, eps_b=eps_B, p_fwd=p_val,
    )
    bw_radio = jet_bw.FluxDensity(T_SECONDS, NU_RADIO, P_BW, model="sync_ssa_smooth")
    bw_xray = jet_bw.FluxDensity(T_SECONDS, NU_XRAY, P_BW, model="sync_ssa_smooth")

    return dict(
        va_radio=va_radio, va_xray=va_xray,
        bw_radio=bw_radio, bw_xray=bw_xray,
    )


def test_tophat_xray(fs_results):
    """Forward shock X-ray should agree within 10%."""
    ratio = _median_ratio(fs_results["bw_xray"], fs_results["va_xray"])
    assert 0.90 < ratio < 1.10, f"TopHat X-ray median ratio: {ratio:.4f}"


def test_tophat_radio(fs_results):
    """Forward shock radio should agree within 10%."""
    ratio = _median_ratio(fs_results["bw_radio"], fs_results["va_radio"])
    assert 0.90 < ratio < 1.10, f"TopHat radio median ratio: {ratio:.4f}"


# ── Wind medium ──

def test_wind_medium():
    """Wind medium (A*=0.1) should agree within 15%."""
    A_star = 0.1
    A_bw = 2.99  # blastwave wind normalization for A*=0.1

    # VegasAfterglow
    model_v = _va_model(
        jet=VA.TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0, spreading=False),
        medium=VA.Wind(A_star=A_star),
        fwd_rad=VA.Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val),
    )
    rv = model_v.flux_density_grid(T_SECONDS, np.array([NU_XRAY]))
    va_flux = np.array(rv.total[0]) / mJy_cgs

    # blastwave
    jet_bw = blastwave.Jet(
        blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
        A_bw, 0.0, tmin=10.0, tmax=1e10, k=2.0,
        grid=blastwave.ForwardJetRes(theta_c, 129),
        spread=False,
        eps_e=eps_e, eps_b=eps_B, p_fwd=p_val,
    )
    P_wind = {**P_BW, "n0": 0.0, "A": A_bw}
    bw_flux = jet_bw.FluxDensity(T_SECONDS, NU_XRAY, P_wind, model="sync_ssa_smooth")

    ratio = _median_ratio(bw_flux, va_flux)
    assert 0.85 < ratio < 1.15, f"Wind X-ray median ratio: {ratio:.4f}"


# ── ODE spreading ──

def test_ode_spreading():
    """ODE spreading mode should agree within 10%."""
    # VegasAfterglow (spreading=True is ODE-like)
    model_v = _va_model(
        jet=VA.TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0, spreading=True),
        medium=VA.ISM(n_ism=n0),
        fwd_rad=VA.Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val),
    )
    rv = model_v.flux_density_grid(T_SECONDS, np.array([NU_RADIO]))
    va_flux = np.array(rv.total[0]) / mJy_cgs

    # blastwave ODE spread
    jet_bw = blastwave.Jet(
        blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
        0.0, n0, tmin=10.0, tmax=1e10,
        grid=blastwave.ForwardJetRes(theta_c, 129),
        spread=True, spread_mode="ode",
        eps_e=eps_e, eps_b=eps_B, p_fwd=p_val,
    )
    bw_flux = jet_bw.FluxDensity(T_SECONDS, NU_RADIO, P_BW, model="sync_ssa_smooth")

    ratio = _median_ratio(bw_flux, va_flux)
    assert 0.90 < ratio < 1.10, f"ODE spreading radio median ratio: {ratio:.4f}"


# ── Reverse shock ──

def test_reverse_shock():
    """Reverse shock total flux should agree within 20%."""
    duration = 1000.0

    # VegasAfterglow
    model_v = _va_model(
        jet=VA.TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0,
                         spreading=False, duration=duration),
        medium=VA.ISM(n_ism=n0),
        fwd_rad=VA.Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val),
        rvs_rad=VA.Radiation(eps_e=eps_e, eps_B=eps_B, p=p_val),
    )
    t_rs = np.logspace(-3, 2, 30) * 86400
    rv = model_v.flux_density_grid(t_rs, np.array([3e9]))
    va_flux = np.array(rv.total[0]) / mJy_cgs

    # blastwave
    jet_bw = blastwave.Jet(
        blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
        0.0, n0, tmin=10.0, tmax=1e10,
        grid=blastwave.ForwardJetRes(theta_c, 129),
        spread=False,
        eps_e=eps_e, eps_b=eps_B, p_fwd=p_val,
        include_reverse_shock=True,
        sigma=0.0, eps_e_rs=eps_e, eps_b_rs=eps_B, p_rs=p_val,
        duration=duration,
    )
    P_rs = {**P_BW, "A": 0.0}
    bw_flux = jet_bw.FluxDensity(t_rs, 3e9, P_rs, model="sync_ssa_smooth")

    ratio = _median_ratio(bw_flux, va_flux)
    assert 0.80 < ratio < 1.20, f"Reverse shock radio median ratio: {ratio:.4f}"


# ── SSC (sync-only at 1 keV) ──

def test_ssc_sync_component():
    """SSC sync-only component at X-ray should agree within 10%."""
    eps_e_ssc = 0.3
    eps_B_ssc = 0.001

    # VegasAfterglow (SSC off = sync-only, but IC cooling included)
    model_v = _va_model(
        jet=VA.TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0, spreading=False),
        medium=VA.ISM(n_ism=n0),
        fwd_rad=VA.Radiation(eps_e=eps_e_ssc, eps_B=eps_B_ssc, p=p_val),
    )
    rv = model_v.flux_density_grid(T_SECONDS, np.array([NU_XRAY]))
    va_flux = np.array(rv.total[0]) / mJy_cgs

    # blastwave (sync_ssa_smooth includes IC cooling via Thomson Y)
    P_ssc = {**P_BW, "eps_e": eps_e_ssc, "eps_b": eps_B_ssc}
    jet_bw = blastwave.Jet(
        blastwave.TopHat(theta_c, E_iso, lf0=Gamma0),
        0.0, n0, tmin=10.0, tmax=1e10,
        grid=blastwave.ForwardJetRes(theta_c, 129),
        spread=False,
        eps_e=eps_e_ssc, eps_b=eps_B_ssc, p_fwd=p_val,
    )
    bw_flux = jet_bw.FluxDensity(T_SECONDS, NU_XRAY, P_ssc, model="sync_ssa_smooth")

    ratio = _median_ratio(bw_flux, va_flux)
    assert 0.90 < ratio < 1.10, f"SSC sync X-ray median ratio: {ratio:.4f}"
