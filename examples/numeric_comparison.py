"""
Numeric model comparison: Chang-Cooper (numeric) vs analytic synchrotron.

Compares model="sync" and model="numeric" at the same physical parameters
to validate the numeric electron distribution solver against the Sari+98
broken power-law approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, "..")
from blastwave import FluxDensity_tophat

# ---------- Plot style ----------
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
    'figure.titlesize': 17,
})

# ---------- Constants ----------
DAY = 86400.0

# ---------- Model parameters ----------
P = {
    "Eiso":    1e53,
    "lf":      300.0,
    "theta_c": 0.1,
    "A":       0.0,
    "n0":      1.0,
    "eps_e":   0.1,
    "eps_b":   0.01,
    "p":       2.2,
    "theta_v": 0.0,
    "d":       1000.0,
    "z":       0.2,
}

kw = dict(tmin=1.0, tmax=1e8)

# ---------- Light curves ----------
t = np.geomspace(0.01 * DAY, 1000.0 * DAY, 200)
tday = t / DAY

bands = [
    ("100 GHz", 1e11, "C0"),
    ("Optical R", 4.56e14, "C2"),
    ("1 keV", 2.418e17, "C3"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for name, nu, color in bands:
    nu_arr = nu * np.ones_like(t)
    print(f"  {name}...")
    f_sync = FluxDensity_tophat(t, nu_arr, P, model="sync", **kw)
    f_num  = FluxDensity_tophat(t, nu_arr, P, model="numeric", **kw)

    m = (f_sync > 0) & (f_num > 0)
    ax1.plot(tday[m], f_sync[m], "-", color=color, lw=2, label=f"{name} sync")
    ax1.plot(tday[m], f_num[m], "--", color=color, lw=2, alpha=0.8, label=f"{name} numeric")

    resid = np.full_like(f_sync, np.nan)
    resid[m] = (f_num[m] / f_sync[m] - 1.0) * 100
    ax2.plot(tday, resid, "-", color=color, lw=2, label=name)

ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("Time (days)"); ax1.set_ylabel("Flux density (mJy)")
ax1.set_title("Light Curves")
ax1.legend(fontsize=8, ncol=2)
ax1.tick_params(which="both", direction="in", top=True, right=True)

ax2.axhline(0, color="gray", ls="--", lw=1)
ax2.axhspan(-10, 10, color="gray", alpha=0.08)
ax2.set_xscale("log")
ax2.set_xlabel("Time (days)"); ax2.set_ylabel("Residual (%)")
ax2.set_title("Numeric / Sync - 1")
ax2.legend()
ax2.set_ylim(-50, 80)
ax2.tick_params(which="both", direction="in", top=True, right=True)

plt.suptitle("Numeric (Chang-Cooper) vs Analytic (Sari+98)", fontweight="bold")
plt.tight_layout()
plt.savefig("numeric_comparison.png", dpi=150, bbox_inches="tight")
print("Saved numeric_comparison.png")
plt.show()
