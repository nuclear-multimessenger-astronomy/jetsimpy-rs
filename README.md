# jetsimpy-rs

A Rust reimplementation of [jetsimpy](https://github.com/haowang-astro/jetsimpy), a GRB (Gamma-Ray Burst) afterglow simulator. This project provides a high-performance Rust core library with Python bindings via PyO3, offering the same physics and API as the original C++/pybind11 implementation.

## Features

- Hydrodynamic simulation of relativistic jet evolution
- Synchrotron radiation calculation with EATS (Equal Arrival Time Surface) integration
- Support for top-hat, Gaussian, and power-law jet profiles
- Python API compatible with the original jetsimpy

## Project Structure

```
jetsimpy-rs/
├── src/              # Pure Rust core library
│   ├── hydro/        # Hydrodynamic solver (PDE, interpolation)
│   ├── afterglow/    # Radiation models (synchrotron, EATS)
│   ├── math/         # Numerical utilities (integration, root-finding)
│   └── constants.rs  # Physical constants
├── rust/             # PyO3 Python extension (jetsimpy_extension)
├── jetsimpy_rs/      # Python wrappers (Jet, FluxDensity_*, grids)
└── tests/            # Comparison and unit tests
```

## Installation

### Requirements

- Rust toolchain (1.70+)
- Python 3.8+
- [maturin](https://github.com/PyO3/maturin) (`pip install maturin`)

### Build the Rust library

```bash
cargo build --release
```

### Build and install the Python extension

```bash
# Build the PyO3 extension
cd rust && maturin develop --release && cd ..

# Install the Python package
pip install .
```

## Quick Start

```python
import numpy as np
import jetsimpy_rs

# Define physical parameters
P = dict(
    Eiso=1e52,       # isotropic equivalent energy (erg)
    lf=300,          # core Lorentz factor
    theta_c=0.1,     # half-opening angle (rad)
    n0=1,            # ISM number density (cm^-3)
    A=0,             # wind density parameter
    eps_e=0.1,       # electron energy fraction
    eps_b=0.01,      # magnetic energy fraction
    p=2.17,          # electron power-law index
    theta_v=0.0,     # viewing angle (rad)
    d=474.33,        # luminosity distance (Mpc)
    z=0.1,           # redshift
    s=6,             # power-law jet slope
)

# Observing times and frequency
tday = np.logspace(-3, 3, 100)
tsecond = tday * 86400
nu = 1e18  # Hz (X-ray)

# Compute flux densities for different jet profiles
fd_tophat = jetsimpy_rs.FluxDensity_tophat(tsecond, nu, P)
fd_gaussian = jetsimpy_rs.FluxDensity_gaussian(tsecond, nu, P)
fd_powerlaw = jetsimpy_rs.FluxDensity_powerlaw(tsecond, nu, P)
```

## References

- Wang, H., Bhattacharya, M., Gill, R., & Giannios, D. (2024). "jetsimpy: A Highly Efficient Hydrodynamic Code for Gamma-Ray Burst Afterglow." *The Astrophysical Journal Supplement Series*, 273(1), 17. [arXiv:2404.11365](https://arxiv.org/abs/2404.11365)
- Original implementation: [haowang-astro/jetsimpy](https://github.com/haowang-astro/jetsimpy)

## License

See [LICENSE](LICENSE) for details.
