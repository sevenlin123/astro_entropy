#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------
# 1. Spectral model
# ------------------------------------------------------------
def create_wavelength_grid(n_points=1000, wl_min=300.0, wl_max=1100.0):
    """Uniform wavelength grid in nm."""
    return np.linspace(wl_min, wl_max, n_points)

def gaussian(wl, amp, cen, wid):
    """Single Gaussian line."""
    return amp * np.exp(-0.5 * ((wl - cen) / wid)**2)

def synthetic_spectrum(wl, n_lines=4, rng=None):
    """Sum of several Gaussian lines with random parameters."""
    rng = rng or np.random.default_rng()
    amps   = rng.uniform(0.5, 2.0, size=n_lines)
    cents  = rng.uniform(350.0, 1000.0, size=n_lines)
    widths = rng.uniform(5.0, 30.0, size=n_lines)
    spec = np.zeros_like(wl)
    for a, c, w in zip(amps, cents, widths):
        spec += gaussian(wl, a, c, w)
    # Add a weak continuum
    spec += 0.2 + 0.1 * rng.standard_normal(size=wl.size)
    return spec

# ------------------------------------------------------------
# 2. Photometric filters
# ------------------------------------------------------------
def filter_gaussian(wl, cen, wid, norm=True):
    """Gaussian filter transmission."""
    filt = np.exp(-0.5 * ((wl - cen) / wid)**2)
    if norm:
        filt /= np.trapz(filt, wl)
    return filt

def build_filter_set(wl):
    """Create a small set of synthetic filters."""
    centers = [400, 520, 650, 780, 910]  # nm
    widths  = [20, 25, 30, 35, 40]       # nm
    return [filter_gaussian(wl, c, w) for c, w in zip(centers, widths)]

# ------------------------------------------------------------
# 3. Forward operation: spectrum → photometry
# ------------------------------------------------------------
def compute_photometry(spectrum, filters, wl):
    """Integrate spectrum through each filter."""
    phot = []
    for filt in filters:
        # Flux = ∫ S(λ) T(λ) dλ / ∫ T(λ) dλ  (filters already normalized)
        f = simps(spectrum * filt, wl)
        phot.append(f)
    return np.array(phot)

# ------------------------------------------------------------
# 4. Reconstruction framework
# ------------------------------------------------------------
def train_reconstructor(X_phot, Y_spec, alpha=1.0):
    """
    Train a linear model mapping photometric vector to full spectrum.
    X_phot: (N_samples, N_filters)
    Y_spec: (N_samples, N_wavelengths)
    Returns fitted estimator.
    """
    reg = Ridge(alpha=alpha, fit_intercept=False, normalize=False)
    reg.fit(X_phot, Y_spec)
    return reg

def reconstruct_spectrum(regressor, photometry):
    """Predict spectrum from photometric measurements."""
    return regressor.predict(photometry.reshape(1, -1))[0]

# ------------------------------------------------------------
# 5. Demo / synthetic experiment
# ------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)

    wl = create_wavelength_grid()
    filters = build_filter_set(wl)

    # Generate training set
    n_train = 200
    train_specs = np.array([synthetic_spectrum(wl, rng=rng) for _ in range(n_train)])
    train_phots = np.array([compute_photometry(spec, filters, wl) for spec in train_specs])

    # Train reconstructor
    reg = train_reconstructor(train_phots, train_specs, alpha=10.0)

    # Test on new synthetic spectrum
    true_spec = synthetic_spectrum(wl, rng=rng)
    true_phot = compute_photometry(true_spec, filters, wl)
    pred_spec = reconstruct_spectrum(reg, true_phot)

    # Simple error metric
    rmse = np.sqrt(np.mean((pred_spec - true_spec)**2))
    print(f"RMSE of reconstruction: {rmse:.4f}")

if __name__ == "__main__":
    main()