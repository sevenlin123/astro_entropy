#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.
"""

import numpy as np
from scipy.special import erfc
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------
# 1. Spectral model
# ----------------------------------------------------------------------
def gaussian_spectrum(wl, amp, cen, wid):
    """Single Gaussian feature."""
    return amp * np.exp(-0.5 * ((wl - cen) / wid) ** 2)

def random_gaussian_spectrum(wl, n_lines=5, rng=None):
    """Generate a random spectrum as sum of Gaussian lines."""
    rng = rng or np.random.default_rng()
    amps = rng.uniform(0.5, 1.5, size=n_lines)
    cens = rng.uniform(wl.min(), wl.max(), size=n_lines)
    wids = rng.uniform(5, 20, size=n_lines)
    spec = np.zeros_like(wl)
    for a, c, w in zip(amps, cens, wids):
        spec += gaussian_spectrum(wl, a, c, w)
    # Add small noise floor
    spec += rng.normal(scale=0.02, size=wl.size)
    return spec

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wl, rng=None):
    """Create an array of synthetic spectra."""
    rng = rng or np.random.default_rng()
    spectra = np.vstack([random_gaussian_spectrum(wl, rng=rng) for _ in range(n_samples)])
    return spectra

# ----------------------------------------------------------------------
# 3. Filter definitions & photometry generation
# ----------------------------------------------------------------------
def gaussian_filter(wl, cen, sigma):
    """Filter transmission curve."""
    return np.exp(-0.5 * ((wl - cen) / sigma) ** 2)

def create_filters(n_filters, wl, rng=None):
    """Generate a list of filter transmission curves."""
    rng = rng or np.random.default_rng()
    cents = np.linspace(wl.min()+50, wl.max()-50, n_filters)
    sigmas = rng.uniform(30, 60, size=n_filters)
    filters = [gaussian_filter(wl, c, s) for c, s in zip(cents, sigmas)]
    return filters

def compute_photometry(spectra, filters):
    """Compute integrated flux in each filter for each spectrum."""
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    phots = np.zeros((n_samples, n_filters))
    for i, filt in enumerate(filters):
        # numerical integration using trapezoidal rule
        phots[:, i] = np.trapz(spectra * filt, axis=1)
    return phots

# ----------------------------------------------------------------------
# 4. Reconstruction via linear regression
# ----------------------------------------------------------------------
def train_reconstruction_model(photometry, spectra):
    """Fit ridge regression model to map photometry -> spectrum."""
    # Use cross-validated alpha selection
    alphas = np.logspace(-4, 4, 50)
    model = RidgeCV(alphas=alphas, store_cv_values=True).fit(photometry, spectra)
    return model

def predict_spectra(model, photometry):
    """Predict spectra from photometric data."""
    return model.predict(photometry)

# ----------------------------------------------------------------------
# 5. Main routine
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)

    # Wavelength grid
    wl = np.linspace(400, 800, 500)  # nm

    # Generate synthetic data
    n_samples = 200
    spectra = generate_synthetic_spectra(n_samples, wl, rng=rng)

    # Create filters and photometry
    n_filters = 7
    filters = create_filters(n_filters, wl, rng=rng)
    photometry = compute_photometry(spectra, filters)

    # Train-test split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = photometry[:split_idx], photometry[split_idx:]
    y_train, y_test = spectra[:split_idx], spectra[split_idx:]

    # Train reconstruction model
    model = train_reconstruction_model(X_train, y_train)

    # Predict on test set
    y_pred = predict_spectra(model, X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    print(f"Reconstruction MSE on test set: {mse:.6f}")

    # Example of first test spectrum reconstruction
    idx = 0
    true_spec = y_test[idx]
    pred_spec = y_pred[idx]
    print("\nExample spectrum (first test sample):")
    print(f"True spectrum (first 10 values): {true_spec[:10]}")
    print(f"Predicted spectrum (first 10 values): {pred_spec[:10]}")

if __name__ == "__main__":
    main()