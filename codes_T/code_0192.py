import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------
# 1. Spectral model
# ----------------------------------------------------------------------
def gaussian(x, amp, cen, wid):
    """Simple Gaussian."""
    return amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)

def build_spectrum(wavelengths, n_lines=3, rng=None):
    """Generate one synthetic spectrum as sum of Gaussian lines."""
    if rng is None:
        rng = np.random.default_rng()
    amps   = rng.uniform(0.5, 1.5, size=n_lines)
    cents  = rng.uniform(wavelengths.min(), wavelengths.max(), size=n_lines)
    widths = rng.uniform(5.0, 15.0, size=n_lines)
    spec = np.zeros_like(wavelengths)
    for a, c, w in zip(amps, cents, widths):
        spec += gaussian(wavelengths, a, c, w)
    # Add weak continuum
    spec += rng.uniform(0.05, 0.1)
    return spec

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths, rng=None):
    """Create an array of synthetic spectra."""
    if rng is None:
        rng = np.random.default_rng()
    return np.array([build_spectrum(wavelengths, rng=rng) for _ in range(n_samples)])

# ----------------------------------------------------------------------
# 3. Generate photometric data
# ----------------------------------------------------------------------
def gaussian_filter(x, width, center):
    """Filter transmission curve (Gaussian)."""
    return np.exp(-0.5 * ((x - center) / width) ** 2)

def build_filters(n_filters=3, rng=None):
    """Return list of filter transmission curves."""
    if rng is None:
        rng = np.random.default_rng()
    centers = rng.uniform(3500, 7500, size=n_filters)
    widths  = rng.uniform(200, 500, size=n_filters)
    return [lambda x, c=c, w=w: gaussian_filter(x, w, c) for c,w in zip(centers, widths)]

def photometry_from_spectra(spectra, wavelengths, filters):
    """Integrate spectra over each filter to obtain photometric fluxes."""
    n_filters = len(filters)
    n_spectra = spectra.shape[0]
    phots = np.empty((n_spectra, n_filters))
    # Pre‑compute filter integrals
    filter_trapz = np.array([np.trapz(f(wavelengths), wavelengths) for f in filters])
    for i, spec in enumerate(spectra):
        for j, filt in enumerate(filters):
            phots[i, j] = np.trapz(spec * filt(wavelengths), wavelengths) / filter_trapz[j]
    return phots

# ----------------------------------------------------------------------
# 4. Reconstruction
# ----------------------------------------------------------------------
def train_reconstruction_model(photometry, spectra):
    """Fit linear regression to map photometry → spectrum."""
    reg = LinearRegression()
    reg.fit(photometry, spectra)
    return reg

def reconstruct_spectrum(reg, photometry):
    """Predict spectrum from photometry."""
    return reg.predict(photometry)

# ----------------------------------------------------------------------
# 5. Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid
    wavelengths = np.linspace(3000, 8000, 1000)

    # Build filters
    filters = build_filters(n_filters=3, rng=rng)

    # Generate training data
    train_specs = generate_synthetic_spectra(200, wavelengths, rng=rng)
    train_phot  = photometry_from_spectra(train_specs, wavelengths, filters)

    # Train model
    model = train_reconstruction_model(train_phot, train_specs)

    # Generate test data
    test_specs = generate_synthetic_spectra(20, wavelengths, rng=rng)
    test_phot  = photometry_from_spectra(test_specs, wavelengths, filters)

    # Reconstruct
    recon_specs = reconstruct_spectrum(model, test_phot)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(test_specs, recon_specs))
    print(f"Reconstruction RMSE: {rmse:.4f}")

    # Show first test spectrum vs reconstructed
    import matplotlib.pyplot as plt
    idx = 0
    plt.plot(wavelengths, test_specs[idx], label="True")
    plt.plot(wavelengths, recon_specs[idx], '--', label="Reconstructed")
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.legend()
    plt.show()