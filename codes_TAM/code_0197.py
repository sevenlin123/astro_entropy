import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def spectral_model(coeffs, wavelengths, n_bases=10):
    """Generate a spectrum as a sum of Gaussian basis functions."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_bases)
    widths = (wavelengths[-1] - wavelengths[0]) / (2 * n_bases)
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for c, w in zip(centers, widths):
        spectrum += coeffs * np.exp(-0.5 * ((wavelengths - c) / w)**2)
    return spectrum

# ---------- Generate synthetic spectra ----------
def generate_synthetic_spectra(n_samples, wavelengths, n_bases=10):
    """Create n_samples spectra with random coefficients."""
    coeffs = np.random.randn(n_samples, n_bases)
    spectra = np.array([spectral_model(c, wavelengths, n_bases) for c in coeffs])
    return spectra, coeffs

# ---------- Filter construction ----------
def generate_filters(n_filters, wavelengths):
    """Create Gaussian filter responses."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_filters)
    widths = (wavelengths[-1] - wavelengths[0]) / (2 * n_filters)
    filters = []
    for c, w in zip(centers, widths):
        filt = np.exp(-0.5 * ((wavelengths - c) / w)**2)
        filters.append(filt / np.trapz(filt, wavelengths))
    return np.array(filters)

# ---------- Photometry ----------
def compute_photometry(spectra, filters, wavelengths):
    """Integrate spectra over filter responses."""
    phot = []
    for filt in filters:
        val = np.trapz(spectra * filt, wavelengths) / np.trapz(filt, wavelengths)
        phot.append(val)
    return np.array(phot).T  # shape: (n_samples, n_filters)

# ---------- Reconstruction ----------
def reconstruct_spectrum(photometry, reg_model, wavelengths):
    """Predict a spectrum from photometric values."""
    pred_flux = reg_model.predict(photometry.reshape(1, -1))
    return pred_flux.flatten()

# ---------- Main workflow ----------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(4000, 7000, 500)  # Å

    # Generate training data
    n_train = 200
    train_specs, _ = generate_synthetic_spectra(n_train, wl)
    n_filters = 5
    filt_resp = generate_filters(n_filters, wl)
    train_phot = compute_photometry(train_specs, filt_resp, wl)

    # Fit linear regression model
    lr = LinearRegression()
    lr.fit(train_phot, train_specs)

    # Generate a new synthetic spectrum to reconstruct
    test_spec, _ = generate_synthetic_spectra(1, wl)
    test_phot = compute_photometry(test_spec, filt_resp, wl)[0]

    # Reconstruct the spectrum
    recon_spec = reconstruct_spectrum(test_phot, lr, wl)

    # Simple comparison
    print("True spectrum mean:", test_spec.mean())
    print("Reconstructed spectrum mean:", recon_spec.mean())
    print("Mean absolute error:", np.mean(np.abs(test_spec - recon_spec)))