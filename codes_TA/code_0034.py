import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Define spectral model
# -----------------------------
def spectral_basis(wavelength):
    """Return basis matrix (n_wave, n_basis) for given wavelengths."""
    return np.vstack([np.ones_like(wavelength),
                      wavelength,
                      wavelength**2]).T  # constant, linear, quadratic

def generate_synthetic_spectra(n_samples, wavelength):
    """Generate synthetic spectra as linear combinations of basis functions."""
    basis = spectral_basis(wavelength)
    coeffs = np.random.randn(n_samples, basis.shape[1])  # random coefficients
    spectra = coeffs @ basis.T  # shape (n_samples, n_wave)
    # add small noise
    spectra += 0.01 * np.random.randn(*spectra.shape)
    return spectra, coeffs

# -----------------------------
# 2. Define filter set
# -----------------------------
def gaussian_filter(wavelength, center, sigma=20.0):
    """Simple Gaussian filter transmission."""
    return np.exp(-0.5 * ((wavelength - center) / sigma)**2)

def create_filters(wavelength):
    """Return list of filter transmissions (n_filters, n_wave)."""
    centers = [360., 440., 550., 650., 790.]  # approximate UBVRI centers in nm
    return np.array([gaussian_filter(wavelength, c) for c in centers])

# -----------------------------
# 3. Generate photometry
# -----------------------------
def generate_photometry(spectra, filters):
    """
    Integrate spectra with each filter to produce synthetic photometric fluxes.
    Returns (n_samples, n_filters).
    """
    fluxes = []
    for filt in filters:
        # integrate spectrum * filter transmission over wavelength
        # using Simpson's rule
        flux = simps(spectra * filt, axis=1) / simps(filt, axis=0)
        fluxes.append(flux)
    return np.column_stack(fluxes)

# -----------------------------
# 4. Reconstruct spectrum
# -----------------------------
def reconstruct_spectra(photometry, filters, wavelength, n_components=3):
    """
    Reconstruct spectra from photometry using linear regression to predict
    coefficients of the basis functions.
    """
    # Build design matrix for photometry: each filter integrated over basis functions
    basis = spectral_basis(wavelength)
    # Precompute integral of basis * filter
    G = np.zeros((len(filters), n_components))
    for i, filt in enumerate(filters):
        for j in range(n_components):
            integrand = basis[:, j] * filt
            G[i, j] = simps(integrand, wavelength) / simps(filt, wavelength)
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(G, np.eye(n_components))  # fit mapping from G to identity
    # Predict coefficients
    coeffs_pred = lr.predict(photometry)
    # Reconstruct spectra
    spectra_rec = coeffs_pred @ basis
    return spectra_rec, coeffs_pred

# -----------------------------
# 5. Main execution
# -----------------------------
def main():
    np.random.seed(42)
    # Wavelength grid
    wl = np.linspace(300., 900., 601)  # 300-900 nm, 1 nm spacing

    # Generate synthetic spectra
    n_samples = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, wl)

    # Create filters
    filters = create_filters(wl)

    # Generate photometry
    phot = generate_photometry(spectra_true, filters)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(phot, filters, wl)

    # Evaluate reconstruction error
    mse = np.mean((spectra_true - spectra_rec)**2)
    print(f"Mean squared error of reconstruction: {mse:.6f}")

    # Show first sample true vs reconstructed spectrum
    import matplotlib.pyplot as plt
    idx = 0
    plt.plot(wl, spectra_true[idx], label='True')
    plt.plot(wl, spectra_rec[idx], '--', label='Reconstructed')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux (arb.u.)')
    plt.legend()
    plt.title('Spectrum reconstruction')
    plt.show()

if __name__ == "__main__":
    main()