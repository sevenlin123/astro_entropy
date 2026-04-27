import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model – basis spectra (three simple Gaussian profiles)
# ----------------------------------------------------------------------
def basis_spectra(wavelengths):
    """Return an array of basis spectra (each column is a basis function)."""
    gauss = lambda mu, sigma: np.exp(-0.5 * ((wavelengths - mu)/sigma)**2)
    b1 = gauss(400, 30)   # centered at 400 nm
    b2 = gauss(550, 50)   # centered at 550 nm
    b3 = gauss(700, 40)   # centered at 700 nm
    return np.vstack([b1, b2, b3]).T      # shape (n_wave, n_basis)

# ----------------------------------------------------------------------
# Generate synthetic spectra
# ----------------------------------------------------------------------
def synth_spectra(num_samples, wavelengths, rng=None):
    """Generate synthetic spectra as random combinations of basis spectra."""
    rng = rng or np.random.default_rng()
    B = basis_spectra(wavelengths)                 # (n_wave, n_basis)
    coeffs = rng.normal(size=(num_samples, B.shape[1]))  # random weights
    spectra = coeffs @ B.T                         # (n_samples, n_wave)
    return spectra, coeffs

# ----------------------------------------------------------------------
# Filter definitions and photometric integration
# ----------------------------------------------------------------------
def filter_transmission(wavelengths, center, width):
    """Simple top‑hat filter transmission curve."""
    return np.where(np.abs(wavelengths - center) <= width / 2.0, 1.0, 0.0)

def compute_flux(spectrum, wavelengths, center, width):
    """Integrate spectrum over filter transmission."""
    T = filter_transmission(wavelengths, center, width)
    return trapz(spectrum * T, wavelengths)

def photometric_data(spectra, wavelengths, filters):
    """
    Convert each synthetic spectrum into photometric fluxes for given filters.
    `filters` is a list of tuples (center_nm, width_nm).
    """
    num_samples = spectra.shape[0]
    fluxes = np.zeros((num_samples, len(filters)))
    for i, (c, w) in enumerate(filters):
        for j in range(num_samples):
            fluxes[j, i] = compute_flux(spectra[j], wavelengths, c, w)
    return fluxes

# ----------------------------------------------------------------------
# Reconstruction from photometry
# ----------------------------------------------------------------------
def reconstruct_from_photometry(fluxes, wavelengths, filters):
    """
    Reconstruct spectra by solving a linear system:
        A * coeffs = fluxes
    where A_ij = integral of basis_i over filter_j.
    """
    # Build design matrix A (num_filters, num_basis)
    B = basis_spectra(wavelengths)                     # (n_wave, n_basis)
    A = np.empty((len(filters), B.shape[1]))
    for k, (c, w) in enumerate(filters):
        T = filter_transmission(wavelengths, c, w)
        for b in range(B.shape[1]):
            A[k, b] = trapz(B[:, b] * T, wavelengths)

    # Fit linear regression to recover coefficients for each sample
    model = LinearRegression(fit_intercept=False)
    model.fit(A, fluxes.T)          # fit each filter -> coefficients
    coeffs_rec = model.coef_.T      # (num_samples, num_basis)

    # Reconstruct spectra
    spectra_rec = coeffs_rec @ B.T
    return spectra_rec, coeffs_rec

# ----------------------------------------------------------------------
# Demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    wavelengths = np.linspace(350, 800, 1000)           # nm
    num_samples = 10

    # 1. Generate synthetic spectra
    spectra_true, coeffs_true = synth_spectra(num_samples, wavelengths, rng)

    # 2. Define filters (center, width)
    filters = [(450, 50), (600, 70), (750, 60)]

    # 3. Compute photometric data
    fluxes = photometric_data(spectra_true, wavelengths, filters)

    # 4. Reconstruct spectra from photometry
    spectra_rec, coeffs_rec = reconstruct_from_photometry(fluxes, wavelengths, filters)

    # Print comparison of true vs recovered coefficients for first spectrum
    print("True coefficients (first spectrum):", coeffs_true[0])
    print("Recovered coefficients (first spectrum):", coeffs_rec[0])

    # Compute reconstruction error (RMSE per wavelength)
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec)**2, axis=1))
    print("Reconstruction RMSE (mean over samples):", np.mean(rmse))