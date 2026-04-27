import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def wavelength_grid(start=400.0, stop=800.0, n_points=1000):
    """Create a linearly spaced wavelength array."""
    return np.linspace(start, stop, n_points)

def gaussian_basis(n_basis, wav, width=15.0):
    """Generate a set of Gaussian basis functions."""
    centres = np.linspace(wav.min(), wav.max(), n_basis)
    basis = [np.exp(-0.5 * ((wav - c) / width) ** 2) for c in centres]
    return np.vstack(basis)          # shape (n_basis, n_wav)

def random_coefficients(n_spectra, n_basis, rng):
    """Draw random non‑negative coefficients for synthetic spectra."""
    return rng.uniform(0.05, 1.0, size=(n_spectra, n_basis))

def synthesize_spectra(coeffs, basis):
    """Construct spectra from basis and coefficients."""
    return coeffs @ basis            # (n_spectra, n_wav)

def gaussian_filters(n_filters, wav, width=20.0):
    """Generate a set of Gaussian photometric filters."""
    centres = np.linspace(wav.min(), wav.max(), n_filters)
    filters = [np.exp(-0.5 * ((wav - c) / width) ** 2) for c in centres]
    return np.vstack(filters)        # shape (n_filters, n_wav)

def compute_photometry(spectra, filters, wav):
    """Calculate photometric fluxes for a set of spectra."""
    n_spec = spectra.shape[0]
    n_fil = filters.shape[0]
    phot = np.empty((n_spec, n_fil))
    denom = np.array([trapz(f, wav) for f in filters])
    for i in range(n_spec):
        numer = np.array([trapz(spectra[i] * f, wav) for f in filters])
        phot[i] = numer / denom
    return phot                      # (n_spectra, n_filters)

# --------------------------------------------------------------------
# Reconstruction pipeline
# --------------------------------------------------------------------
def train_reconstruction_model(photometry, coeffs):
    """Fit a linear model that maps photometry → basis coefficients."""
    lr = LinearRegression(fit_intercept=False)
    lr.fit(photometry, coeffs)
    return lr

def reconstruct_spectra(photometry, lr, basis):
    """Reconstruct spectra from photometry using the trained model."""
    coeffs_est = lr.predict(photometry)        # (n_spectra, n_basis)
    return coeffs_est @ basis                 # (n_spectra, n_wav)

# --------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)

    # Spectral grid and basis
    wav = wavelength_grid()
    n_basis = 6
    basis = gaussian_basis(n_basis, wav)

    # Photometric system
    n_filters = 4
    filters = gaussian_filters(n_filters, wav)

    # Training data
    n_train = 500
    coeffs_train = random_coefficients(n_train, n_basis, rng)
    spectra_train = synthesize_spectra(coeffs_train, basis)
    phot_train = compute_photometry(spectra_train, filters, wav)

    # Train model
    lr = train_reconstruction_model(phot_train, coeffs_train)

    # Test data
    n_test = 10
    coeffs_true = random_coefficients(n_test, n_basis, rng)
    spectra_true = synthesize_spectra(coeffs_true, basis)
    phot_test = compute_photometry(spectra_true, filters, wav)

    # Reconstruction
    spectra_rec = reconstruct_spectra(phot_test, lr, basis)

    # Error metrics
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec) ** 2))
    mae  = np.mean(np.abs(spectra_true - spectra_rec))
    print(f"RMSE between true and reconstructed spectra: {rmse:.4f}")
    print(f"MAE   between true and reconstructed spectra: {mae:.4f}")

if __name__ == "__main__":
    main()