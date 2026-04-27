import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model definition
# ----------------------------------------------------------------------
def gaussian_basis(wave, centers, widths):
    """Return basis matrix of Gaussians evaluated at wave."""
    gauss = np.exp(-0.5 * ((wave[:, None] - centers[None, :]) / widths[None, :]) ** 2)
    return gauss  # shape (N_wave, N_basis)

def spectral_model(wave):
    """Define spectral basis: fixed centers and widths."""
    centers = np.linspace(4000, 8000, 10)          # Angstroms
    widths  = 100 * np.ones_like(centers)          # Angstroms
    return gaussian_basis(wave, centers, widths)   # shape (N_wave, N_basis)

# ----------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------
def generate_synthetic_spectrum(wave, n_spectra=5):
    """Generate random spectra as linear combinations of basis functions."""
    basis = spectral_model(wave)
    rng = np.random.default_rng()
    weights = rng.normal(size=(n_spectra, basis.shape[1]))
    spectra = weights @ basis.T                      # shape (n_spectra, N_wave)
    # Add small Gaussian noise
    spectra += rng.normal(scale=0.02, size=spectra.shape)
    return spectra, weights

def generate_filters(n_filters=5):
    """Define simple Gaussian filters."""
    centers = np.linspace(4200, 7800, n_filters)
    widths  = 300 * np.ones_like(centers)
    def filter_response(wave, c, w):
        return np.exp(-0.5 * ((wave - c) / w) ** 2)
    return [(c, w, lambda wave, c=c, w=w: filter_response(wave, c, w))
            for c, w in zip(centers, widths)]

def photometric_fluxes(spectra, wave, filters):
    """Integrate spectra over filter responses to produce synthetic photometry."""
    fluxes = []
    for c, w, filt in filters:
        R = filt(wave)
        # Normalized photometric flux
        flux = np.trapz(spectra * R[:, None], wave, axis=1) / np.trapz(R, wave)
        fluxes.append(flux)
    return np.column_stack(fluxes)  # shape (n_spectra, n_filters)

# ----------------------------------------------------------------------
# Reconstruction framework
# ----------------------------------------------------------------------
def build_forward_matrix(wave, filters, basis):
    """Construct matrix mapping basis coefficients to photometric fluxes."""
    M = np.empty((len(filters), basis.shape[1]))
    for i, (c, w, filt) in enumerate(filters):
        R = filt(wave)
        denom = np.trapz(R, wave)
        for j in range(basis.shape[1]):
            numerator = np.trapz(basis[:, j] * R, wave)
            M[i, j] = numerator / denom
    return M  # shape (n_filters, n_basis)

def reconstruct_weights(M, photometry):
    """Recover basis coefficients by least‑squares fit."""
    reg = LinearRegression(fit_intercept=False).fit(M.T, photometry.T)
    return reg.coef_.T  # shape (n_spectra, n_basis)

def reconstruct_spectra(coefficients, wave, basis):
    """Build spectra from recovered coefficients."""
    return coefficients @ basis.T  # shape (n_spectra, N_wave)

# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wave = np.linspace(3500, 9000, 2000)  # Angstroms

    # Generate synthetic spectra and true weights
    spectra_true, weights_true = generate_synthetic_spectrum(wave)

    # Define filters
    filters = generate_filters()

    # Produce synthetic photometry
    photometry = photometric_fluxes(spectra_true, wave, filters)

    # Build forward matrix and reconstruct weights
    basis = spectral_model(wave)
    M = build_forward_matrix(wave, filters, basis)
    weights_rec = reconstruct_weights(M, photometry)

    # Reconstruct spectra
    spectra_rec = reconstruct_spectra(weights_rec, wave, basis)

    # Print comparison for first spectrum
    print("True vs recovered first spectrum (first 5 values):")
    print("True :", spectra_true[0][:5])
    print("Rec  :", spectra_rec[0][:5])

    print("\nTrue vs recovered weights (first spectrum, first 3 coeffs):")
    print("True :", weights_true[0][:3])
    print("Rec  :", weights_rec[0][:3])