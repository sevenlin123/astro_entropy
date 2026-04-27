import numpy as np
from sklearn.linear_model import LinearRegression

def create_basis(n_wave=200, n_basis=5):
    """Create a set of Gaussian basis functions over a wavelength grid."""
    wav = np.linspace(400, 800, n_wave)  # nm
    basis = []
    centers = np.linspace(450, 750, n_basis)
    widths = 30 + 20 * np.arange(n_basis)  # increasing widths
    for c, w in zip(centers, widths):
        g = np.exp(-0.5 * ((wav - c) / w) ** 2)
        basis.append(g)
    basis = np.vstack(basis)  # shape (n_basis, n_wave)
    return wav, basis.T  # basis as (n_wave, n_basis)

def generate_synthetic_spectra(n_samples, basis):
    """Generate spectra as linear combinations of basis functions with random coeffs."""
    coeffs = np.random.randn(n_samples, basis.shape[1])
    spectra = coeffs @ basis.T  # (n_samples, n_wave)
    return spectra, coeffs

def create_filters(n_filters=3, wav=None):
    """Create simple rectangular filter transmission curves."""
    if wav is None:
        wav = np.linspace(400, 800, 200)
    filters = []
    for i in range(n_filters):
        low = 400 + i*100
        high = low + 80
        filt = np.where((wav >= low) & (wav <= high), 1.0, 0.0)
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, n_wave)

def compute_photometry(spectra, filters):
    """Integrate spectra over filter responses."""
    return spectra @ filters.T  # (n_samples, n_filters)

def reconstruct_spectra_from_photometry(photometry, basis, filters):
    """Recover spectra by regressing photometry onto basis-filter integrals."""
    # Construct design matrix: A_{ij} = integral of basis_j * filter_i
    A = filters @ basis  # (n_filters, n_basis)
    # Solve for coefficients per sample
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A.T, photometry.T)  # A.T: (n_basis, n_filters), target: (n_filters, n_samples)
    coeffs_hat = reg.coef_.T  # (n_samples, n_basis)
    spectra_hat = coeffs_hat @ basis.T  # (n_samples, n_wave)
    return spectra_hat, coeffs_hat

def main():
    wav, basis = create_basis()
    spectra, true_coeffs = generate_synthetic_spectra(20, basis)
    filters = create_filters(wav=wav)
    photometry = compute_photometry(spectra, filters)
    spectra_rec, coeffs_rec = reconstruct_spectra_from_photometry(photometry, basis, filters)
    mse = np.mean((spectra - spectra_rec) ** 2)
    print(f"Reconstruction MSE: {mse:.4f}")

if __name__ == "__main__":
    main()