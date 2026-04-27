import numpy as np
from sklearn.linear_model import LinearRegression

def build_basis(num_features, num_wavelengths):
    """
    Build a set of Gaussian basis spectra.
    """
    wavelengths = np.linspace(4000, 7000, num_wavelengths)
    centers = np.linspace(4100, 6900, num_features)
    widths = np.full(num_features, 200.0)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :])**2)
    return wavelengths, basis

def generate_synthetic_spectra(n_samples, basis):
    """
    Generate synthetic spectra as linear combinations of the basis.
    """
    n_features = basis.shape[1]
    coeffs = np.random.rand(n_samples, n_features)  # random coefficients in [0,1]
    spectra = coeffs @ basis.T  # shape (n_samples, n_wavelengths)
    return spectra, coeffs

def build_filters(num_filters, wavelengths):
    """
    Construct simple Gaussian filter responses.
    """
    centers = np.linspace(4250, 6750, num_filters)
    widths = np.full(num_filters, 300.0)
    filters = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :])**2)
    return filters

def compute_photometry(spectra, filters):
    """
    Integrate spectra through the filter responses.
    """
    # Normalize filters to unit area for simplicity
    norm_filters = filters / np.sum(filters, axis=0)
    photometry = spectra @ norm_filters
    return photometry

def reconstruct_spectra(photometry, basis, filters):
    """
    Reconstruct spectra from photometric measurements.
    """
    # Build matrix A such that y = A @ c
    A = filters @ basis.T   # shape (n_filters, n_features)
    recon_spectra = []
    recon_coeffs = []
    for y in photometry:
        # Solve least-squares for coefficients
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        recon_spectra.append(basis.T @ coeffs)
        recon_coeffs.append(coeffs)
    return np.vstack(recon_spectra), np.vstack(recon_coeffs)

def main():
    # Settings
    n_samples = 50
    n_features = 10
    n_filters = 3
    n_wavelengths = 301

    # Build basis
    wavelengths, basis = build_basis(n_features, n_wavelengths)

    # Generate synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis)

    # Build filters
    filters = build_filters(n_filters, wavelengths)

    # Compute photometry
    photometry = compute_photometry(spectra, filters)

    # Reconstruct spectra
    recon_spectra, recon_coeffs = reconstruct_spectra(photometry, basis, filters)

    # Evaluation
    mse_true = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared reconstruction error: {mse_true:.4f}")

    # Optional: display first spectrum comparison
    import matplotlib.pyplot as plt
    i = 0
    plt.plot(wavelengths, spectra[i], label='True')
    plt.plot(wavelengths, recon_spectra[i], '--', label='Reconstructed')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux (arb. units)')
    plt.legend()
    plt.title('Spectrum Reconstruction')
    plt.show()

if __name__ == "__main__":
    main()