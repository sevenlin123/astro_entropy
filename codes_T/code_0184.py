import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model: set of Gaussian basis functions
# ----------------------------------------------------------------------
def build_basis(wavelengths, centers, widths):
    """Return matrix of shape (len(wavelengths), len(centers))"""
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :]) ** 2)
    return basis / basis.sum(axis=0)   # normalize each basis function

# ----------------------------------------------------------------------
# Synthetic spectra generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_spectra, wavelengths, basis, coeff_std=0.5):
    """Generate n_spectra random linear combinations of basis functions."""
    rng = np.random.default_rng()
    coeffs = rng.normal(scale=coeff_std, size=(n_spectra, basis.shape[1]))
    spectra = coeffs @ basis.T          # shape (n_spectra, len(wavelengths))
    return spectra, coeffs

# ----------------------------------------------------------------------
# Photometric filter definitions (simple triangular filters)
# ----------------------------------------------------------------------
def build_filters(filter_specs, wavelengths):
    """Return matrix (len(filters), len(wavelengths)) of filter transmissions."""
    filters = np.zeros((len(filter_specs), len(wavelengths)))
    for i, (center, width) in enumerate(filter_specs):
        # Triangular filter: transmission rises to 1 at center, falls linearly
        left = center - width / 2
        right = center + width / 2
        idx_left = wavelengths >= left
        idx_center = wavelengths <= center
        idx_right = wavelengths >= center
        idx_right_mask = wavelengths <= right
        filters[i, idx_left & idx_center] = (wavelengths[idx_left & idx_center] - left) / (center - left)
        filters[i, idx_center & idx_right & idx_right_mask] = (right - wavelengths[idx_center & idx_right & idx_right_mask]) / (right - center)
    return filters

# ----------------------------------------------------------------------
# Generate photometric measurements from spectra
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """Compute integrated fluxes through each filter (dot product)."""
    return spectra @ filters.T   # shape (n_spectra, n_filters)

# ----------------------------------------------------------------------
# Reconstruction of spectrum from photometry
# ----------------------------------------------------------------------
def reconstruct_spectra(photometry, filters, basis):
    """
    Given photometry (n_spectra x n_filters) and filter matrix,
    fit linear regression to find best coefficients in basis space.
    """
    # Build design matrix that maps basis coefficients to photometry:
    # photometry = coeffs @ (basis @ filters^T)
    # So we solve for coeffs in least squares sense.
    design = basis @ filters.T          # shape (len(wavelengths), n_filters)
    model = LinearRegression(fit_intercept=False)
    model.fit(design.T, photometry.T)   # Transpose to match sklearn shape
    coeffs_rec = model.coef_.T           # shape (n_spectra, len(basis))
    spectra_rec = coeffs_rec @ basis.T
    return spectra_rec, coeffs_rec

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    # Define wavelength grid (400-800 nm)
    wavelengths = np.linspace(400, 800, 200)

    # Build basis functions
    centers = np.linspace(450, 750, 10)
    widths = np.full_like(centers, 30.0)
    basis = build_basis(wavelengths, centers, widths)

    # Generate synthetic spectra
    n_spectra = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(n_spectra, wavelengths, basis)

    # Define filters (center, width)
    filter_specs = [
        (440, 80),   # U
        (550, 90),   # B
        (650, 70),   # V
        (770, 60),   # R
        (890, 100),  # I
    ]
    filters = build_filters(filter_specs, wavelengths)

    # Compute photometric measurements
    photometry = compute_photometry(spectra_true, filters)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, filters, basis)

    # Evaluate reconstruction error
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared reconstruction error: {mse:.4e}")

    # Example: plot one original vs reconstructed spectrum (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(6, 4))
        plt.plot(wavelengths, spectra_true[idx], label='True')
        plt.plot(wavelengths, spectra_rec[idx], '--', label='Reconstructed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux (arb. units)')
        plt.title('Spectrum reconstruction example')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        pass  # matplotlib not available, ignore

if __name__ == "__main__":
    main()