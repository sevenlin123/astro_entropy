import numpy as np

# -------------------------------------------------------------
# Spectral model: Gaussian basis functions
# -------------------------------------------------------------
def make_basis_functions(wavelength, n_basis=10, width=50.0):
    """
    Create n_basis Gaussian basis functions over the wavelength grid.
    """
    centers = np.linspace(wavelength.min(), wavelength.max(), n_basis)
    basis = np.exp(-((wavelength[:, None] - centers[None, :]) ** 2)
                   / (2 * width ** 2))
    return basis  # shape: (N_wave, n_basis)


# -------------------------------------------------------------
# Synthetic spectrum generation
# -------------------------------------------------------------
def generate_synthetic_spectrum(basis, seed=None):
    """
    Generate a random linear combination of the basis functions.
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=basis.shape[1])
    spectrum = basis @ weights
    return spectrum, weights


# -------------------------------------------------------------
# Filter transmission curves
# -------------------------------------------------------------
def make_filters(wavelength, n_filters=5, sigma=30.0):
    """
    Generate simple Gaussian filters across the wavelength grid.
    """
    centers = np.linspace(wavelength.min() + 0.1 * (wavelength.max() - wavelength.min()),
                          wavelength.max() - 0.1 * (wavelength.max() - wavelength.min()),
                          n_filters)
    filters = np.exp(-((wavelength[:, None] - centers[None, :]) ** 2)
                     / (2 * sigma ** 2))
    # Normalize each filter to unit area
    filters /= np.trapz(filters, wavelength, axis=0)
    return filters  # shape: (N_wave, n_filters)


# -------------------------------------------------------------
# Photometry computation
# -------------------------------------------------------------
def compute_photometry(spectrum, filters, wavelength):
    """
    Integrate the product of the spectrum with each filter.
    """
    return np.trapz(spectrum[:, None] * filters, wavelength, axis=0)


# -------------------------------------------------------------
# Reconstruction of spectrum from photometry
# -------------------------------------------------------------
def reconstruct_spectrum(photometry, basis, filters, wavelength):
    """
    Solve for the basis coefficients that reproduce the photometry,
    then reconstruct the full spectrum.
    """
    # Forward model matrix: photometry = M * coeffs
    # where M_ij = ∫ basis_j(λ) * filter_i(λ) dλ
    M = np.trapz(basis[:, None, :] * filters[None, :, :], wavelength,
                 axis=0)  # shape: (n_filters, n_basis)

    # Solve least‑squares for coefficients
    coeffs, _, _, _ = np.linalg.lstsq(M.T, photometry, rcond=None)
    # Reconstruct spectrum
    recon_spec = basis @ coeffs
    return recon_spec, coeffs


# -------------------------------------------------------------
# Main routine
# -------------------------------------------------------------
def main():
    # Wavelength grid (400–800 nm)
    wavelength = np.linspace(400.0, 800.0, 1000)

    # Build basis functions
    basis = make_basis_functions(wavelength, n_basis=10, width=40.0)

    # Generate synthetic spectrum
    true_spectrum, true_weights = generate_synthetic_spectrum(basis, seed=42)

    # Define filters
    filters = make_filters(wavelength, n_filters=5, sigma=25.0)

    # Compute photometry from the true spectrum
    photometry = compute_photometry(true_spectrum, filters, wavelength)

    # Reconstruct the spectrum from photometry
    recon_spectrum, recon_weights = reconstruct_spectrum(
        photometry, basis, filters, wavelength
    )

    # Print results
    print("True weights:\n", true_weights)
    print("\nRecovered weights:\n", recon_weights)
    print("\nDifference in norms:",
          np.linalg.norm(true_weights - recon_weights))

    # Optional: compute reconstruction error in flux space
    flux_error = np.linalg.norm(true_spectrum - recon_spectrum) / np.linalg.norm(true_spectrum)
    print("\nRelative reconstruction error:", flux_error)


if __name__ == "__main__":
    main()