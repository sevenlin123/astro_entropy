import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------- spectral model ------------------------------------
def define_basis(n_wav=1000, n_basis=3):
    """
    Create a set of orthonormal basis spectra.
    Returns:
        wav: array of wavelengths
        basis: (n_basis, n_wav) matrix of basis functions
    """
    wav = np.linspace(400, 700, n_wav)  # nm
    basis = np.zeros((n_basis, n_wav))
    for i in range(n_basis):
        # Gaussian bumps at different central wavelengths
        center = 400 + (i + 1) * 100
        sigma = 30
        basis[i] = np.exp(-0.5 * ((wav - center) / sigma) ** 2)
    # Orthonormalize
    u, _, v = np.linalg.svd(basis.T, full_matrices=False)
    basis = u.T
    return wav, basis

# -------------------- synthetic spectra generation ----------------------
def generate_synthetic_spectra(basis, n_spectra=10, noise_level=0.01, random_state=None):
    """
    Generate synthetic spectra as random linear combinations of basis functions.
    """
    rng = np.random.default_rng(random_state)
    coeffs = rng.uniform(0.5, 1.5, size=(n_spectra, basis.shape[0]))
    spectra = coeffs @ basis
    # Add noise
    spectra += noise_level * rng.standard_normal(spectra.shape)
    return coeffs, spectra

# -------------------- photometry computation ----------------------------
def define_filters(wav, n_filters=4):
    """
    Define simple top-hat filter transmission curves.
    """
    filters = np.zeros((n_filters, len(wav)))
    band_edges = np.linspace(wav[0], wav[-1], n_filters + 1)
    for i in range(n_filters):
        filt = (wav >= band_edges[i]) & (wav < band_edges[i+1])
        filters[i, filt] = 1.0
    return filters

def compute_photometry(spectra, filters):
    """
    Integrate spectra over each filter to get photometric fluxes.
    """
    return spectra @ filters.T

# -------------------- reconstruction ------------------------------------
def reconstruct_spectrum(phots, filters, basis):
    """
    Reconstruct spectrum by regressing to recover coefficients.
    Returns reconstructed spectra and predicted coefficients.
    """
    # Flatten filters for regression
    X = filters.T  # shape (n_wav, n_filters)
    y = phots.T    # shape (n_filters, n_samples)
    model = LinearRegression()
    model.fit(X, y)
    coeffs_rec = model.coef_.T   # shape (n_basis, n_samples)
    spectra_rec = coeffs_rec @ basis
    return coeffs_rec, spectra_rec

# -------------------- demo ------------------------------------------------
if __name__ == "__main__":
    wav, basis = define_basis()
    coeffs_true, spectra_true = generate_synthetic_spectra(basis, random_state=42)
    filters = define_filters(wav)
    phots = compute_photometry(spectra_true, filters)

    coeffs_rec, spectra_rec = reconstruct_spectrum(phots, filters, basis)

    # Simple sanity check: compare true vs recovered coefficients
    print("True coefficients:\n", coeffs_true[:5])
    print("\nReconstructed coefficients (first 5 samples):\n", coeffs_rec[:, :5])

    # Compare spectra (mean squared error)
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"\nMean squared error between true and reconstructed spectra: {mse:.4f}")