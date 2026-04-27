import numpy as np
from sklearn.linear_model import Ridge

# -----------------------------
# Spectral model definition
# -----------------------------
def create_spectral_basis(n_wavelengths=200, n_components=3, seed=0):
    """
    Generate a set of orthogonal basis spectra.
    Returns an array of shape (n_wavelengths, n_components).
    """
    rng = np.random.default_rng(seed)
    # Use sinusoids with different frequencies
    wavelengths = np.linspace(0, 1, n_wavelengths)
    basis = []
    for k in range(1, n_components + 1):
        basis.append(np.sin((k * np.pi) * wavelengths))
    return np.vstack(basis).T   # shape (n_wavelengths, n_components)

def generate_synthetic_spectra(basis, n_samples=10, noise_level=0.01, seed=1):
    """
    Generate synthetic spectra as random linear combinations of the basis.
    Adds small Gaussian noise.
    Returns an array of shape (n_samples, n_wavelengths).
    """
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(-1, 1, size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T  # shape (n_samples, n_wavelengths)
    spectra += rng.normal(scale=noise_level, size=spectra.shape)
    return spectra, coeffs

# -----------------------------
# Photometric system
# -----------------------------
def create_filter_transmissions(n_wavelengths=200, n_filters=5):
    """
    Create simple rectangular filter transmission curves.
    Returns an array of shape (n_filters, n_wavelengths).
    """
    transmissions = np.zeros((n_filters, n_wavelengths))
    width = n_wavelengths // n_filters
    for i in range(n_filters):
        start = i * width
        end = (i + 1) * width if i < n_filters - 1 else n_wavelengths
        transmissions[i, start:end] = 1.0
    return transmissions

def compute_photometry(spectra, transmissions):
    """
    Compute synthetic photometric measurements by integrating spectra over filter bands.
    Returns an array of shape (n_samples, n_filters).
    """
    # Simple dot product as integral approximation
    return spectra @ transmissions.T

# -----------------------------
# Reconstruction
# -----------------------------
def reconstruct_spectrum(photometry, transmissions, basis, alpha=1.0):
    """
    Reconstruct spectra from photometric data using Ridge regression.
    Returns reconstructed spectra of shape (n_samples, n_wavelengths).
    """
    # Build design matrix: photometry = coefficients @ (basis.T @ transmissions.T)
    M = basis.T @ transmissions.T  # shape (n_components, n_filters)
    # Solve for coefficients given photometry
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(M.T, photometry.T)  # fit per coefficient
    coeffs_recon = ridge.coef_.T   # shape (n_samples, n_components)
    # Reconstruct spectra
    spectra_recon = coeffs_recon @ basis.T
    return spectra_recon, coeffs_recon

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Setup
    n_wavelengths = 200
    n_components = 3
    n_samples = 10
    n_filters = 5

    # Generate basis spectra
    basis = create_spectral_basis(n_wavelengths, n_components)

    # Generate synthetic spectra and true coefficients
    spectra_true, coeffs_true = generate_synthetic_spectra(basis, n_samples)

    # Create filter transmissions
    transmissions = create_filter_transmissions(n_wavelengths, n_filters)

    # Compute photometric observations
    photometry = compute_photometry(spectra_true, transmissions)

    # Reconstruct spectra from photometry
    spectra_rec, coeffs_rec = reconstruct_spectrum(photometry, transmissions, basis)

    # Print results
    print("True coefficients (first sample):", coeffs_true[0])
    print("Reconstructed coefficients (first sample):", coeffs_rec[0])
    print("Spectral reconstruction error (L2 norm, first sample):",
          np.linalg.norm(spectra_true[0] - spectra_rec[0]))