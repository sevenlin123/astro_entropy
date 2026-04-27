import numpy as np
from scipy.special import erf
from sklearn.linear_model import LinearRegression

# ---------------------------------------------
# Spectral model definition
# ---------------------------------------------
def gaussian_basis(wavelength, centers, widths):
    """Return matrix of gaussian basis functions."""
    return np.exp(-0.5 * ((wavelength[:, None] - centers[None, :]) / widths[None, :])**2)

def create_basis(num_bases, lam_min, lam_max, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    centers = rng.uniform(lam_min, lam_max, num_bases)
    widths = rng.uniform(20, 50, num_bases)
    lam = np.linspace(lam_min, lam_max, 1000)
    return lam, gaussian_basis(lam, centers, widths)

# ---------------------------------------------
# Synthetic spectra generation
# ---------------------------------------------
def generate_synthetic_spectra(basis, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    weights = rng.uniform(-1, 1, basis.shape[1])
    spectrum = basis @ weights
    return spectrum, weights

# ---------------------------------------------
# Filter curves definition
# ---------------------------------------------
def gaussian_filter_curve(wavelength, center, width):
    return np.exp(-0.5 * ((wavelength - center) / width)**2)

def create_filters(num_filters, lam_min, lam_max, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    centers = rng.uniform(lam_min + 50, lam_max - 50, num_filters)
    widths = rng.uniform(30, 80, num_filters)
    filt_funcs = []
    for c, w in zip(centers, widths):
        filt_funcs.append(lambda lam, c=c, w=w: gaussian_filter_curve(lam, c, w))
    return filt_funcs, centers, widths

# ---------------------------------------------
# Photometry computation
# ---------------------------------------------
def compute_photometry(spectrum, wavelength, filt_funcs):
    # integrate spectrum * filter over wavelength
    phots = []
    for f in filt_funcs:
        trans = f(wavelength)
        phots.append(np.trapz(spectrum * trans, wavelength))
    return np.array(phots)

# ---------------------------------------------
# Reconstruction from photometry
# ---------------------------------------------
def reconstruct_spectrum(filters, basis, wavelength, photometry):
    # Build response matrix R_{i,j} = ∫ basis_j * filter_i dλ
    R = np.empty((len(filters), basis.shape[1]))
    for i, f in enumerate(filters):
        trans = f(wavelength)
        for j in range(basis.shape[1]):
            R[i, j] = np.trapz(basis[:, j] * trans, wavelength)
    # Solve linear system R * w = photometry
    reg = LinearRegression(fit_intercept=False)
    reg.fit(R, photometry)
    weights = reg.coef_
    reconstructed = basis @ weights
    return reconstructed, weights

# ---------------------------------------------
# Main demo
# ---------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    lam_min, lam_max = 400, 800  # nm

    # Basis and synthetic spectrum
    lam, basis = create_basis(num_bases=5, lam_min=lam_min, lam_max=lam_max, rng=rng)
    true_spectrum, true_weights = generate_synthetic_spectra(basis, rng=rng)

    # Filters and photometry
    filt_funcs, filt_centers, filt_widths = create_filters(num_filters=3,
                                                           lam_min=lam_min,
                                                           lam_max=lam_max,
                                                           rng=rng)
    photometry = compute_photometry(true_spectrum, lam, filt_funcs)

    # Reconstruction
    recon_spectrum, recon_weights = reconstruct_spectrum(filt_funcs, basis, lam, photometry)

    # Print results
    print("True weights:", true_weights)
    print("Reconstructed weights:", recon_weights)
    print("Weight error (L2 norm):", np.linalg.norm(true_weights - recon_weights))
    print("Spectrum reconstruction error (L2 norm):",
          np.linalg.norm(true_spectrum - recon_spectrum))