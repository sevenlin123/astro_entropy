import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# 1. Define spectral model: set of Gaussian basis functions
# ----------------------------------------------------------------------
def create_gaussian_bases(wavelength, n_bases=10, sigma=30.0):
    """
    Return an array of shape (len(wavelength), n_bases) containing Gaussian
    basis functions centered at evenly spaced wavelengths.
    """
    centers = np.linspace(wavelength.min() + sigma,
                          wavelength.max() - sigma,
                          n_bases)
    bases = np.array([norm.pdf(wavelength, loc=c, scale=sigma)
                      for c in centers]).T
    # Normalize each basis function
    bases /= bases.sum(axis=0)
    return bases

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, bases):
    """
    Generate synthetic spectra as linear combinations of the basis functions.
    """
    coeffs = np.random.rand(n_samples, bases.shape[1])
    spectra = coeffs @ bases.T
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Define filter responses (simple top‑hat bands)
# ----------------------------------------------------------------------
def create_filters(wavelength):
    """
    Create a dictionary of three filter response curves (top‑hat).
    """
    filt_defs = {
        'U': (300, 400),
        'B': (400, 500),
        'V': (500, 600)
    }
    filters = {}
    for name, (lam_min, lam_max) in filt_defs.items():
        resp = np.where((wavelength >= lam_min) & (wavelength <= lam_max), 1.0, 0.0)
        filters[name] = resp
    return filters

# ----------------------------------------------------------------------
# 4. Generate photometric fluxes from spectra
# ----------------------------------------------------------------------
def compute_photometry(spectra, wavelength, filters):
    """
    Integrate each spectrum over each filter to obtain photometric fluxes.
    Returns an array of shape (n_samples, n_filters).
    """
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    phots = np.empty((n_samples, n_filters))
    for i, (fname, resp) in enumerate(filters.items()):
        # Simple trapezoidal integration
        integrand = spectra * resp[None, :]
        phots[:, i] = np.trapz(integrand, x=wavelength, axis=1)
    return phots

# ----------------------------------------------------------------------
# 5. Reconstruct spectrum from photometry
# ----------------------------------------------------------------------
def reconstruct_spectra_from_photometry(photometry, wavelength, filters, bases):
    """
    Estimate basis coefficients that best reproduce the photometry.
    Uses ridge regression for stability.
    Returns reconstructed spectra.
    """
    # Build design matrix: each filter response integrated against each basis
    n_filters = len(filters)
    n_bases = bases.shape[1]
    design = np.empty((n_filters, n_bases))
    for i, resp in enumerate(filters.values()):
        design[i, :] = np.trapz(bases * resp[None, :], x=wavelength, axis=1)

    # Fit ridge regression: photometry = design @ coeffs^T
    reg = Ridge(alpha=1e-3, fit_intercept=False)
    reg.fit(design, photometry.T)   # regress each band on the same coeff vector
    coeffs_rec = reg.coef_.T         # shape (n_samples, n_bases)

    # Reconstruct spectra
    spectra_rec = coeffs_rec @ bases.T
    return spectra_rec, coeffs_rec

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (in nm)
    wav = np.linspace(250, 750, 1000)

    # Create basis functions
    bases = create_gaussian_bases(wav, n_bases=12, sigma=40.0)

    # Generate synthetic spectra
    n_samples = 5
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, bases)

    # Create filter set
    filt_dict = create_filters(wav)

    # Compute photometric fluxes
    phot = compute_photometry(spectra_true, wav, filt_dict)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra_from_photometry(
        phot, wav, filt_dict, bases
    )

    # Print results for first sample
    print("True coefficients (first sample):")
    print(coeffs_true[0])
    print("\nRecovered coefficients (first sample):")
    print(coeffs_rec[0])

    # Compare spectra shapes
    print("\nSpectra shape:", spectra_true.shape)
    print("Reconstructed spectra shape:", spectra_rec.shape)