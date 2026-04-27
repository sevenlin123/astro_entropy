import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

# ----------------------------------------------------------------------
# Spectral model: sum of three basis spectra
# ----------------------------------------------------------------------
def create_basis_spectra(wavelength):
    """Return three simple basis spectra (flux units arbitrary)."""
    # Blackbody-like shapes with different characteristic widths
    s1 = np.exp(-((wavelength - 4000.)**2) / (2 * 800**2))
    s2 = np.exp(-((wavelength - 6000.)**2) / (2 * 1200**2))
    s3 = np.exp(-((wavelength - 8000.)**2) / (2 * 1600**2))
    return np.vstack([s1, s2, s3])  # shape (3, N)

# ----------------------------------------------------------------------
# Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectrum(basis, coeffs):
    """Combine basis spectra with given coefficients."""
    return np.dot(coeffs, basis)

def generate_random_coefficients(n_samples, n_components, rng=np.random.default_rng()):
    return rng.uniform(0.5, 1.5, size=(n_samples, n_components))

# ----------------------------------------------------------------------
# Photometric filter definitions
# ----------------------------------------------------------------------
def gaussian_filter(wavelength, center, width):
    return np.exp(-0.5 * ((wavelength - center) / width)**2)

def create_filter_set(wavelength):
    """Return dictionary of filter transmissions."""
    filters = {
        'U': gaussian_filter(wavelength, 3650., 300.),
        'B': gaussian_filter(wavelength, 4450., 350.),
        'V': gaussian_filter(wavelength, 5510., 400.),
        'R': gaussian_filter(wavelength, 6580., 450.),
        'I': gaussian_filter(wavelength, 8060., 500.)
    }
    return filters

# ----------------------------------------------------------------------
# Compute photometry from a spectrum
# ----------------------------------------------------------------------
def compute_photometry(spectrum, wavelength, filters):
    """Integrate spectrum times each filter transmission."""
    fluxes = {}
    for name, filt in filters.items():
        integrand = spectrum * filt
        fluxes[name] = simps(integrand, wavelength)
    return np.array([fluxes[f] for f in sorted(filters)])

# ----------------------------------------------------------------------
# Reconstruction from photometry
# ----------------------------------------------------------------------
def build_response_matrix(wavelength, filters, basis):
    """
    Build matrix M such that photometric fluxes = M @ coeffs,
    where M_ij = ∫ basis_i(λ) * filter_j(λ) dλ
    """
    M = np.empty((len(filters), basis.shape[0]))
    filt_names = sorted(filters)
    for j, name in enumerate(filt_names):
        filt = filters[name]
        for i in range(basis.shape[0]):
            integrand = basis[i] * filt
            M[j, i] = simps(integrand, wavelength)
    return M

def reconstruct_spectrum_from_photometry(photometry, M):
    """Least-squares solve for coefficients, then reconstruct spectrum."""
    coeffs, *_ = np.linalg.lstsq(M.T, photometry, rcond=None)
    return coeffs

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid
    wav = np.linspace(3000, 10000, 2000)  # Angstrom

    # Basis spectra
    basis = create_basis_spectra(wav)

    # Filters
    filters = create_filter_set(wav)

    # Build response matrix
    M = build_response_matrix(wav, filters, basis)

    # Synthetic dataset
    n_objects = 10
    coeffs_true = generate_random_coefficients(n_objects, basis.shape[0], rng=rng)

    spectra = np.array([generate_synthetic_spectrum(basis, c) for c in coeffs_true])
    photometry = np.array([compute_photometry(sp, wav, filters) for sp in spectra])

    # Reconstruct coefficients from photometry
    coeffs_rec = np.array([reconstruct_spectrum_from_photometry(p, M) for p in photometry])

    # Reconstruct spectra
    spectra_rec = np.array([generate_synthetic_spectrum(basis, c) for c in coeffs_rec])

    # Simple diagnostics
    print("True vs reconstructed coefficients (first object):")
    print("True : ", coeffs_true[0])
    print("Reconstructed: ", coeffs_rec[0])

    print("\nDifference between true and reconstructed spectra (norm) for first object:")
    diff_norm = np.linalg.norm(spectra[0] - spectra_rec[0])
    print(diff_norm)