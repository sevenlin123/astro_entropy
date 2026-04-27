import numpy as np
from scipy.integrate import simps

# -----------------------------
# Define spectral basis functions
# -----------------------------
def gaussian(x, center, width):
    """One–dimensional Gaussian."""
    return np.exp(-0.5 * ((x - center) / width)**2)

def make_basis_functions(wl, centers, widths):
    """Return an array of basis functions evaluated on wl."""
    return np.array([gaussian(wl, c, w) for c, w in zip(centers, widths)])

# -----------------------------
# Define filter responses
# -----------------------------
def filter_response(wl, center, width):
    """Simple Gaussian filter transmission curve."""
    return gaussian(wl, center, width)

def make_filters(wl):
    """Return a list of filter transmission curves."""
    # Example: three broad-band filters
    filt_centers = [5200, 6200, 7200]   # nm
    filt_widths  = [300, 300, 300]      # nm
    return [filter_response(wl, c, w) for c, w in zip(filt_centers, filt_widths)]

# -----------------------------
# Synthetic spectrum generation
# -----------------------------
def generate_random_coeffs(n_basis, rng=None):
    """Draw random coefficients for the basis functions."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-1, 1, size=n_basis)

def generate_synthetic_spectrum(coeffs, basis_funcs):
    """Combine basis functions linearly."""
    return coeffs @ basis_funcs

# -----------------------------
# Photometry calculation
# -----------------------------
def compute_photometry(spectrum, wl, filters):
    """
    Compute fluxes in each filter:
    F_i = ∫ S(λ) T_i(λ) dλ / ∫ T_i(λ) dλ
    """
    fluxes = []
    for filt in filters:
        num = simps(spectrum * filt, wl)
        den = simps(filt, wl)
        fluxes.append(num / den)
    return np.array(fluxes)

# -----------------------------
# Reconstruction
# -----------------------------
def build_response_matrix(basis_funcs, wl, filters):
    """
    Build matrix A where A_ij = ∫ basis_j(λ) T_i(λ) dλ / ∫ T_i(λ) dλ
    """
    A = []
    for filt in filters:
        row = []
        den = simps(filt, wl)
        for bf in basis_funcs:
            num = simps(bf * filt, wl)
            row.append(num / den)
        A.append(row)
    return np.array(A)

def reconstruct_coeffs(photometry, response_matrix):
    """Least–squares solution for coefficients."""
    coeffs, *_ = np.linalg.lstsq(response_matrix, photometry, rcond=None)
    return coeffs

def reconstruct_spectrum(coeffs, basis_funcs):
    """Reconstruct full spectrum from recovered coefficients."""
    return coeffs @ basis_funcs

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Wavelength grid
    wl = np.linspace(4000, 8000, 1000)   # nm

    # Basis functions
    centers = [5000, 6000, 7000]
    widths  = [200, 200, 200]
    basis_funcs = make_basis_functions(wl, centers, widths)

    # Filters
    filters = make_filters(wl)

    # True coefficients and spectrum
    true_coeffs = generate_random_coeffs(len(centers), rng=rng)
    true_spectrum = generate_synthetic_spectrum(true_coeffs, basis_funcs)

    # Photometric measurements
    photometry = compute_photometry(true_spectrum, wl, filters)

    # Build response matrix and reconstruct
    R = build_response_matrix(basis_funcs, wl, filters)
    rec_coeffs = reconstruct_coeffs(photometry, R)
    rec_spectrum = reconstruct_spectrum(rec_coeffs, basis_funcs)

    # Display results
    print("True coefficients :", true_coeffs)
    print("Recovered coeffs :", rec_coeffs)
    print("\nSpectral comparison (first 10 points):")
    print("Wavelength (nm) | True flux | Reconstructed flux")
    for i in range(10):
        print(f"{wl[i]:8.1f} | {true_spectrum[i]:10.4f} | {rec_spectrum[i]:20.4f}")