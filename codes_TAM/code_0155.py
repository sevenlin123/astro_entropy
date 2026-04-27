import numpy as np
from numpy.linalg import lstsq
from scipy.interpolate import interp1d

# -------------------------------------------------------------
# 1. Spectral model definition
# -------------------------------------------------------------
def generate_basis_spectra(n_points: int, n_bases: int, seed: int = 0):
    """
    Create a set of orthogonal-ish basis spectra.
    Each basis is a smooth curve with random amplitude.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_points)
    basis = []
    for i in range(n_bases):
        # Smooth random curve
        coeffs = rng.normal(size=5)
        poly = np.polyval(coeffs, x)
        basis.append(poly + 0.1 * rng.standard_normal(n_points))
    return np.array(basis)  # shape (n_bases, n_points)

# -------------------------------------------------------------
# 2. Synthetic spectrum generation
# -------------------------------------------------------------
def synthesize_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return np.dot(coeffs, basis)  # shape (n_points,)

# -------------------------------------------------------------
# 3. Photometric data generation
# -------------------------------------------------------------
def integrate_over_filter(spectrum, wavelength, filt_resp):
    """Compute photon flux through a single filter."""
    # Assume filt_resp is transmission vs wavelength
    f_interp = interp1d(wavelength, filt_resp, bounds_error=False, fill_value=0)
    trans = f_interp(wavelength)
    return np.trapz(spectrum * trans, wavelength)

def generate_photometry(spectrum, wavelength, filters):
    """Return fluxes in all filters."""
    return np.array([integrate_over_filter(spectrum, wavelength, f)
                     for f in filters])

# -------------------------------------------------------------
# 4. Reconstruction from photometry
# -------------------------------------------------------------
def build_filter_matrix(basis, wavelength, filters):
    """Matrix A where A[j,i] = integral of basis_i * filter_j."""
    n_filters = len(filters)
    n_bases = basis.shape[0]
    A = np.zeros((n_filters, n_bases))
    for j, filt in enumerate(filters):
        for i in range(n_bases):
            A[j, i] = integrate_over_filter(basis[i], wavelength, filt)
    return A

def reconstruct_coefficients(A, photometry):
    """Least-squares solution for basis coefficients."""
    coeffs, *_ = lstsq(A, photometry, rcond=None)
    return coeffs

def reconstruct_spectrum(A, coeffs, basis):
    """Reconstruct spectrum from solved coefficients."""
    # Using original basis
    return synthesize_spectrum(basis, coeffs)

# -------------------------------------------------------------
# Main demonstration
# -------------------------------------------------------------
def main():
    # Wavelength grid (nm)
    wl = np.linspace(400, 800, 500)

    # Create basis spectra
    n_bases = 5
    basis = generate_basis_spectra(len(wl), n_bases, seed=42)

    # Define synthetic filters (Gaussian responses)
    def gaussian_filter(center, width):
        return np.exp(-0.5 * ((wl - center) / width) ** 2)

    filters = [
        gaussian_filter(450, 30),
        gaussian_filter(550, 30),
        gaussian_filter(650, 30),
        gaussian_filter(750, 30)
    ]

    # Build filter matrix once
    A = build_filter_matrix(basis, wl, filters)

    # Generate synthetic dataset
    n_samples = 10
    rng = np.random.default_rng(123)
    true_coeffs_list = rng.uniform(0.5, 1.5, size=(n_samples, n_bases))
    recon_errors = []

    for idx, true_coeffs in enumerate(true_coeffs_list):
        # Synthesize spectrum
        spec = synthesize_spectrum(basis, true_coeffs)

        # Generate photometric observations
        phot = generate_photometry(spec, wl, filters)

        # Reconstruct coefficients
        rec_coeffs = reconstruct_coefficients(A, phot)

        # Reconstruct spectrum
        rec_spec = synthesize_spectrum(basis, rec_coeffs)

        # Compute reconstruction error (relative L2 norm)
        err = np.linalg.norm(spec - rec_spec) / np.linalg.norm(spec)
        recon_errors.append(err)

        print(f"Sample {idx+1}:")
        print(f"  True coeffs   : {true_coeffs}")
        print(f"  Recovered coeffs: {rec_coeffs}")
        print(f"  Reconstruction relative error: {err:.4f}\n")

    print(f"Average reconstruction error: {np.mean(recon_errors):.4f}")

if __name__ == "__main__":
    main()