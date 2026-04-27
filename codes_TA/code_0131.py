import numpy as np
from scipy.constants import speed_of_light
from numpy.linalg import lstsq

def generate_wavelength_grid(start=4000, stop=8000, n_points=1000):
    """Generate a uniform wavelength grid in Angstroms."""
    return np.linspace(start, stop, n_points)

def gaussian(x, amp, cen, width):
    """One‑dimensional Gaussian function."""
    return amp * np.exp(-0.5 * ((x - cen) / width)**2)

def create_basis(wave, n_basis=10):
    """
    Create a set of Gaussian basis functions spanning the wavelength grid.
    Returns an array of shape (len(wave), n_basis).
    """
    centers = np.linspace(wave[0], wave[-1], n_basis)
    width = (wave[-1] - wave[0]) / (n_basis * 4)
    basis = np.vstack([gaussian(wave, 1.0, c, width) for c in centers]).T
    return basis

def generate_synthetic_spectrum(basis, n_real=5):
    """
    Build a synthetic spectrum as a random linear combination of basis functions.
    Returns the spectrum and the used amplitudes.
    """
    amps = np.random.rand(n_real) * 5.0
    chosen = np.zeros(basis.shape[1])
    chosen[:n_real] = amps
    spectrum = basis @ chosen
    return spectrum, chosen

def create_filters(wave, n_filters=3):
    """
    Construct simple Gaussian filters.
    Returns a matrix of shape (len(wave), n_filters).
    """
    centers = np.linspace(4500, 6500, n_filters)
    width = 300.0
    filters = np.vstack([gaussian(wave, 1.0, c, width) for c in centers]).T
    return filters

def compute_photometry(spectrum, filters):
    """
    Integrate the product of spectrum and filter transmissions.
    Assumes uniform wavelength sampling.
    Returns photometric fluxes for each filter.
    """
    dw = spectrum.size  # placeholder for uniform spacing; will be multiplied later
    # Actually compute integral as sum over wavelength grid
    integrals = (filters * spectrum[:, None]).sum(axis=0)  # shape (n_filters,)
    return integrals

def reconstruct_spectrum(filters, basis, photometry):
    """
    Estimate the spectrum by solving for basis amplitudes that reproduce the
    observed photometry. Uses linear least squares.
    """
    # Build design matrix: each entry = integral of basis_i * filter_j
    dw = 1.0  # relative unit; uniform spacing cancels out
    design = (filters[:, :, None] * basis.T[None, :, :]).sum(axis=1) * dw
    # design shape: (n_filters, n_basis)
    coeffs, *_ = lstsq(design, photometry, rcond=None)
    reconstructed = basis @ coeffs
    return reconstructed, coeffs

def main():
    wave = generate_wavelength_grid()
    basis = create_basis(wave)
    spectrum_true, true_amps = generate_synthetic_spectrum(basis)
    filters = create_filters(wave)
    photometry = compute_photometry(spectrum_true, filters)
    spectrum_rec, rec_amps = reconstruct_spectrum(filters, basis, photometry)

    print("True amplitudes:", true_amps)
    print("Reconstructed amplitudes:", rec_amps[:len(true_amps)])
    # Compare spectra (optional, no plotting)
    error = np.linalg.norm(spectrum_true - spectrum_rec) / np.linalg.norm(spectrum_true)
    print(f"Relative reconstruction error: {error:.3f}")

if __name__ == "__main__":
    main()