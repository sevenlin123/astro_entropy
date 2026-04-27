import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass

# ------------------------------
# Spectral model utilities
# ------------------------------

@dataclass
class SpectrumModel:
    wavelength: np.ndarray            # 1D array of wavelength values [Å]
    basis: np.ndarray                # 2D array (n_basis, n_wavelength)

def gaussian(x, amp, cen, wid):
    """One–dimensional Gaussian."""
    return amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)

def create_basis(n_basis, wave, rng=None):
    """Generate a set of Gaussian basis spectra."""
    rng = rng or np.random.default_rng()
    basis = np.zeros((n_basis, len(wave)))
    for i in range(n_basis):
        amp  = rng.uniform(0.5, 1.5)
        cen  = rng.uniform(wave.min() + 200, wave.max() - 200)
        wid  = rng.uniform(50, 150)
        basis[i] = gaussian(wave, amp, cen, wid)
    return basis

def synthesize_spectra(n_obj, model, coeff_range, rng=None):
    """Create synthetic spectra as linear combinations of the basis."""
    rng = rng or np.random.default_rng()
    coeffs = rng.uniform(coeff_range[0], coeff_range[1], size=(n_obj, model.basis.shape[0]))
    spectra = coeffs @ model.basis
    return spectra, coeffs

# ------------------------------
# Filter utilities
# ------------------------------

def top_hat_filter(wave, cen, width):
    """Simple top–hat filter centered at cen with given width."""
    return np.where(np.abs(wave - cen) <= width / 2, 1.0, 0.0)

def create_filters(filter_specs, wave):
    """Generate filter response curves."""
    filters = []
    for cen, width in filter_specs:
        filters.append(top_hat_filter(wave, cen, width))
    return np.array(filters)  # shape (n_filters, n_wave)

# ------------------------------
# Photometry computation
# ------------------------------

def photometry_from_spectrum(spectrum, filters, wave):
    """Integrate spectrum through each filter."""
    return np.trapz(spectrum[:, None] * filters, wave, axis=1).T

def add_noise(data, sigma):
    """Add Gaussian noise to the photometry."""
    return data + np.random.normal(scale=sigma, size=data.shape)

# ------------------------------
# Reconstruction
# ------------------------------

def reconstruct_coeffs(phot, filters, basis, wave):
    """
    Reconstruct basis coefficients from photometric measurements.
    The relationship is linear: phot = M @ coeffs,
    where M_ij = integral(basis_j * filter_i).
    """
    # Build design matrix M
    M = np.trapz(basis.T[:, None] * filters, wave, axis=2)  # shape (n_filters, n_basis)
    # Fit using ordinary least squares
    reg = LinearRegression(fit_intercept=False)
    reg.fit(M, phot)
    return reg.coef_.reshape(-1, 1).T  # shape (1, n_basis)

# ------------------------------
# Main demonstration
# ------------------------------

def main():
    rng = np.random.default_rng(42)

    # Wavelength grid
    wave = np.linspace(4000, 8000, 400)  # 4000–8000 Å

    # Spectral model
    n_basis = 5
    basis = create_basis(n_basis, wave, rng=rng)
    model = SpectrumModel(wavelength=wave, basis=basis)

    # Synthetic spectra
    n_objs = 10
    coeff_range = (0.0, 2.0)
    spectra, true_coeffs = synthesize_spectra(n_objs, model, coeff_range, rng=rng)

    # Filters (e.g., g, r, i, z)
    filter_specs = [(4800, 600), (6200, 600), (7500, 600), (8800, 600)]
    filters = create_filters(filter_specs, wave)

    # Compute photometry
    phot = photometry_from_spectrum(spectra, filters, wave)
    # Add measurement noise
    phot_noisy = add_noise(phot, sigma=0.02)

    # Reconstruct coefficients for each object
    rec_coeffs_list = []
    for p in phot_noisy:
        rec = reconstruct_coeffs(p, filters, basis, wave)
        rec_coeffs_list.append(rec.flatten())
    rec_coeffs = np.vstack(rec_coeffs_list)

    # Evaluate reconstruction quality
    rmse = np.sqrt(((rec_coeffs - true_coeffs) ** 2).mean(axis=0))
    print("True coefficients (first object):", true_coeffs[0])
    print("Reconstructed coefficients (first object):", rec_coeffs[0])
    print("RMSE per basis element:", rmse)

if __name__ == "__main__":
    main()