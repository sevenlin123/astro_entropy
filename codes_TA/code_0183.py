import numpy as np
from scipy.special import eval_legendre
from scipy.integrate import simps

# -------------------------------------------------------------
# Spectral model: linear combination of Legendre polynomials
# -------------------------------------------------------------
def legendre_basis(lam, n_basis):
    """Return matrix of Legendre polynomials evaluated at `lam`."""
    # normalize wavelength to [-1, 1]
    lam_norm = 2 * (lam - lam.min()) / (lam.max() - lam.min()) - 1
    basis = np.array([eval_legendre(k, lam_norm) for k in range(n_basis)]).T
    return basis

# -------------------------------------------------------------
# Synthetic data generation
# -------------------------------------------------------------
def generate_synthetic_spectra(n_samples, n_basis, lam, rng=None):
    """Generate synthetic spectra as linear combos of basis functions."""
    rng = np.random.default_rng(rng)
    coeffs = rng.normal(size=(n_samples, n_basis))
    basis = legendre_basis(lam, n_basis)
    spectra = coeffs @ basis.T
    return spectra, coeffs

def gaussian_filter(lam, center, width):
    """Return Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((lam - center) / width)**2)

def generate_filters(filter_specs, lam):
    """Return array of filter transmission curves."""
    filters = np.array([gaussian_filter(lam, c, w) for c, w in filter_specs])
    return filters

def photometry_from_spectra(spectra, filters):
    """Compute photometric fluxes by integrating spectra × filter."""
    # Multiply spectra (samples × lam) with filter (filters × lam)
    integrands = spectra[:, None, :] * filters[None, :, :]
    fluxes = simps(integrands, lam, axis=2)
    return fluxes

def add_noise(fluxes, noise_std=0.01, rng=None):
    rng = np.random.default_rng(rng)
    noisy_fluxes = fluxes + rng.normal(scale=noise_std, size=fluxes.shape)
    return noisy_fluxes

# -------------------------------------------------------------
# Reconstruction framework
# -------------------------------------------------------------
def build_design_matrix(filters, lam, n_basis):
    """Build matrix that maps basis coefficients to photometric fluxes."""
    basis = legendre_basis(lam, n_basis)          # (lam, n_basis)
    # integrate basis × filter over λ
    design = np.array([simps(basis * f[:, None], lam, axis=0) for f in filters])
    # design shape: (n_filters, n_basis)
    return design

def reconstruct_coefficients(photometry, design_matrix):
    """Solve least‑squares problem to recover coefficients."""
    coeffs, *_ = np.linalg.lstsq(design_matrix.T, photometry.T, rcond=None)
    return coeffs.T  # back to (n_samples, n_basis)

def reconstruct_spectra(coeffs, lam, n_basis):
    """Reconstruct spectra from coefficients and basis."""
    basis = legendre_basis(lam, n_basis)      # (lam, n_basis)
    spectra = coeffs @ basis.T               # (n_samples, lam)
    return spectra

# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
def main():
    rng = 42
    lam = np.linspace(400, 900, 501)          # wavelength grid (nm)
    n_basis = 5
    n_samples = 10

    # Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_samples, n_basis, lam, rng=rng)

    # Define filter specifications: (center_nm, width_nm)
    filter_specs = [(350, 50), (440, 60), (550, 70),
                    (660, 80), (770, 90)]
    filters = generate_filters(filter_specs, lam)

    # Compute noiseless photometry
    fluxes = photometry_from_spectra(spectra_true, filters)

    # Add Gaussian noise
    noise_std = 0.02
    fluxes_noisy = add_noise(fluxes, noise_std=noise_std, rng=rng)

    # Build design matrix and reconstruct
    design = build_design_matrix(filters, lam, n_basis)
    coeffs_rec = reconstruct_coefficients(fluxes_noisy, design)
    spectra_rec = reconstruct_spectra(coeffs_rec, lam, n_basis)

    # Evaluate reconstruction error
    rmse = np.sqrt(((spectra_true - spectra_rec)**2).mean(axis=1))
    print("RMSE per spectrum:", rmse)
    print("Mean RMSE:", rmse.mean())

if __name__ == "__main__":
    main()