import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# -----------------------------------
# Parameters
# -----------------------------------
N_WAVEL = 1000        # number of wavelength points
WAVE_MIN, WAVE_MAX = 300, 2500   # nm
N_BASIS = 5           # number of basis spectra
N_SAMPLES = 50        # number of synthetic objects
N_FILTERS = 6         # number of photometric bands
RHO = 10.0            # regularisation strength

# -----------------------------------
# Helper functions
# -----------------------------------
def create_wavelength_grid(n=N_WAVEL, wmin=WAVE_MIN, wmax=WAVE_MAX):
    """Create a uniform wavelength grid."""
    return np.linspace(wmin, wmax, n)

def generate_basis_spectra(grid, n_basis=N_BASIS, seed=0):
    """Generate synthetic basis spectra with Gaussian features."""
    rng = np.random.default_rng(seed)
    basis = []
    for i in range(n_basis):
        amp = rng.uniform(0.5, 1.5, size=3)
        cen = rng.uniform(400, 2400, size=3)
        wid = rng.uniform(50, 200, size=3)
        spec = np.zeros_like(grid)
        for a, c, w in zip(amp, cen, wid):
            spec += a * np.exp(-0.5 * ((grid - c)/w)**2)
        basis.append(spec)
    return np.vstack(basis)      # shape (n_basis, len(grid))

def generate_synthetic_spectra(basis, n_samples=N_SAMPLES, seed=1):
    """Generate synthetic spectra as random linear combos of basis."""
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis          # shape (n_samples, len(grid))
    return spectra, coeffs

def gaussian_filter(grid, center, width):
    """Return a simple Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((grid - center) / width)**2)

def create_filters(grid, n_filters=N_FILTERS, seed=2):
    """Generate a set of Gaussian filter responses."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(WAVE_MIN + 50, WAVE_MAX - 50, size=n_filters)
    widths  = rng.uniform(30, 80, size=n_filters)
    filters = np.array([gaussian_filter(grid, c, w) for c, w in zip(centers, widths)])
    return filters, centers, widths

def compute_photometry(spectra, filters):
    """Integrate spectra through filter responses."""
    # spectra: (n_samples, n_wave)
    # filters : (n_filters, n_wave)
    phots = np.zeros((spectra.shape[0], filters.shape[0]))
    for i in range(filters.shape[0]):
        trans = filters[i]
        # Simple numerical integration
        phots[:, i] = simps(spectra * trans, axis=1) / simps(trans, axis=0)
    return phots

def fit_reconstruction(phots, coeffs, alpha=RHO):
    """Fit ridge regression to map photometry to basis coefficients."""
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(phots, coeffs)
    return reg

def reconstruct_spectrum(reg, phots, basis):
    """Reconstruct spectra from photometry."""
    coeff_est = reg.predict(phots)
    recon = coeff_est @ basis
    return recon, coeff_est

# -----------------------------------
# Main routine
# -----------------------------------
def main():
    # Step 1: Create wavelength grid and basis spectra
    grid = create_wavelength_grid()
    basis = generate_basis_spectra(grid)

    # Step 2: Generate synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(basis)

    # Step 3: Build filter set and compute photometry
    filters, centers, widths = create_filters(grid)
    photometry = compute_photometry(spectra, filters)

    # Step 4: Fit reconstruction model
    reg = fit_reconstruction(photometry, true_coeffs)

    # Step 5: Reconstruct spectra
    recon_spectra, est_coeffs = reconstruct_spectrum(reg, photometry, basis)

    # -----------------------------------
    # Evaluation
    # -----------------------------------
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared error of reconstruction: {mse:.3e}")

    # Plot one example
    idx = 0
    plt.figure(figsize=(10,4))
    plt.plot(grid, spectra[idx], label='True Spectrum')
    plt.plot(grid, recon_spectra[idx], '--', label='Reconstructed')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux (arb. units)')
    plt.title('Spectrum Reconstruction Example')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()