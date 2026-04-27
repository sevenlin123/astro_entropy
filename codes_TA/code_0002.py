import numpy as np
from scipy.stats import norm

# ------------------ Spectral Model ------------------
def gaussian_basis(wl, center, width):
    """Return a Gaussian basis function evaluated on wavelengths wl."""
    return norm.pdf(wl, loc=center, scale=width)

def build_basis_functions(n_basis, wl):
    """Construct n_basis Gaussian basis functions across wl."""
    centers = np.linspace(450, 750, n_basis)
    widths = np.full(n_basis, 20.0)   # fixed width
    B = np.vstack([gaussian_basis(wl, c, w) for c, w in zip(centers, widths)])
    return B.T   # shape (n_lambda, n_basis)

# ------------------ Filter Definitions ------------------
def filter_gaussian(wl, center, width):
    """Filter transmission profile as a Gaussian."""
    return norm.pdf(wl, loc=center, scale=width)

def build_filter_matrix(wl, centers, widths):
    """Build filter matrix F (n_filters, n_lambda)."""
    return np.vstack([filter_gaussian(wl, c, w) for c, w in zip(centers, widths)])

# ------------------ Synthetic Data Generation ------------------
def generate_synthetic_spectra(n_stars, B, rng=None):
    """Generate synthetic spectra with random coefficients."""
    rng = np.random.default_rng(rng)
    coeffs_true = rng.normal(scale=1.0, size=(n_stars, B.shape[1]))
    spectra = coeffs_true @ B.T   # shape (n_stars, n_lambda)
    return coeffs_true, spectra

def compute_photometry(spectra, F, rng=None):
    """Compute photometric fluxes from spectra via filter matrix."""
    rng = np.random.default_rng(rng)
    raw = spectra @ F.T            # shape (n_stars, n_filters)
    norm = F.sum(axis=1)           # normalization per filter
    photometry = raw / norm        # simple average transmission
    noise = rng.normal(scale=0.01, size=photometry.shape)
    return photometry + noise

# ------------------ Reconstruction ------------------
def reconstruct_coefficients(photometry, G):
    """Least-squares reconstruction of coefficients from photometry."""
    # Solve G * coeffs = photometry.T  -> coeffs.T = G.pinv() @ photometry.T
    coeffs_hat = np.linalg.lstsq(G, photometry.T, rcond=None)[0].T
    return coeffs_hat

def reconstruct_spectra(coeffs_hat, B):
    """Reconstruct spectra from estimated coefficients."""
    return coeffs_hat @ B.T

# ------------------ Main Routine ------------------
def main():
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)

    # Wavelength grid
    wl = np.linspace(400, 800, 801)   # 1 Å resolution

    # Basis functions
    n_basis = 10
    B = build_basis_functions(n_basis, wl)

    # Filters (UBVRI approximation)
    filter_centers = [360, 440, 550, 640, 790]   # U,B,V,R,I in nm
    filter_widths  = [30, 40, 50, 40, 30]
    F = build_filter_matrix(wl, filter_centers, filter_widths)

    # Forward matrix G
    G = F @ B  # shape (n_filters, n_basis)

    # Generate synthetic data
    n_stars = 100
    coeffs_true, spectra_true = generate_synthetic_spectra(n_stars, B, rng=rng_seed)
    photometry = compute_photometry(spectra_true, F, rng=rng_seed)

    # Reconstruction
    coeffs_est = reconstruct_coefficients(photometry, G)
    spectra_est = reconstruct_spectra(coeffs_est, B)

    # Evaluation
    mse = ((spectra_true - spectra_est)**2).mean(axis=1)
    print(f"Mean MSE over {n_stars} stars: {mse.mean():.4f}")

    # Example plot for one star (optional)
    # Uncomment the following block if you wish to visualize:
    #
    # import matplotlib.pyplot as plt
    # i = 0
    # plt.plot(wl, spectra_true[i], label='True')
    # plt.plot(wl, spectra_est[i], '--', label='Reconstructed')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Flux (arb. units)')
    # plt.title('Spectrum Reconstruction Example')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()