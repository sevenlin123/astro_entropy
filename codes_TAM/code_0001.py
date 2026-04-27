import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------
# 1. Spectral model (basis functions)
# -----------------------------------------------------------

def wavelength_grid(start=3000.0, end=10000.0, num_points=1000):
    """Create a linear wavelength grid in Å."""
    return np.linspace(start, end, num_points)

def gaussian_basis(lambdas, centers, widths):
    """Return Gaussian basis functions evaluated on `lambdas`."""
    basis = []
    for c, w in zip(centers, widths):
        g = np.exp(-0.5 * ((lambdas - c) / w)**2)
        basis.append(g)
    return np.vstack(basis)          # shape: (n_basis, n_lambda)

def linear_basis(lambdas, n_basis):
    """Return simple linear basis functions (e.g., polynomials)."""
    basis = [np.ones_like(lambdas)]
    for i in range(1, n_basis):
        basis.append((lambdas - lambdas.mean())**i)
    return np.vstack(basis)

# -----------------------------------------------------------
# 2. Generate synthetic spectra
# -----------------------------------------------------------

def generate_synthetic_spectra(n_objs, basis, weight_range=(0.1, 1.0), noise_sigma=0.01):
    """
    Generate synthetic spectra by linearly combining basis functions
    with random weights. Returns spectra and true weights.
    """
    n_basis, n_lam = basis.shape
    weights = np.random.uniform(weight_range[0], weight_range[1],
                                size=(n_objs, n_basis))
    spectra = weights @ basis          # shape: (n_objs, n_lambda)
    # Add Gaussian noise
    spectra += np.random.normal(scale=noise_sigma, size=spectra.shape)
    return spectra, weights

# -----------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# -----------------------------------------------------------

def filter_response(lambdas, centers, widths, fwhm=None):
    """Return simple Gaussian filter responses."""
    if fwhm is None:
        fwhm = widths
    responses = []
    for c, w in zip(centers, widths):
        resp = np.exp(-0.5 * ((lambdas - c) / w)**2)
        responses.append(resp)
    return np.vstack(responses)       # shape: (n_filters, n_lambda)

def compute_photometry(spectra, filters, lambdas):
    """
    Integrate each spectrum through each filter to get photometric fluxes.
    Uses Simpson's rule for integration.
    """
    n_objs, n_lam = spectra.shape
    n_filt = filters.shape[0]
    phot = np.zeros((n_objs, n_filt))
    for i in range(n_filt):
        filt = filters[i]
        # element-wise product of spectra and filter
        integrand = spectra * filt[None, :]
        phot[:, i] = simps(integrand, lambdas, axis=1)
    return phot

# -----------------------------------------------------------
# 4. Reconstruct spectra from photometry
# -----------------------------------------------------------

def build_design_matrix(filters, basis):
    """
    Build design matrix F such that photometry = F @ weights.
    F_ij = ∫ basis_j(λ) * filter_i(λ) dλ
    """
    n_filt = filters.shape[0]
    n_basis, n_lam = basis.shape
    F = np.zeros((n_filt, n_basis))
    for i in range(n_filt):
        filt = filters[i]
        for j in range(n_basis):
            integrand = basis[j] * filt
            F[i, j] = simps(integrand, dx=(filters[:,0].max() - filters[:,0].min())/(n_lam-1))
    return F

def reconstruct_weights(photometry, design_matrix):
    """
    Solve least‑squares problem: weights = (FᵀF)⁻¹Fᵀ photometryᵀ
    Returns weights per object.
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, photometry.T)
    return reg.coef_.T   # shape: (n_objs, n_basis)

def reconstruct_spectra(weights, basis):
    """Recombine basis functions with estimated weights."""
    return weights @ basis

# -----------------------------------------------------------
# 5. Demo usage
# -----------------------------------------------------------

if __name__ == "__main__":
    # Wavelength grid
    lam = wavelength_grid()

    # Define basis (gaussian peaks)
    nbasis = 5
    centers = np.linspace(3500, 9500, nbasis)
    widths  = np.full(nbasis, 300.0)
    basis = gaussian_basis(lam, centers, widths)

    # Generate synthetic spectra
    n_objects = 50
    spectra, true_weights = generate_synthetic_spectra(n_objects, basis,
                                                       weight_range=(0.2, 1.0),
                                                       noise_sigma=0.02)

    # Define filters (simple Gaussian bandpasses)
    n_filters = 3
    filt_centers = np.array([4000, 7000, 9000])
    filt_widths  = np.array([200, 300, 250])
    filters = filter_response(lam, filt_centers, filt_widths)

    # Compute photometry
    phot = compute_photometry(spectra, filters, lam)

    # Build design matrix
    F = build_design_matrix(filters, basis)

    # Reconstruct weights
    recon_weights = reconstruct_weights(phot, F)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectra(recon_weights, basis)

    # Evaluate reconstruction
    rmse = np.sqrt(np.mean((spectra - recon_spectra)**2))
    print(f"RMSE between true and reconstructed spectra: {rmse:.4f}")

    # Compare true vs reconstructed weights for first object
    print("True weights (first object):", true_weights[0])
    print("Reconstructed weights (first object):", recon_weights[0])