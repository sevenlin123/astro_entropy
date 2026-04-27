import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# Spectral basis: five Gaussian functions
# --------------------------------------------------------------------
def gaussian_basis(wl, centers, widths):
    """Return array of shape (n_basis, n_wavelengths)"""
    return np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths[None, :]) ** 2)

# --------------------------------------------------------------------
# Generate a synthetic spectrum
# --------------------------------------------------------------------
def generate_spectrum(wl, coeffs, basis_funcs):
    """Compute flux = sum(coeff * basis)"""
    return coeffs @ basis_funcs

# --------------------------------------------------------------------
# Filter definitions (simple top‑hat filters)
# --------------------------------------------------------------------
def generate_filters(wl):
    """Return dict of filter transmissions."""
    filters = {}
    # U filter (3500–4100 Å)
    filt_u = np.logical_and(wl >= 3500, wl <= 4100).astype(float)
    filters['U'] = filt_u
    # B filter (4200–5200 Å)
    filt_b = np.logical_and(wl >= 4200, wl <= 5200).astype(float)
    filters['B'] = filt_b
    # V filter (5300–6300 Å)
    filt_v = np.logical_and(wl >= 5300, wl <= 6300).astype(float)
    filters['V'] = filt_v
    return filters

# --------------------------------------------------------------------
# Photometry from a spectrum
# --------------------------------------------------------------------
def photometry_from_spectrum(flux, wl, filters):
    """
    Compute integrated flux in each filter:
        F_i = ∫ F(λ) T_i(λ) dλ / ∫ T_i(λ) dλ
    """
    phots = {}
    for name, trans in filters.items():
        numerator = np.trapz(flux * trans, wl)
        denominator = np.trapz(trans, wl)
        phots[name] = numerator / denominator
    return phots

# --------------------------------------------------------------------
# Reconstruction of coefficients from photometry
# --------------------------------------------------------------------
def reconstruct_coefficients(phots, wl, filters, basis_funcs):
    """
    Solve linear system A c = y where
        A_ij = ∫ φ_j(λ) T_i(λ) dλ / ∫ T_i(λ) dλ
    """
    band_names = list(phots.keys())
    A = np.zeros((len(band_names), basis_funcs.shape[0]))
    y = np.zeros(len(band_names))
    for i, name in enumerate(band_names):
        trans = filters[name]
        denom = np.trapz(trans, wl)
        for j in range(basis_funcs.shape[0]):
            A[i, j] = np.trapz(basis_funcs[j] * trans, wl) / denom
        y[i] = phots[name]

    # Least squares solution (could also use Ridge if needed)
    lr = LinearRegression(fit_intercept=False).fit(A, y)
    return lr.coef_

# --------------------------------------------------------------------
# Main synthetic example
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (Å)
    wl = np.linspace(3000, 10000, 1000)

    # Basis parameters
    centers = np.array([3500, 4500, 5500, 6500, 7500])
    widths  = np.full_like(centers, 200.)
    basis_funcs = gaussian_basis(wl, centers, widths)

    # Generate random true coefficients
    np.random.seed(42)
    true_coeffs = np.random.uniform(-1, 1, size=basis_funcs.shape[0])

    # Synthetic spectrum
    true_flux = generate_spectrum(wl, true_coeffs, basis_funcs)

    # Filters
    filters = generate_filters(wl)

    # Photometric data
    phots = photometry_from_spectrum(true_flux, wl, filters)

    # Reconstruction
    recon_coeffs = reconstruct_coefficients(phots, wl, filters, basis_funcs)
    recon_flux   = generate_spectrum(wl, recon_coeffs, basis_funcs)

    # Simple diagnostics
    print("True coefficients   :", true_coeffs)
    print("Reconstructed coefs:", recon_coeffs)
    print("Flux error rms      :", np.sqrt(np.mean((true_flux - recon_flux)**2)))