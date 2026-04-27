import numpy as np
from sklearn.linear_model import Ridge

# -----------------------------
# Spectral model
# -----------------------------
def gaussian_basis(wl, centers, widths):
    """Generate Gaussian basis functions."""
    G = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths[None, :])**2)
    return G

def synth_spectrum(wl, coeffs, basis):
    """Construct synthetic spectrum as linear combo of basis."""
    return np.dot(basis, coeffs)

# -----------------------------
# Photometry generation
# -----------------------------
def gaussian_filter_response(wl, center, width):
    """Simple Gaussian bandpass filter."""
    return np.exp(-0.5 * ((wl - center)/width)**2)

def compute_phot_flux(spectrum, filt_resp, wl):
    """Integrate spectrum over filter."""
    return np.trapz(spectrum * filt_resp, wl) / np.trapz(filt_resp, wl)

# -----------------------------
# Reconstruction
# -----------------------------
def reconstruct_coeffs(phot_flux, basis, filters, wl):
    """Estimate coefficients from photometric fluxes."""
    # Build design matrix A_{j,i} = ∫ B_i(λ) F_j(λ) dλ / ∫ F_j(λ) dλ
    A = np.array([
        [np.trapz(basis[:, i] * filters[j], wl) / np.trapz(filters[j], wl)]
        for j in range(len(filters))
    ])
    # Solve for coefficients using ridge regression
    reg = Ridge(alpha=1e-6, fit_intercept=False)
    reg.fit(A, phot_flux)
    return reg.coef_

# -----------------------------
# Example workflow
# -----------------------------
def main():
    np.random.seed(42)
    # Wavelength grid
    wl = np.linspace(400, 800, 500)  # nm
    # Basis functions
    centers = np.array([450, 500, 550, 600, 650])
    widths  = np.array([20, 20, 20, 20, 20])
    basis = gaussian_basis(wl, centers, widths)  # shape (Nwl, Nbasis)

    # Filters (U,B,V,R,I approximated)
    filt_centers = np.array([360, 440, 550, 640, 790])
    filt_widths  = np.array([50, 50, 50, 50, 50])
    filters = [gaussian_filter_response(wl, c, w) for c, w in zip(filt_centers, filt_widths)]

    # Generate synthetic data
    n_samples = 10
    true_coeffs = np.random.uniform(0.5, 1.5, size=(n_samples, len(centers)))
    spectra = np.array([synth_spectrum(wl, c, basis) for c in true_coeffs])

    # Add Gaussian noise to spectra
    noise_level = 0.02
    noisy_spectra = spectra + noise_level * np.random.normal(size=spectra.shape)

    # Compute photometric fluxes
    phot_fluxes = np.array([ [compute_phot_flux(noisy_spectra[i], f, wl) for f in filters]
                             for i in range(n_samples)])

    # Reconstruct spectra
    recon_spectra = []
    recon_coeffs = []
    for i in range(n_samples):
        coeff_hat = reconstruct_coeffs(phot_fluxes[i], basis, filters, wl)
        recon_coeffs.append(coeff_hat)
        recon_spectra.append(synth_spectrum(wl, coeff_hat, basis))

    recon_spectra = np.array(recon_spectra)
    recon_coeffs = np.array(recon_coeffs)

    # Evaluation
    error = np.mean((true_coeffs - recon_coeffs)**2)
    print(f"Mean squared error on coefficients: {error:.4f}")

if __name__ == "__main__":
    main()