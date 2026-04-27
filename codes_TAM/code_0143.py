import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ---------- Spectral model ----------
def generate_gaussian_basis(n_basis, w_min=300, w_max=800, w_step=1):
    """
    Create a set of Gaussian basis functions.
    
    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    w_min, w_max : float
        Wavelength limits (nm).
    w_step : float
        Wavelength step (nm).
        
    Returns
    -------
    wavelengths : ndarray
        Array of wavelengths (nm).
    basis : ndarray (n_basis, n_wavelengths)
        Basis functions evaluated on the wavelength grid.
    """
    wavelengths = np.arange(w_min, w_max + w_step, w_step)
    centers = np.linspace(w_min + 50, w_max - 50, n_basis)
    widths = np.full(n_basis, 20.0)   # constant width for simplicity
    basis = np.exp(-0.5 * ((wavelengths[None, :] - centers[:, None]) / widths[:, None]) ** 2)
    return wavelengths, basis


# ---------- Synthetic spectra ----------
def synthesize_spectra(n_samples, basis, coeff_scale=1.0):
    """
    Generate synthetic spectra as linear combinations of the basis.
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic spectra to generate.
    basis : ndarray (n_basis, n_wavelengths)
        Basis functions.
    coeff_scale : float
        Scale factor for coefficient amplitudes.
    
    Returns
    -------
    spectra : ndarray (n_samples, n_wavelengths)
        Synthetic spectra.
    true_coeffs : ndarray (n_samples, n_basis)
        True coefficients used to generate the spectra.
    """
    n_basis, n_wavelengths = basis.shape
    true_coeffs = coeff_scale * np.random.randn(n_samples, n_basis)
    spectra = true_coeffs @ basis
    return spectra, true_coeffs


# ---------- Filter responses ----------
def top_hat_filter(w_min, w_max, resolution, w_min_all=300, w_max_all=800):
    """Return a top‑hat filter response array."""
    mask = (resolution >= w_min) & (resolution <= w_max)
    return mask.astype(float)


def build_filters():
    """
    Build a set of simple top‑hat filters.
    
    Returns
    -------
    filter_names : list of str
        Names of the filters.
    filter_responses : ndarray (n_filters, n_wavelengths)
        Filter transmission arrays.
    """
    wavelengths = np.arange(300, 801, 1)
    filters = {
        'U': top_hat_filter(350, 400, wavelengths),
        'B': top_hat_filter(450, 500, wavelengths),
        'V': top_hat_filter(550, 600, wavelengths),
        'R': top_hat_filter(650, 700, wavelengths),
    }
    filter_names = list(filters.keys())
    filter_responses = np.array([filters[name] for name in filter_names])
    return wavelengths, filter_names, filter_responses


# ---------- Photometric integration ----------
def integrate_flux(spectra, filters, wavelengths):
    """
    Integrate spectra over filter responses.
    
    Parameters
    ----------
    spectra : ndarray (n_samples, n_wavelengths)
    filters : ndarray (n_filters, n_wavelengths)
    wavelengths : ndarray (n_wavelengths,)
    
    Returns
    -------
    photometry : ndarray (n_samples, n_filters)
    """
    # element‑wise product then integrate
    fluxes = spectra[:, :, None] * filters[None, :, :]
    photometry = np.array([simps(fluxes[:, f, :], wavelengths) for f in range(filters.shape[0])]).T
    return photometry


# ---------- Reconstruction ----------
def reconstruct_spectra(photometry, basis, filters, alpha=1.0):
    """
    Reconstruct spectra from photometric data.
    
    Parameters
    ----------
    photometry : ndarray (n_samples, n_filters)
    basis : ndarray (n_basis, n_wavelengths)
    filters : ndarray (n_filters, n_wavelengths)
    alpha : float
        Regularization strength for Ridge regression.
    
    Returns
    -------
    recon_spectra : ndarray (n_samples, n_wavelengths)
    recon_coeffs : ndarray (n_samples, n_basis)
    """
    # Construct design matrix: A_fi = ∫ basis_i(λ) * filter_f(λ) dλ
    n_basis, n_wave = basis.shape
    n_filters = filters.shape[0]
    A = np.zeros((n_filters, n_basis))
    for i in range(n_basis):
        for f in range(n_filters):
            integrand = basis[i] * filters[f]
            A[f, i] = simps(integrand, np.linspace(300, 800, n_wave))
    
    # Solve for coefficients via ridge regression
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(A, photometry.T)          # fit expects (n_samples, n_features)
    recon_coeffs = ridge.coef_.T         # shape (n_samples, n_basis)
    
    # Reconstruct spectra
    recon_spectra = recon_coeffs @ basis
    return recon_spectra, recon_coeffs


# ---------- Main demonstration ----------
if __name__ == "__main__":
    # Define spectral basis
    wav, basis = generate_gaussian_basis(n_basis=5)

    # Generate synthetic spectra
    spectra, true_coefs = synthesize_spectra(n_samples=10, basis=basis, coeff_scale=5.0)

    # Build filters
    _, filter_names, filters = build_filters()

    # Generate photometric measurements
    photometry = integrate_flux(spectra, filters, wav)

    # Reconstruct spectra from photometry
    recon_spectra, recon_coefs = reconstruct_spectra(photometry, basis, filters, alpha=0.1)

    # Evaluate reconstruction error (optional)
    mse = np.mean((spectra - recon_spectra) ** 2)
    print(f"Mean squared reconstruction error: {mse:.4f}")

    # Plotting (if desired, requires matplotlib)
    # import matplotlib.pyplot as plt
    # idx = 0
    # plt.plot(wav, spectra[idx], label="True")
    # plt.plot(wav, recon_spectra[idx], '--', label="Reconstructed")
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Flux")
    # plt.legend()
    # plt.show()