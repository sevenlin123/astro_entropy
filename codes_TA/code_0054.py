import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# --------------------
# Spectral model setup
# --------------------
def gaussian_basis(wave, centers, widths):
    """
    Build an array of Gaussian basis functions.
    :param wave: 1‑D array of wavelengths
    :param centers: list of centers of Gaussians
    :param widths: list of widths of Gaussians
    :return: 2‑D array (n_wave, n_basis)
    """
    n_basis = len(centers)
    basis = np.zeros((len(wave), n_basis))
    for i, (c, w) in enumerate(zip(centers, widths)):
        basis[:, i] = np.exp(-0.5 * ((wave - c) / w)**2)
    return basis


def generate_spectrum(basis, coeffs):
    """
    Construct a spectrum as a linear combination of basis functions.
    """
    return basis @ coeffs


# --------------------
# Filter construction
# --------------------
def gaussian_filter(wave, center, width, amplitude=1.0):
    """
    Simple Gaussian filter transmission curve.
    """
    return amplitude * np.exp(-0.5 * ((wave - center) / width)**2)


def build_filters(wave, filter_centers, filter_widths):
    """
    Return a list of filter transmission curves.
    """
    return [gaussian_filter(wave, c, w) for c, w in zip(filter_centers, filter_widths)]


# --------------------
# Photometry generation
# --------------------
def photometry_from_spectrum(spectrum, filters):
    """
    Compute synthetic photometric fluxes by integrating the product
    of spectrum and filter transmission.
    """
    phot = []
    for filt in filters:
        phot.append(simps(spectrum * filt, x=wave))
    return np.array(phot)


# --------------------
# Reconstruction
# --------------------
def construct_design_matrix(basis, filters):
    """
    For each filter j and basis i compute the integral ∫ B_i(λ)*T_j(λ)dλ.
    """
    n_basis = basis.shape[1]
    n_filters = len(filters)
    M = np.empty((n_filters, n_basis))
    for j, filt in enumerate(filters):
        for i in range(n_basis):
            M[j, i] = simps(basis[:, i] * filt, x=wave)
    return M


def reconstruct_spectrum(basis, filters, photometry, alpha=1.0):
    """
    Reconstruct spectrum coefficients by solving the linear system
    M·a = photometry with ridge regression.
    """
    M = construct_design_matrix(basis, filters)
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(M, photometry)
    coeffs_hat = reg.coef_
    recon_spec = basis @ coeffs_hat
    return recon_spec, coeffs_hat


# --------------------
# Example usage
# --------------------
if __name__ == "__main__":
    # Wavelength grid
    wave = np.linspace(400, 700, 301)  # nm

    # Basis definition
    centers = [450, 500, 550, 600, 650]
    widths = [20, 20, 20, 20, 20]
    basis = gaussian_basis(wave, centers, widths)

    # True coefficients
    true_coeffs = np.array([5.0, -2.0, 3.0, 0.5, -1.0])

    # Generate synthetic spectrum
    true_spectrum = generate_spectrum(basis, true_coeffs)

    # Build photometric filters
    filt_centers = [460, 560, 660]
    filt_widths = [30, 30, 30]
    filters = build_filters(wave, filt_centers, filt_widths)

    # Compute synthetic photometry and add noise
    noiseless_flux = photometry_from_spectrum(true_spectrum, filters)
    noise = np.random.normal(scale=0.05, size=noiseless_flux.shape)
    noisy_flux = noiseless_flux + noise

    # Reconstruct spectrum
    recon_spectrum, recon_coeffs = reconstruct_spectrum(basis, filters, noisy_flux, alpha=0.1)

    # Evaluate reconstruction
    mse = np.mean((true_spectrum - recon_spectrum)**2)
    print(f"True coefficients     : {true_coeffs}")
    print(f"Reconstructed coeffs : {recon_coeffs}")
    print(f"Mean squared error   : {mse:.4f}")

    # Optional: compare spectra visually (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(wave, true_spectrum, label="True spectrum")
        plt.plot(wave, recon_spectrum, '--', label="Reconstructed")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux")
        plt.legend()
        plt.title("Spectrum Reconstruction")
        plt.show()
    except ImportError:
        pass