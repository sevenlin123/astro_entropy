import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Lasso

# ---------- Spectral model ----------
def generate_basis(wavelengths, n_templates=5, seed=0):
    """
    Generate a set of template spectra (basis functions).
    Each template is a Gaussian centered at a random wavelength.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(4000, 8000, size=n_templates)
    widths  = rng.uniform(200, 600, size=n_templates)
    amplitudes = rng.uniform(0.5, 1.5, size=n_templates)
    
    basis = []
    for c, w, a in zip(centers, widths, amplitudes):
        spec = a * np.exp(-0.5 * ((wavelengths - c)/w)**2)
        basis.append(spec)
    return np.array(basis)  # shape (n_templates, n_wavelengths)

# ---------- Synthetic spectra ----------
def generate_synthetic_spectrum(basis, coeffs=None, noise_std=0.02, seed=42):
    """
    Produce a synthetic spectrum as a linear combination of basis templates.
    Optionally add Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    if coeffs is None:
        coeffs = rng.uniform(0.5, 1.5, size=basis.shape[0])
    spectrum = coeffs @ basis
    noise = rng.normal(scale=noise_std, size=spectrum.shape)
    return spectrum, coeffs

# ---------- Photometry ----------
def gaussian_bandpass(center, width, wavelengths):
    """Simple Gaussian bandpass response."""
    return np.exp(-0.5 * ((wavelengths - center)/width)**2)

def generate_bandpasses(n_bands=5, seed=1):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(4500, 7500, size=n_bands)
    widths  = rng.uniform(300, 700, size=n_bands)
    return list(zip(centers, widths))

def compute_photometry(spectrum, bandpasses, wavelengths):
    """
    Integrate the product of the spectrum with each bandpass response.
    """
    phot = []
    for c, w in bandpasses:
        response = gaussian_bandpass(c, w, wavelengths)
        flux = simps(spectrum * response, wavelengths)
        phot.append(flux)
    return np.array(phot)

# ---------- Reconstruction ----------
def reconstruct_from_photometry(basis, phot, bandpasses, wavelengths,
                                alpha=0.1, max_iter=1000):
    """
    Estimate the coefficients of the basis that best reproduce the photometry.
    Solves a constrained linear problem using Lasso (L1 regularisation).
    """
    # Build design matrix: integral of each basis with each bandpass
    M = np.empty((len(phot), basis.shape[0]))
    for i, (c, w) in enumerate(bandpasses):
        resp = gaussian_bandpass(c, w, wavelengths)
        M[i, :] = simps(basis * resp[:, np.newaxis], wavelengths, axis=1)
    # Fit coefficients
    lasso = Lasso(alpha=alpha, max_iter=max_iter, positive=True)
    lasso.fit(M, phot)
    coeffs_hat = lasso.coef_
    spectrum_hat = coeffs_hat @ basis
    return spectrum_hat, coeffs_hat

# ---------- Main routine ----------
if __name__ == "__main__":
    # Wavelength grid (in Å)
    wav = np.linspace(3500, 9000, 3000)

    # Basis templates
    basis = generate_basis(wav, n_templates=6)

    # Generate a synthetic spectrum
    true_spectrum, true_coeffs = generate_synthetic_spectrum(basis)

    # Bandpasses
    bands = generate_bandpasses(n_bands=7)

    # Compute synthetic photometry
    photometry = compute_photometry(true_spectrum, bands, wav)

    # Reconstruct spectrum
    recon_spec, reco_coeffs = reconstruct_from_photometry(
        basis, photometry, bands, wav, alpha=0.05)

    # ----- simple diagnostics -----
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(wav, true_spectrum, label='True spectrum')
    plt.plot(wav, recon_spec, '--', label='Reconstructed spectrum')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("True coefficients:     ", true_coeffs)
    print("Recovered coefficients:", reco_coeffs)