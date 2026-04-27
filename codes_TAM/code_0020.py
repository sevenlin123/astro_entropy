import numpy as np
from scipy.stats import norm

def generate_wavelengths(start=400, end=800, n=1000):
    """Generate a wavelength grid."""
    return np.linspace(start, end, n)

def create_basis_spectra(wavelengths, n_components=5):
    """Create a set of Gaussian basis spectra."""
    rng = np.random.default_rng()
    centers = rng.uniform(wavelengths[0], wavelengths[-1], size=n_components)
    widths  = rng.uniform(10, 30, size=n_components)
    basis = []
    for c, w in zip(centers, widths):
        spec = norm.pdf(wavelengths, loc=c, scale=w)
        basis.append(spec / spec.max())   # normalise
    return np.array(basis)  # shape (n_components, n_wavelengths)

def generate_random_coeffs(n_components):
    """Random linear combination coefficients."""
    rng = np.random.default_rng()
    return rng.uniform(0.5, 1.5, size=n_components)

def generate_synthetic_spectrum(basis, coeffs):
    """Form a synthetic spectrum as a linear combination of basis spectra."""
    return np.dot(coeffs, basis)

def define_filters(wavelengths, n_filters=3):
    """Define simple Gaussian filters."""
    rng = np.random.default_rng()
    centers = rng.uniform(wavelengths[0], wavelengths[-1], size=n_filters)
    widths  = rng.uniform(15, 40, size=n_filters)
    filters = []
    for c, w in zip(centers, widths):
        filt = norm.pdf(wavelengths, loc=c, scale=w)
        filters.append(filt / filt.max())   # normalise
    return np.array(filters)  # shape (n_filters, n_wavelengths)

def compute_photometry(spectrum, filters):
    """Integrate spectrum over each filter."""
    phot = []
    for filt in filters:
        integrand = spectrum * filt
        phot.append(np.trapz(integrand))
    return np.array(phot)

def reconstruct_spectrum_from_photometry(filters, photometry, basis, wavelengths):
    """
    Recover spectrum coefficients from photometry via least‑squares
    and reconstruct the spectrum.
    """
    n_filters, _ = filters.shape
    n_components = basis.shape[0]
    M = np.zeros((n_filters, n_components))
    for j in range(n_filters):
        for i in range(n_components):
            integrand = basis[i] * filters[j]
            M[j, i] = np.trapz(integrand)
    coeffs, *_ = np.linalg.lstsq(M, photometry, rcond=None)
    recon_spec = np.dot(coeffs, basis)
    return recon_spec, coeffs

def main():
    rng = np.random.default_rng(seed=42)
    wavelengths = generate_wavelengths()
    basis = create_basis_spectra(wavelengths)
    n_components = basis.shape[0]
    coeffs_true = generate_random_coeffs(n_components)
    true_spectrum = generate_synthetic_spectrum(basis, coeffs_true)
    filters = define_filters(wavelengths)
    photometry = compute_photometry(true_spectrum, filters)
    recon_spectrum, coeffs_rec = reconstruct_spectrum_from_photometry(
        filters, photometry, basis, wavelengths
    )
    err = np.linalg.norm(true_spectrum - recon_spectrum) / np.linalg.norm(true_spectrum)
    print(f"Reconstruction relative L2 error: {err:.4f}")
    print("True coefficients :", coeffs_true)
    print("Recovered coeffs :", coeffs_rec)

if __name__ == "__main__":
    main()