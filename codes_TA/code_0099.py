import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

def build_spectral_basis(wavelengths, n_basis=5):
    """Create a set of orthogonal basis spectra (Gaussian bumps)."""
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = 0.05 * (wavelengths.max() - wavelengths.min())
    basis = []
    for c in centers:
        gauss = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        basis.append(gauss)
    return np.vstack(basis)  # shape (n_basis, n_wave)

def generate_synthetic_spectra(basis, n_spectra=10, noise_level=0.02):
    """Generate spectra as random linear combos of basis plus noise."""
    coeffs = np.random.uniform(0.5, 1.5, size=(n_spectra, basis.shape[0]))
    spectra = coeffs @ basis
    spectra += noise_level * np.random.randn(*spectra.shape)
    return spectra, coeffs

def create_filters(wavelengths, n_filters=3):
    """Simple Gaussian bandpasses."""
    centers = np.linspace(wavelengths.min()+0.2*(wavelengths.max()-wavelengths.min()),
                          wavelengths.max()-0.2*(wavelengths.max()-wavelengths.min()),
                          n_filters)
    widths = 0.04 * (wavelengths.max() - wavelengths.min())
    filters = []
    for c in centers:
        filt = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        filters.append(filt)
    return np.vstack(filters)  # shape (n_filters, n_wave)

def generate_photometry(spectra, filters, wavelengths):
    """Integrate each spectrum over each filter to produce photometric fluxes."""
    phots = []
    for spec in spectra:
        flux = []
        for filt in filters:
            # numerical integration over wavelength grid
            flux.append(simps(spec * filt, wavelengths))
        phots.append(flux)
    return np.array(phots)  # shape (n_spectra, n_filters)

def reconstruct_spectrum(photometry, basis, filters, wavelengths):
    """
    Given photometry and basis, recover coefficients by solving a linear system.
    The system is: P = M * coeffs, where M_ij = integral(basis_j * filter_i).
    """
    # Build measurement matrix M
    n_basis = basis.shape[0]
    n_filters = filters.shape[0]
    M = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            M[i, j] = simps(basis[j] * filters[i], wavelengths)
    # Fit coefficients using least squares
    reg = LinearRegression(fit_intercept=False)
    reg.fit(M, photometry.T)
    coeffs_rec = reg.coef_.T
    reconstructed = coeffs_rec @ basis
    return reconstructed, coeffs_rec

def main():
    # Wavelength grid
    wav = np.linspace(3500, 10000, 1200)  # in nm
    # Build basis
    basis = build_spectral_basis(wav, n_basis=5)
    # Generate synthetic spectra and true coefficients
    spectra, true_coeffs = generate_synthetic_spectra(basis, n_spectra=8, noise_level=0.01)
    # Create filters
    filters = create_filters(wav, n_filters=4)
    # Generate photometry
    phots = generate_photometry(spectra, filters, wav)
    # Reconstruct spectra
    recon_spectra, rec_coeffs = reconstruct_spectrum(phots, basis, filters, wav)

    # Compare true vs recovered coefficients
    print("True coefficients vs Reconstructed coefficients (first spectrum):")
    print(true_coeffs[0])
    print(rec_coeffs[0])

if __name__ == "__main__":
    main()