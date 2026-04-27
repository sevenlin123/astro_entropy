import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def generate_basis(wavelengths, n_basis=5):
    """
    Create a simple set of basis spectra:
    - one flat continuum
    - several Gaussian absorption features
    """
    basis = np.ones((n_basis, len(wavelengths)))  # continuum
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis - 1)[1:]
    widths  = np.diff(np.concatenate([[wavelengths[0]], centers, [wavelengths[-1]]])) / 2

    for i, (c, w) in enumerate(zip(centers, widths)):
        gauss = np.exp(-0.5 * ((wavelengths - c) / w)**2)
        basis[i+1] *= (1 - 0.3 * gauss)          # absorption
    return basis

def generate_synthetic_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return coeffs @ basis

# ---------- Filters ----------
def gaussian_filter(wavelengths, center, width, throughput=1.0):
    """Simple Gaussian filter profile."""
    return throughput * np.exp(-0.5 * ((wavelengths - center) / width)**2)

def generate_filters(wavelengths):
    """Three broad band filters."""
    filt_U = gaussian_filter(wavelengths, 3500, 400)
    filt_V = gaussian_filter(wavelengths, 5500, 500)
    filt_R = gaussian_filter(wavelengths, 7500, 600)
    return np.vstack([filt_U, filt_V, filt_R])   # shape (n_filters, n_wave)

# ---------- Photometry ----------
def photometric_flux(spectrum, filters):
    """Integral of spectrum times filter transmission."""
    return (spectrum[:, None] * filters).sum(axis=0)   # (n_filters,)

# ---------- Reconstruction ----------
def reconstruct_spectrum(filters, basis, photometry):
    """
    Given measured photometry, recover coefficients via linear least squares.
    The matrix A has elements A_{ij} = \int basis_j * filter_i.
    """
    A = np.array([np.trapz(basis[j] * filters[i], axis=0)
                  for i in range(filters.shape[0])
                  for j in range(basis.shape[0])]).reshape(
                      filters.shape[0], basis.shape[0])

    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, photometry)
    coeffs_hat = lr.coef_
    spectrum_rec = coeffs_hat @ basis
    return coeffs_hat, spectrum_rec

# ---------- Main ----------
def main():
    # Wavelength grid (nm)
    wavelengths = np.linspace(3000, 10000, 1500)

    # Generate basis and true coefficients
    basis = generate_basis(wavelengths, n_basis=5)
    true_coeffs = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # arbitrary

    # Synthetic spectrum
    spec_true = generate_synthetic_spectrum(basis, true_coeffs)

    # Filter set
    filters = generate_filters(wavelengths)

    # Simulate photometric measurements
    phot_meas = photometric_flux(spec_true, filters)

    # Reconstruct
    coeffs_est, spec_rec = reconstruct_spectrum(filters, basis, phot_meas)

    # Output results
    print("True coefficients:", true_coeffs)
    print("Estimated coefficients:", coeffs_est)
    print("\nWavelength (nm)\tTrue Flux\tReconstructed Flux")
    for wl, f_true, f_rec in zip(wavelengths[::200],
                                 spec_true[::200],
                                 spec_rec[::200]):
        print(f"{wl:.1f}\t{f_true:.4f}\t{f_rec:.4f}")

if __name__ == "__main__":
    main()