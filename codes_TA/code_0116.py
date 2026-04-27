import numpy as np
from sklearn.linear_model import LinearRegression

def define_spectral_basis(wavelengths, n_basis):
    """Construct a set of Gaussian basis functions."""
    mus = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    sigma = (wavelengths.max() - wavelengths.min()) / (n_basis * 4.0)
    basis = np.exp(
        -0.5 * ((wavelengths[:, None] - mus[None, :]) / sigma) ** 2
    )
    return basis

def generate_synthetic_spectra(basis, n_spectra, noise_std=0.01):
    """Generate synthetic spectra as linear combinations of the basis."""
    n_basis = basis.shape[1]
    coeffs_true = np.random.uniform(low=0.0, high=1.0, size=(n_spectra, n_basis))
    spectra = coeffs_true @ basis.T
    spectra += np.random.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs_true

def generate_filters(wavelengths, n_filters):
    """Create simple boxcar filters centered at different wavelengths."""
    centers = np.linspace(wavelengths.min() + 0.2*(wavelengths.max()-wavelengths.min()),
                          wavelengths.max() - 0.2*(wavelengths.max()-wavelengths.min()),
                          n_filters)
    width = (wavelengths.max() - wavelengths.min()) / (n_filters * 2.0)
    filters = []
    for c in centers:
        filt = np.where(np.abs(wavelengths - c) <= width / 2.0, 1.0, 0.0)
        filters.append(filt)
    return np.array(filters)

def photometric_fluxes_from_spectra(spectra, wavelengths, filters):
    """Compute fluxes in each filter by integrating spectrum*filter."""
    delta_lambda = np.diff(wavelengths, append=wavelengths[-1]+np.diff(wavelengths)[-1])
    fluxes = spectra @ (filters.T * delta_lambda)
    return fluxes

def reconstruct_coefficients_from_photometry(fluxes, basis, wavelengths, filters):
    """Reconstruct basis coefficients by solving linear system."""
    delta_lambda = np.diff(wavelengths, append=wavelengths[-1]+np.diff(wavelengths)[-1])
    # Build design matrix: each row corresponds to a filter, columns to basis coefficients
    A = np.zeros((filters.shape[0], basis.shape[1]))
    for i, filt in enumerate(filters):
        A[i] = (basis.T * (filt * delta_lambda)).sum(axis=1)
    # Fit linear model without intercept
    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, fluxes.T)   # shape (n_filters, n_spectra)
    coeffs_est = lr.coef_.T
    return coeffs_est

def main():
    # Parameters
    wav_start, wav_end, wav_step = 400.0, 1000.0, 2.0
    wavelengths = np.arange(wav_start, wav_end + wav_step, wav_step)
    n_basis = 5
    n_filters = 3
    n_spectra = 10
    noise_std = 0.02

    # Build basis
    basis = define_spectral_basis(wavelengths, n_basis)

    # Generate synthetic spectra and true coefficients
    spectra, coeffs_true = generate_synthetic_spectra(basis, n_spectra, noise_std)

    # Define filters
    filters = generate_filters(wavelengths, n_filters)

    # Compute photometric fluxes
    fluxes = photometric_fluxes_from_spectra(spectra, wavelengths, filters)

    # Reconstruct coefficients from photometry
    coeffs_est = reconstruct_coefficients_from_photometry(fluxes, basis, wavelengths, filters)

    # Reconstruct spectra
    spectra_rec = coeffs_est @ basis.T

    # Evaluate reconstruction accuracy
    rms_err = np.sqrt(((spectra - spectra_rec) ** 2).mean(axis=1))
    for i, err in enumerate(rms_err):
        print(f"Spectrum {i+1} RMS error: {err:.5f}")

if __name__ == "__main__":
    main()