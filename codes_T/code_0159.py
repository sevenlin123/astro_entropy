import numpy as np
from scipy.optimize import lsq_linear
from sklearn.metrics import mean_squared_error

# ----------------------------
# Spectral model definitions
# ----------------------------
def gaussian_basis(n_basis, wavelength):
    """Generate a set of Gaussian basis spectra."""
    centers = np.linspace(wavelength[0], wavelength[-1], n_basis)
    widths  = (wavelength[-1] - wavelength[0]) / (2 * n_basis)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelength - c)/widths)**2)
        basis.append(g)
    return np.vstack(basis)  # shape (n_basis, n_wavelength)

def synthetic_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return coeffs @ basis  # shape (n_wavelength,)

# ----------------------------
# Photometry generation
# ----------------------------
def top_hat_filters(n_filters, wavelength):
    """Generate simple top‑hat filter transmissions."""
    flt_wid = (wavelength[-1] - wavelength[0]) / (n_filters + 1)
    filters = []
    for i in range(n_filters):
        start = wavelength[0] + i*flt_wid
        stop  = start + flt_wid
        filt = np.logical_and(wavelength >= start, wavelength <= stop).astype(float)
        filters.append(filt)
    return np.vstack(filters)  # shape (n_filters, n_wavelength)

def photometry_from_spectrum(spectrum, filters):
    """Integrate spectrum over each filter."""
    return filters @ spectrum  # shape (n_filters,)

# ----------------------------
# Reconstruction routine
# ----------------------------
def reconstruct_spectrum(filters, photometry, basis):
    """
    Reconstruct spectrum by solving a linear least‑squares problem
    for the basis coefficients using the photometry constraints.
    """
    # Build design matrix: photometry = (filters @ basis.T) @ coeffs
    A = filters @ basis.T  # shape (n_filters, n_basis)
    res = lsq_linear(A, photometry, bounds=(0, np.inf))
    coeffs_rec = res.x
    spectrum_rec = coeffs_rec @ basis
    return spectrum_rec, coeffs_rec

# ----------------------------
# Main routine
# ----------------------------
def main():
    # Wavelength grid (nm)
    wav = np.linspace(400, 800, 500)

    # Basis spectra
    nbasis = 6
    basis = gaussian_basis(nbasis, wav)

    # True coefficients
    true_coeffs = np.random.rand(nbasis)

    # Synthetic spectrum
    true_spec = synthetic_spectrum(basis, true_coeffs)

    # Filters
    nfilters = 5
    filters = top_hat_filters(nfilters, wav)

    # Photometric measurements
    phot = photometry_from_spectrum(true_spec, filters)

    # Reconstruction
    rec_spec, rec_coeffs = reconstruct_spectrum(filters, phot, basis)

    # Evaluation
    mse = mean_squared_error(true_spec, rec_spec)
    print(f"True coefficients:  {true_coeffs}")
    print(f"Recovered coeffs:   {rec_coeffs}")
    print(f"Reconstruction MSE: {mse:.5e}")

if __name__ == "__main__":
    main()