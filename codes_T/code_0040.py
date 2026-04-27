import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Spectral model: simple basis of Gaussian peaks
# ----------------------------------------------------------------------
def gaussian_basis(wavelengths, centers, widths):
    """Return a matrix of Gaussian basis functions."""
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None]) / widths)**2)
    return basis

def normalize_basis(basis):
    """Normalize each basis column to unit area."""
    areas = trapz(basis, axis=0)
    return basis / areas

# ----------------------------------------------------------------------
# 2. Synthetic spectra generation
# ----------------------------------------------------------------------
def synthetic_spectrum(coeffs, basis):
    """Generate spectrum as linear combination of basis functions."""
    return basis @ coeffs

# ----------------------------------------------------------------------
# 3. Photometric data generation
# ----------------------------------------------------------------------
def filter_transmission(wavelengths, center, width):
    """Gaussian bandpass filter."""
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def photometry_from_spectrum(spectrum, wavelengths, filters):
    """Integrate spectrum over each filter to produce photometric fluxes."""
    phot = []
    for filt in filters:
        # Flux = integral( S(lambda) * T(lambda) ) / integral(T(lambda))
        num = trapz(spectrum * filt, wavelengths)
        denom = trapz(filt, wavelengths)
        phot.append(num / denom)
    return np.array(phot)

# ----------------------------------------------------------------------
# 4. Spectrum reconstruction from photometry
# ----------------------------------------------------------------------
def construct_design_matrix(filters, basis, wavelengths):
    """
    For each filter, compute the integral of each basis function through
    the filter.  This gives the matrix that maps coefficients to photometry.
    """
    M = []
    for filt in filters:
        row = []
        for col in basis.T:
            # integral(col * filt) / integral(filt)
            val = trapz(col * filt, wavelengths) / trapz(filt, wavelengths)
            row.append(val)
        M.append(row)
    return np.array(M)

def train_regression_model(M, n_samples=200):
    """
    Generate training data: random coefficients -> photometry,
    then fit a linear regression that maps photometry back to coefficients.
    """
    rng = np.random.default_rng(42)
    coeffs_train = rng.standard_normal((n_samples, M.shape[1]))
    phot_train = coeffs_train @ M.T          # photometry = coeffs * M^T
    reg = LinearRegression()
    reg.fit(phot_train, coeffs_train)
    return reg

def reconstruct_spectrum(reg, phot_obs, basis):
    """Predict coefficients from observed photometry and rebuild spectrum."""
    coeff_pred = reg.predict([phot_obs])[0]
    return basis @ coeff_pred

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    # Wavelength grid
    wav = np.linspace(400, 800, 1000)  # nm

    # Basis functions: 3 Gaussians
    centers = np.array([450, 550, 650])
    widths  = np.array([20, 20, 20])
    basis = gaussian_basis(wav, centers, widths)
    basis = normalize_basis(basis)

    # Filters: 4 bandpasses
    filt_centers = np.array([420, 520, 620, 720])
    filt_widths  = np.array([30, 30, 30, 30])
    filters = [filter_transmission(wav, c, w) for c, w in zip(filt_centers, filt_widths)]

    # True coefficients
    true_coeffs = np.array([1.0, 0.5, 0.8])

    # Generate true spectrum
    true_spectrum = synthetic_spectrum(true_coeffs, basis)

    # Compute photometry from true spectrum
    phot_obs = photometry_from_spectrum(true_spectrum, wav, filters)

    # Construct design matrix
    M = construct_design_matrix(filters, basis, wav)

    # Train regression model
    reg = train_regression_model(M)

    # Reconstruct spectrum
    recon_spectrum = reconstruct_spectrum(reg, phot_obs, basis)

    # Print results
    print("True coefficients:", true_coeffs)
    print("Reconstructed coefficients:", reg.predict([phot_obs])[0])
    print("\nFirst 10 values of true spectrum:\n", true_spectrum[:10])
    print("\nFirst 10 values of reconstructed spectrum:\n", recon_spectrum[:10])

if __name__ == "__main__":
    main()