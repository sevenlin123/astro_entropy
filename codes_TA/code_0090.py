import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: linear combination of basis functions
def define_basis_functions(n_points=200, n_bases=5):
    wavelengths = np.linspace(400, 800, n_points)  # nm
    bases = []
    for i in range(n_bases):
        center = 400 + i * 80
        width = 30
        base = np.exp(-0.5 * ((wavelengths - center)/width)**2)
        bases.append(base)
    return wavelengths, np.array(bases)

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_spectra, wavelengths, bases, noise_level=0.01):
    coeffs = np.random.randn(n_spectra, bases.shape[0])
    spectra = coeffs @ bases  # linear combination
    noise = noise_level * np.random.randn(*spectra.shape)
    return coeffs, spectra + noise

# 3. Generate photometric data from synthetic spectra
def generate_filters(wavelengths, n_filters=3):
    filters = []
    for i in range(n_filters):
        center = 450 + i * 100
        width = 50
        filt = np.exp(-0.5 * ((wavelengths - center)/width)**2)
        filters.append(filt / simps(filt, wavelengths))  # normalize
    return np.array(filters)

def photometric_measurements(spectra, wavelengths, filters):
    # integrate spectrum * filter response over wavelength
    measurements = np.zeros((spectra.shape[0], filters.shape[0]))
    for j, filt in enumerate(filters):
        prod = spectra * filt
        measurements[:, j] = simps(prod, wavelengths)
    return measurements

# 4. Reconstruct spectra from photometric data
def reconstruct_spectra(measurements, wavelengths, filters, n_bases):
    # Build matrix mapping coefficients to photometric measurements
    # A_ij = ∫ basis_j * filter_i
    A = np.zeros((filters.shape[0], n_bases))
    for i, filt in enumerate(filters):
        for j in range(n_bases):
            A[i, j] = simps(bases[j] * filt, wavelengths)
    # Solve for coefficients using ridge regression
    reg = Ridge(alpha=1e-6, fit_intercept=False)
    reg.fit(A, measurements.T)
    coeffs_recon = reg.coef_.T
    # Reconstruct spectra
    recon_spectra = coeffs_recon @ bases
    return recon_spectra, coeffs_recon

# ---- Main demonstration ----
np.random.seed(42)
wavelengths, bases = define_basis_functions()
coeffs_true, spectra_true = generate_synthetic_spectra(10, wavelengths, bases)
filters = generate_filters(wavelengths)
measurements = photometric_measurements(spectra_true, wavelengths, filters)
recon_spectra, coeffs_recon = reconstruct_spectra(measurements, wavelengths, filters, bases.shape[0])

# Print comparison of first true vs reconstructed spectrum
print("True spectrum (first):", spectra_true[0][:5])
print("Reconstructed spectrum (first):", recon_spectra[0][:5])