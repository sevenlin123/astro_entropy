import numpy as np
from scipy import interpolate
from sklearn.linear_model import LinearRegression

# --------------------
# 1. Define a spectral model (basis functions)
# --------------------
def create_spectral_basis(n_basis=5, n_wavelengths=200, seed=0):
    """
    Generate a set of basis spectra.
    Each basis spectrum is a simple Gaussian profile with random center and width.
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(300, 1000, n_wavelengths)  # nm
    basis = []
    for _ in range(n_basis):
        amp = rng.uniform(0.5, 1.5)
        center = rng.uniform(350, 950)
        sigma = rng.uniform(20, 60)
        spec = amp * np.exp(-0.5 * ((wavelengths - center)/sigma)**2)
        basis.append(spec)
    return np.vstack(basis), wavelengths

# --------------------
# 2. Generate synthetic spectra
# --------------------
def generate_synthetic_spectra(basis, n_samples=10, seed=42):
    """
    Generate synthetic spectra as random linear combinations of the basis spectra.
    """
    rng = np.random.default_rng(seed)
    n_basis = basis.shape[0]
    coeffs = rng.uniform(0.1, 2.0, size=(n_samples, n_basis))
    spectra = coeffs @ basis
    return spectra, coeffs

# --------------------
# 3. Create photometric filter curves
# --------------------
def create_filters(n_filters=4, wavelengths=None, seed=24):
    """
    Create simple Gaussian filter transmission curves.
    """
    if wavelengths is None:
        raise ValueError("Wavelength grid must be provided.")
    rng = np.random.default_rng(seed)
    filters = []
    centers = rng.uniform(wavelengths[0]+50, wavelengths[-1]-50, size=n_filters)
    widths = rng.uniform(30, 80, size=n_filters)
    for c, w in zip(centers, widths):
        trans = np.exp(-0.5 * ((wavelengths - c)/w)**2)
        filters.append(trans)
    return np.vstack(filters)

# --------------------
# 4. Compute photometric measurements
# --------------------
def compute_photometry(spectra, filters, wavelengths):
    """
    Convolve spectra with filter curves to obtain synthetic photometry.
    """
    # Normalize each filter by its integral for proper flux scaling
    norm_filt = filters / np.trapz(filters, wavelengths, axis=1, keepdims=True)
    # Compute dot product (integral) between each spectrum and each filter
    photometry = spectra @ norm_filt.T
    return photometry

# --------------------
# 5. Reconstruct spectrum from photometry
# --------------------
def reconstruct_spectrum(photometry, basis, filters, wavelengths, n_components=None):
    """
    Recover the spectrum from photometric measurements by fitting the basis coefficients.
    """
    # Build design matrix: response of each basis to each filter
    norm_filt = filters / np.trapz(filters, wavelengths, axis=1, keepdims=True)
    design = basis @ norm_filt.T  # shape: (n_basis, n_filters)
    
    # Fit linear model: coefficients that best reproduce the photometry
    lr = LinearRegression(fit_intercept=False)
    lr.fit(design.T, photometry.T)
    coeffs_recon = lr.coef_.T
    
    # Reconstruct spectrum
    recon_spectrum = coeffs_recon @ basis
    return recon_spectrum, coeffs_recon

# --------------------
# 6. Main execution
# --------------------
if __name__ == "__main__":
    # Step 1: Basis
    basis, wavelengths = create_spectral_basis()
    
    # Step 2: Synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(basis)
    
    # Step 3: Filters
    filters = create_filters(wavelengths=wavelengths)
    
    # Step 4: Photometry
    photometry = compute_photometry(spectra, filters, wavelengths)
    
    # Step 5: Reconstruction for first sample
    recon_spec, recon_coeffs = reconstruct_spectrum(photometry[0:1], basis, filters, wavelengths)
    
    # Output comparison
    print("True coefficients for first sample:", true_coeffs[0])
    print("Reconstructed coefficients for first sample:", recon_coeffs[0])
    print("\nTrue spectrum vs reconstructed spectrum (first 10 points):")
    print("True :", spectra[0][:10])
    print("Recon:", recon_spec[0][:10])