import numpy as np
from scipy.special import erf
from sklearn.linear_model import LinearRegression

def create_wavelength_grid(n_points=200, w_min=4000.0, w_max=7000.0):
    return np.linspace(w_min, w_max, n_points)

def gaussian_basis(wavelengths, centers, widths):
    """Return basis matrix with Gaussian functions."""
    n_wave = len(wavelengths)
    n_basis = len(centers)
    B = np.zeros((n_wave, n_basis))
    for i, (c, w) in enumerate(zip(centers, widths)):
        B[:, i] = np.exp(-0.5 * ((wavelengths - c) / w)**2)
    return B

def generate_synthetic_spectra(B, n_samples=10, noise_std=0.01):
    """Generate spectra as random linear combinations of basis functions."""
    rng = np.random.default_rng(seed=42)
    coeffs = rng.standard_normal(size=(n_samples, B.shape[1]))
    spectra = coeffs @ B.T
    spectra += rng.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

def define_filters(wavelengths):
    """Simple rectangular filters."""
    filt_centers = [4500, 5500, 6500, 7500]
    filt_width = 500.0
    filters = []
    for fc in filt_centers:
        filt = np.exp(-0.5 * ((wavelengths - fc) / filt_width)**2)
        filters.append(filt)
    return np.array(filters)

def compute_photometry(spectra, filters):
    """Integrate spectra through filters."""
    # Assume equal spacing in wavelengths
    phot = spectra @ filters.T
    return phot

def build_filter_basis_matrix(filters, B):
    """Compute matrix A such that phot = A @ coeffs."""
    # Integrate basis functions through each filter
    A = filters @ B
    return A

def reconstruct_coefficients(phot, A):
    """Least‑squares solution for coefficients."""
    lr = LinearRegression(fit_intercept=False)
    lr.fit(A.T, phot.T)
    return lr.coef_.T

def reconstruct_spectra(coeffs, B):
    """Reconstruct spectra from coefficients."""
    return coeffs @ B.T

def main():
    wavelengths = create_wavelength_grid()
    centers = [4200, 4800, 5400, 6000, 6600]
    widths = [200] * len(centers)
    B = gaussian_basis(wavelengths, centers, widths)
    
    spectra_true, coeffs_true = generate_synthetic_spectra(B)
    
    filters = define_filters(wavelengths)
    photometry = compute_photometry(spectra_true, filters)
    
    A = build_filter_basis_matrix(filters, B)
    coeffs_rec = reconstruct_coefficients(photometry, A)
    
    spectra_rec = reconstruct_spectra(coeffs_rec, B)
    
    # Simple error metric
    error = np.mean((spectra_true - spectra_rec)**2)
    print(f"Mean squared reconstruction error: {error:.6f}")

if __name__ == "__main__":
    main()