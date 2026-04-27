import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

def generate_spectrum(wavelengths, coeffs, basis_funcs):
    """Generate synthetic spectrum using a linear combination of basis functions."""
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for coeff, func in zip(coeffs, basis_funcs):
        spectrum += coeff * func(wavelengths)
    return spectrum

def generate_photometry(spectrum, wavelengths, filters):
    """Simulate photometric measurements by integrating over filter response curves."""
    photometry = []
    for filt in filters:
        # filt is a tuple: (filter_wavelengths, filter_response)
        filt_wl, filt_resp = filt
        # Interpolate spectrum onto filter wavelengths
        interp_spec = interp1d(wavelengths, spectrum, kind='linear', fill_value=0, bounds_error=False)
        spec_on_filt = interp_spec(filt_wl)
        # Integrate flux across filter
        photometry.append(np.trapz(spec_on_filt * filt_resp, filt_wl))
    return np.array(photometry)

def reconstruct_spectrum(photometry, wavelengths, filters, basis_funcs):
    """Reconstruct spectral coefficients using ridge regression."""
    # Build design matrix A: each row corresponds to one filter, each column to a basis function
    n_filters = len(filters)
    n_basis = len(basis_funcs)
    A = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters):
        filt_wl, filt_resp = filt
        # Evaluate basis at filter wavelengths
        for j, func in enumerate(basis_funcs):
            A[i, j] = np.trapz(func(filt_wl) * filt_resp, filt_wl)
    # Solve for coefficients using ridge regression
    reg = Ridge(alpha=1.0)
    reg.fit(A, photometry)
    coeffs = reg.coef_
    # Reconstruct spectrum
    recon_spectrum = np.zeros_like(wavelengths, dtype=float)
    for coeff, func in zip(coeffs, basis_funcs):
        recon_spectrum += coeff * func(wavelengths)
    return recon_spectrum

# ====== 生成示例数据和测试
if __name__ == "__main__":
    # Define wavelength range
    wl = np.linspace(4000, 10000, 500)  # in Angstroms
    # Define simple basis functions (e.g., Gaussian bumps)
    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma)**2)

    # Create basis functions
    basis_funcs = [
        lambda x: gaussian(x, 4500, 200),
        lambda x: gaussian(x, 5500, 200),
        lambda x: gaussian(x, 6500, 3e4?????? ??????) ??