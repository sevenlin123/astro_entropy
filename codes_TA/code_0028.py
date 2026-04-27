import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def generate_wavelength_grid(start=400.0, stop=800.0, num=1000):
    """Create a uniform wavelength grid (nm)."""
    return np.linspace(start, stop, num)

def gaussian_profile(wavelengths, center, width, amplitude=1.0):
    """Generate a Gaussian profile."""
    return amplitude * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def generate_basis_spectra(n_basis, wavelengths, seed=42):
    """Generate synthetic basis spectra as random Gaussians."""
    rng = np.random.default_rng(seed)
    basis = []
    for _ in range(n_basis):
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform(5.0, 20.0)
        amp = rng.uniform(0.5, 1.5)
        spec = gaussian_profile(wavelengths, center, width, amp)
        basis.append(spec)
    return np.vstack(basis)          # shape: (n_basis, n_wavelengths)

def generate_filter_transmissions(n_filters, wavelengths, seed=24):
    """Generate synthetic filter transmission curves."""
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform(15.0, 35.0)
        trans = gaussian_profile(wavelengths, center, width)
        trans /= trans.max()               # normalize to max 1
        filters.append(trans)
    return np.vstack(filters)              # shape: (n_filters, n_wavelengths)

def synthetic_spectrum(coeffs, basis):
    """Linear combination of basis spectra."""
    return coeffs @ basis                # shape: (n_wavelengths,)

def compute_photometry(spectrum, filters, wavelengths):
    """Integrate spectrum over each filter transmission curve."""
    dw = np.diff(wavelengths).mean()
    phot = np.sum(spectrum * filters, axis=1) * dw
    return phot                         # shape: (n_filters,)

def build_integration_matrix(basis, filters, wavelengths):
    """Compute the integral of each basis spectrum through each filter."""
    dw = np.diff(wavelengths).mean()
    n_filters, n_basis = filters.shape[0], basis.shape[0]
    M = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            M[i, j] = np.sum(basis[j] * filters[i]) * dw
    return M

def reconstruct_coeffs(photometry, integration_matrix):
    """Solve for coefficients using least‑squares."""
    model = LinearRegression(fit_intercept=False).fit(integration_matrix, photometry)
    return model.coef_

def reconstruct_spectrum(coeffs, basis):
    """Reconstruct spectrum from recovered coefficients."""
    return coeffs @ basis

# ----------------------------------------------------------------------
# Demo workflow
# ----------------------------------------------------------------------
def main():
    # 1. Define grid and basis
    wavelengths = generate_wavelength_grid()
    basis = generate_basis_spectra(n_basis=6, wavelengths=wavelengths)

    # 2. Generate synthetic spectrum
    rng = np.random.default_rng(99)
    true_coeffs = rng.uniform(0.2, 1.0, size=basis.shape[0])
    true_spectrum = synthetic_spectrum(true_coeffs, basis)

    # 3. Create filter set and obtain photometry
    filters = generate_filter_transmissions(n_filters=5, wavelengths=wavelengths)
    phot = compute_photometry(true_spectrum, filters, wavelengths)

    # 4. Reconstruct spectrum from photometry
    M = build_integration_matrix(basis, filters, wavelengths)
    rec_coeffs = reconstruct_coeffs(phot, M)
    rec_spectrum = reconstruct_spectrum(rec_coeffs, basis)

    # 5. Show comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, true_spectrum, label='True spectrum', lw=2)
    plt.plot(wavelengths, rec_spectrum, '--', label='Reconstructed spectrum', lw=2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux (arb. units)')
    plt.title('Spectrum Reconstruction from Photometry')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()