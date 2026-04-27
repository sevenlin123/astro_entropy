import numpy as np
from scipy.integrate import trapz

def generate_wavelength_grid(start=400, stop=800, num=400):
    """Wavelength grid in nm."""
    return np.linspace(start, stop, num)

def generate_basis_functions(wl, n_basis=5):
    """Simple polynomial basis functions."""
    X = np.vstack([wl**i for i in range(n_basis)]).T
    # normalize each column
    X /= np.max(X, axis=0)
    return X

def generate_synthetic_spectrum(basis, coeffs):
    """Linear combination of basis functions."""
    return basis @ coeffs

def generate_filters(n_filters, wl):
    """Random Gaussian filters."""
    filters = []
    rng = np.random.default_rng(seed=42)
    centers = np.linspace(450, 750, n_filters)
    widths = np.linspace(20, 40, n_filters)
    for cen, wid in zip(centers, widths):
        filt = np.exp(-0.5 * ((wl - cen) / wid)**2)
        filt /= np.max(filt)  # normalize peak to 1
        filters.append(filt)
    return filters

def compute_band_flux(spectrum, filters, wl):
    """Integrate spectrum over each filter."""
    fluxes = [trapz(spectrum * filt, wl) for filt in filters]
    return np.array(fluxes)

def reconstruct_spectrum(band_flux, filters, basis, wl):
    """
    Solve linear system A*c = f_obs where
    A[i, j] = ∫ basis_j(λ) * filter_i(λ) dλ
    """
    n_filters = len(filters)
    n_basis = basis.shape[1]
    A = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            A[i, j] = trapz(basis[:, j] * filt, wl)
    coeffs, *_ = np.linalg.lstsq(A, band_flux, rcond=None)
    return generate_synthetic_spectrum(basis, coeffs), coeffs

def main():
    # Wavelength grid
    wl = generate_wavelength_grid()
    # Basis functions
    basis = generate_basis_functions(wl, n_basis=5)
    # True coefficients
    rng = np.random.default_rng(seed=123)
    true_coeffs = rng.uniform(-1, 1, size=basis.shape[1])
    # Generate synthetic spectrum
    true_spectrum = generate_synthetic_spectrum(basis, true_coeffs)
    # Filters
    filters = generate_filters(n_filters=6, wl=wl)
    # Photometric measurements
    band_flux = compute_band_flux(true_spectrum, filters, wl)
    # Reconstruction
    recon_spectrum, recon_coeffs = reconstruct_spectrum(
        band_flux, filters, basis, wl
    )
    # Evaluate reconstruction quality
    error = np.sqrt(np.mean((true_spectrum - recon_spectrum)**2))
    print("Reconstruction RMS error:", error)
    print("True coefficients:", true_coeffs)
    print("Recovered coefficients:", recon_coeffs)

if __name__ == "__main__":
    main()