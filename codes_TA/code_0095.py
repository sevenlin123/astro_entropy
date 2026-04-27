import numpy as np
from scipy.signal import gaussian as scipy_gaussian
from sklearn.linear_model import LinearRegression

def gaussian(wl, center, sigma):
    """Simple Gaussian function."""
    return np.exp(-0.5 * ((wl - center) / sigma) ** 2)

def create_basis_spectra(wl, centers, sigmas):
    """Create a set of basis spectra as Gaussians."""
    return np.vstack([gaussian(wl, c, s) for c, s in zip(centers, sigmas)]).T

def generate_synthetic_spectrum(basis, weights):
    """Generate a synthetic spectrum as a linear combination of basis spectra."""
    return basis @ weights

def create_filters(wl, filter_centers, filter_widths):
    """Create top‑hat (Gaussian) filters."""
    filters = []
    for fc, fw in zip(filter_centers, filter_widths):
        f = gaussian(wl, fc, fw)
        f /= np.trapz(f, wl)          # normalise so that ∫f(λ)dλ=1
        filters.append(f)
    return np.array(filters)

def compute_photometry(spectrum, wl, filters):
    """Compute photometric fluxes by integrating the spectrum with each filter."""
    return np.array([np.trapz(spectrum * f, wl) for f in filters])

def reconstruct_weights(photometry, basis, filters, wl):
    """Reconstruct basis weights from photometric fluxes."""
    # Build response matrix R_{ij} = ∫ basis_j(λ) * filter_i(λ) dλ
    R = np.array([
        [np.trapz(basis[:, j] * f, wl) for j in range(basis.shape[1])]
        for f in filters
    ])
    # Solve linear system using ordinary least squares
    lr = LinearRegression(fit_intercept=False)
    lr.fit(R, photometry)
    return lr.coef_

def main():
    np.random.seed(42)

    # Wavelength grid
    wl = np.linspace(3000, 10000, 2000)  # Ångström

    # Basis spectra: 5 Gaussian absorption/emission features
    centers = np.linspace(3500, 9500, 5)
    sigmas  = np.full_like(centers, 200.0)
    basis   = create_basis_spectra(wl, centers, sigmas)

    # Generate random positive weights
    true_weights = np.abs(np.random.randn(basis.shape[1]))

    # Synthetic spectrum
    spec = generate_synthetic_spectrum(basis, true_weights)

    # Define 4 photometric filters
    filter_centers = np.array([4000, 5500, 7000, 8500])
    filter_widths  = np.array([500, 500, 500, 500])
    filters        = create_filters(wl, filter_centers, filter_widths)

    # Compute photometric fluxes
    photometry = compute_photometry(spec, wl, filters)

    # Reconstruct weights
    recovered_weights = reconstruct_weights(photometry, basis, filters, wl)

    print("True weights:     ", true_weights)
    print("Recovered weights:", recovered_weights)

if __name__ == "__main__":
    main()