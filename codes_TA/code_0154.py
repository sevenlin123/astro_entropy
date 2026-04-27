import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

def create_basis(n_bases, wavelengths):
    """Generate Gaussian basis functions."""
    rng = np.random.default_rng()
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), size=n_bases)
    widths = rng.uniform(15, 40, size=n_bases)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :])**2)
    return basis.T  # shape (n_bases, n_wavelengths)

def generate_synthetic_spectra(n_samples, basis):
    """Generate spectra as random linear combos of the basis."""
    rng = np.random.default_rng()
    coeffs = rng.uniform(0.5, 1.5, size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis  # shape (n_samples, n_wavelengths)
    return spectra, coeffs

def build_filters(wavelengths):
    """Define simple top‑hat photometric filters."""
    filt_specs = [
        (380, 500),  # U
        (500, 600),  # B
        (600, 720),  # V
    ]
    filters = []
    for low, high in filt_specs:
        trans = np.where((wavelengths >= low) & (wavelengths <= high), 1.0, 0.0)
        filters.append(trans)
    return np.array(filters)  # shape (n_filters, n_wavelengths)

def photometric_fluxes(spectra, filters, wavelengths):
    """Integrate spectra over each filter."""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    fluxes = np.empty((n_samples, n_filters))
    for i in range(n_filters):
        trans = filters[i]
        denom = simps(trans, wavelengths)
        fluxes[:, i] = simps(spectra * trans[None, :], wavelengths, axis=1) / denom
    return fluxes

def reconstruct_coeffs(fluxes, filters, basis, wavelengths):
    """Fit coefficients to match photometric fluxes."""
    n_filters = filters.shape[0]
    n_bases = basis.shape[0]
    X = np.empty((n_filters, n_bases))
    for j in range(n_filters):
        for k in range(n_bases):
            X[j, k] = simps(filters[j] * basis[k], wavelengths)
    # Fit each sample separately
    recon_coeffs = np.empty_like(fluxes)
    for i in range(fluxes.shape[0]):
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, fluxes[i])
        recon_coeffs[i] = lr.coef_
    return recon_coeffs

def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from fitted coefficients."""
    return coeffs @ basis  # shape (n_samples, n_wavelengths)

def main():
    wavelengths = np.linspace(350, 750, 401)  # nm
    n_bases = 5
    basis = create_basis(n_bases, wavelengths)
    spectra, true_coeffs = generate_synthetic_spectra(10, basis)
    filters = build_filters(wavelengths)
    fluxes = photometric_fluxes(spectra, filters, wavelengths)
    recon_coeffs = reconstruct_coeffs(fluxes, filters, basis, wavelengths)
    recon_spectra = reconstruct_spectra(recon_coeffs, basis)

    # Compare first spectrum
    idx = 0
    print("Original spectrum (first 10 values):")
    print(spectra[idx, :10])
    print("\nReconstructed spectrum (first 10 values):")
    print(recon_spectra[idx, :10])

if __name__ == "__main__":
    main()