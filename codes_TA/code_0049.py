import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def build_basis_spectra(n_basis, wavelengths):
    """Generate n_basis smooth basis spectra (Gaussian bumps)."""
    centers = np.linspace(0.3, 0.7, n_basis) * (wavelengths[-1] - wavelengths[0]) + wavelengths[0]
    widths = 0.05 * (wavelengths[-1] - wavelengths[0])
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        basis.append(g)
    return np.array(basis)          # shape: (n_basis, len(wavelengths))

def create_synthetic_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return coeffs @ basis            # shape: (len(wavelengths),)

# ---------- Filters ----------
def box_filter(wavelengths, center, width):
    """Box-shaped filter transmission."""
    return np.where(np.abs(wavelengths - center) <= width/2, 1.0, 0.0)

def build_filters(wavelengths, centers, width):
    """Create filter transmission curves."""
    return [box_filter(wavelengths, c, width) for c in centers]

# ---------- Photometry ----------
def integrate_spectrum(spectrum, transmission, wavelengths):
    """Integrate spectrum over a filter."""
    dl = np.gradient(wavelengths)
    return np.sum(spectrum * transmission * dl)

def photometry_from_spectrum(spectrum, filters, wavelengths):
    """Compute photometric fluxes for each filter."""
    return np.array([integrate_spectrum(spectrum, f, wavelengths) for f in filters])

# ---------- Reconstruction ----------
def reconstruct_coeffs(photometry, filter_integrals):
    """
    Solve for coefficients using least-squares linear regression
    (no intercept).
    """
    lr = LinearRegression(fit_intercept=False)
    lr.fit(filter_integrals, photometry)
    return lr.coef_          # shape: (n_basis,)

def reconstruct_spectrum(coeffs, basis):
    """Reconstruct full spectrum from coefficients and basis."""
    return coeffs @ basis

# ---------- Main ----------
def main():
    # Wavelength grid (400-700 nm, 300 points)
    wavelengths = np.linspace(400.0, 700.0, 300)

    # Basis spectra
    n_basis = 5
    basis = build_basis_spectra(n_basis, wavelengths)   # (n_basis, N)

    # True coefficients for synthetic spectrum
    np.random.seed(42)
    true_coeffs = 0.5 + np.random.rand(n_basis) * 1.0   # between 0.5 and 1.5

    # Synthetic spectrum
    synth_spec = create_synthetic_spectrum(basis, true_coeffs)

    # Filters
    filter_centers = np.array([450, 525, 600, 675])     # nm
    filter_width = 50.0                                 # nm
    filters = build_filters(wavelengths, filter_centers, filter_width)

    # Photometric fluxes (with noise)
    photometric_flux = photometry_from_spectrum(synth_spec, filters, wavelengths)
    noise = 0.02 * photometric_flux * np.random.randn(*photometric_flux.shape)
    noisy_flux = photometric_flux + noise

    # Pre-compute filter integrals for each basis component
    filter_integrals = np.array(
        [photometry_from_spectrum(b, filters, wavelengths) for b in basis]
    ).T                                   # shape: (n_filters, n_basis)

    # Reconstruct coefficients
    rec_coeffs = reconstruct_coeffs(noisy_flux, filter_integrals)

    # Reconstruct spectrum
    rec_spec = reconstruct_spectrum(rec_coeffs, basis)

    # Evaluation
    mse = np.mean((rec_spec - synth_spec)**2)
    print(f"True coefficients:    {true_coeffs}")
    print(f"Recovered coeffs:     {rec_coeffs}")
    print(f"Mean squared error on spectrum: {mse:.6e}")

if __name__ == "__main__":
    main()