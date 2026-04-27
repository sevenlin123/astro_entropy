import numpy as np
from scipy import integrate
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def generate_basis_spectra(n_basis, lam):
    """Create n_basis random basis spectra."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(size=(n_basis, len(lam))) * 0.5 + 1.0

def synth_spectrum(coeffs, basis):
    """Linear combination of basis spectra."""
    return coeffs @ basis

# ---------- Synthetic photometry ----------
def gaussian_filter(lam, center, width):
    """Simple Gaussian transmission curve."""
    return np.exp(-0.5 * ((lam - center)/width)**2)

def generate_filters(lam, centers, widths):
    """Generate list of filter transmission curves."""
    return [gaussian_filter(lam, c, w) for c, w in zip(centers, widths)]

def photometry_from_spectrum(spectrum, filters, lam):
    """Integrate spectrum over each filter."""
    fluxes = []
    for filt in filters:
        flux = integrate.trapz(spectrum * filt, lam)
        fluxes.append(flux)
    return np.array(fluxes)

# ---------- Reconstruction ----------
def reconstruct_spectrum_from_photometry(fluxes, filters, basis, lam):
    """
    Estimate basis coefficients from photometric fluxes using least squares,
    then reconstruct the full spectrum.
    """
    # Build design matrix: integral of each basis spectrum times each filter
    n_basis = basis.shape[0]
    n_filters = len(filters)
    A = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            A[i, j] = integrate.trapz(basis[j] * filt, lam)
    # Solve least-squares problem
    model = LinearRegression(fit_intercept=False).fit(A, fluxes)
    coeffs = model.coef_
    return synth_spectrum(coeffs, basis)

# ---------- Demo ----------
if __name__ == "__main__":
    # Wavelength grid
    lam = np.linspace(3000, 10000, 2000)  # Angstroms

    # Basis spectra
    n_basis = 5
    basis = generate_basis_spectra(n_basis, lam)

    # True coefficients
    rng = np.random.default_rng(seed=7)
    true_coeffs = rng.uniform(0.5, 1.5, size=n_basis)

    # Generate synthetic spectrum
    true_spec = synth_spectrum(true_coeffs, basis)

    # Filters
    centers = [4000, 5000, 6000, 7000, 8000]  # Angstroms
    widths  = [200, 250, 300, 350, 400]
    filters = generate_filters(lam, centers, widths)

    # Photometry
    obs_fluxes = photometry_from_spectrum(true_spec, filters, lam)

    # Reconstruct spectrum
    recon_spec = reconstruct_spectrum_from_photometry(obs_fluxes, filters, basis, lam)

    # Compare (simple printout)
    print("True coefficients:", true_coeffs)
    print("Reconstructed coefficients:")
    # Solve for reconstructed coeffs similarly as reconstruction step
    n_filters = len(filters)
    n_basis = basis.shape[0]
    A = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            A[i, j] = integrate.trapz(basis[j] * filt, lam)
    recon_coeffs = np.linalg.lstsq(A, obs_fluxes, rcond=None)[0]
    print(recon_coeffs)

    # Residual
    res = np.abs(true_spec - recon_spec) / true_spec
    print("Mean relative error in spectrum:", np.mean(res))