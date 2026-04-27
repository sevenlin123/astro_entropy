import numpy as np
from scipy.integrate import trapz

# ---------- Spectral model ----------
def gaussian_basis_functions(wavelength, n_basis):
    """
    Return n_basis Gaussian basis functions evaluated on wavelength grid.
    Centers are evenly spaced; widths are chosen relative to grid span.
    """
    wl_min, wl_max = wavelength[0], wavelength[-1]
    centers = np.linspace(wl_min + 0.1*(wl_max-wl_min),
                          wl_max - 0.1*(wl_max-wl_min), n_basis)
    width = 0.05 * (wl_max - wl_min)  # fixed width
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelength - c)/width)**2)
        basis.append(g)
    return np.vstack(basis)   # shape (n_basis, len(wavelength))

# ---------- Synthetic spectra ----------
def generate_synthetic_spectrum(basis, rng=np.random.default_rng()):
    """Draw random coefficients and build synthetic spectrum."""
    coeffs = rng.uniform(0.5, 1.5, size=basis.shape[0])
    spectrum = coeffs @ basis          # linear combination
    return spectrum, coeffs

# ---------- Photometric data ----------
def filter_transmission(wavelength, center, width):
    """Top-hat transmission curve."""
    return np.where((wavelength >= center - width/2) &
                    (wavelength <= center + width/2), 1.0, 0.0)

def measure_photometry(spectrum, wavelength, filters):
    """
    Compute integrated flux through each filter.
    Filters is a list of dicts with keys 'center' and 'width'.
    Returns array of photometric fluxes.
    """
    fluxes = []
    for filt in filters:
        trans = filter_transmission(wavelength, filt['center'], filt['width'])
        flux = trapz(spectrum * trans, wavelength) / trapz(trans, wavelength)
        fluxes.append(flux)
    return np.array(fluxes)

# ---------- Reconstruction ----------
def reconstruct_coefficients(filters, wavelength, basis, photometry):
    """
    Solve for basis coefficients that best reproduce the photometry.
    """
    n_filters = len(filters)
    n_basis = basis.shape[0]
    # Precompute matrix: M[j,i] = integral of basis[i] * filter[j] / integral filter[j]
    M = np.empty((n_filters, n_basis))
    for j, filt in enumerate(filters):
        trans = filter_transmission(wavelength, filt['center'], filt['width'])
        denom = trapz(trans, wavelength)
        for i in range(n_basis):
            numerator = trapz(basis[i] * trans, wavelength)
            M[j, i] = numerator / denom
    # Least-squares solution
    coeffs, *_ = np.linalg.lstsq(M, photometry, rcond=None)
    return coeffs

def reconstruct_spectrum(coeffs, basis):
    """Build spectrum from coefficients and basis functions."""
    return coeffs @ basis

# ---------- Main routine ----------
def main():
    rng = np.random.default_rng(42)
    # Wavelength grid
    wavelength = np.linspace(400, 800, 2000)   # nm
    # Basis functions
    n_basis = 6
    basis = gaussian_basis_functions(wavelength, n_basis)
    # Generate synthetic spectrum
    true_spectrum, true_coeffs = generate_synthetic_spectrum(basis, rng)
    # Define filters
    filters = [
        {'center': 450, 'width': 50},
        {'center': 530, 'width': 60},
        {'center': 610, 'width': 55},
        {'center': 690, 'width': 70},
        {'center': 770, 'width': 65}
    ]
    # Measure photometry
    photometry = measure_photometry(true_spectrum, wavelength, filters)
    # Reconstruct coefficients
    recon_coeffs = reconstruct_coefficients(filters, wavelength, basis, photometry)
    # Reconstruct spectrum
    recon_spectrum = reconstruct_spectrum(recon_coeffs, basis)
    # Evaluate reconstruction
    error = np.abs(recon_spectrum - true_spectrum).mean()
    print(f"Mean absolute error in reconstructed spectrum: {error:.4f}")
    # Optional: print true vs recovered coefficients
    for i, (t, r) in enumerate(zip(true_coeffs, recon_coeffs)):
        print(f"Basis {i+1}: true coeff={t:.3f}, recovered={r:.3f}")

if __name__ == "__main__":
    main()