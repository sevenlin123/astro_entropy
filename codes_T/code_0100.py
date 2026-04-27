#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# Spectral model utilities
# ------------------------------------------------------------------
def create_wavelength_grid(start=300, stop=2500, num=2000):
    """Create a wavelength array in nm."""
    return np.linspace(start, stop, num)

def gaussian_template(wl, center, width, amplitude=1.0):
    """Return a Gaussian spectral template."""
    return amplitude * np.exp(-0.5 * ((wl - center) / width)**2)

def build_basis_spectra(wl):
    """Generate a set of basis spectra (e.g., Gaussian lines)."""
    templates = []
    centers = [500, 700, 900, 1100, 1300]  # nm
    widths = [50, 70, 90, 110, 130]
    for c, w in zip(centers, widths):
        templates.append(gaussian_template(wl, c, w))
    return np.vstack(templates)  # shape (n_basis, n_wl)

# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------
def generate_random_spectra(basis, n_objects=50):
    """
    Create synthetic spectra as random linear combinations of basis.
    Returns coefficients and spectra.
    """
    coeffs = np.random.rand(n_objects, basis.shape[0])  # uniform [0,1]
    spectra = coeffs @ basis  # shape (n_objects, n_wl)
    return coeffs, spectra

def define_filters():
    """Define simple photometric filter transmission curves."""
    filters = {}
    filt_names = ["U", "B", "V", "R", "I"]
    centers = [360, 440, 550, 640, 790]  # nm
    widths = [60, 80, 100, 120, 140]
    for name, cen, wid in zip(filt_names, centers, widths):
        filt = gaussian_template(wl, cen, wid, amplitude=1.0)
        filters[name] = filt / filt.max()
    return filters

def compute_photometry(spectra, filters):
    """
    Integrate spectra over filter transmissions.
    spectra: (n_objs, n_wl)
    filters: dict of (name, transmission array)
    Returns flux array shape (n_objs, n_filters).
    """
    wl = wl_array  # global wavelength array
    phot = np.zeros((spectra.shape[0], len(filters)))
    for i, (name, trans) in enumerate(filters.items()):
        integrand = spectra * trans
        phot[:, i] = simps(integrand, wl)
    return phot

# ------------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------------
def reconstruct_from_photometry(phot, basis, filters, alpha=1.0):
    """
    Given photometric fluxes, reconstruct spectrum via ridge regression.
    Returns reconstructed spectra (n_objs, n_wl).
    """
    # Build design matrix: for each filter, compute its integral over each basis
    wl = wl_array
    basis_flux = np.zeros((basis.shape[0], len(filters)))  # (n_basis, n_filters)
    for j, (name, trans) in enumerate(filters.items()):
        for i in range(basis.shape[0]):
            basis_flux[i, j] = simps(basis[i] * trans, wl)
    # Solve for coefficients that best reproduce photometry
    clf = Ridge(alpha=alpha, fit_intercept=False)
    clf.fit(basis_flux.T, phot.T)
    coeffs_rec = clf.coef_.T  # shape (n_objs, n_basis)
    spectra_rec = coeffs_rec @ basis
    return spectra_rec, coeffs_rec

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Setup wavelength grid
    wl_array = create_wavelength_grid()

    # 2. Build basis spectra
    basis_spectra = build_basis_spectra(wl_array)

    # 3. Generate synthetic objects
    true_coeffs, true_spectra = generate_random_spectra(basis_spectra, n_objects=10)

    # 4. Define filters
    phot_filters = define_filters()

    # 5. Compute photometry
    phot_flux = compute_photometry(true_spectra, phot_filters)

    # 6. Reconstruct spectra
    rec_spectra, rec_coeffs = reconstruct_from_photometry(phot_flux, basis_spectra, phot_filters)

    # 7. Display results for first object
    obj = 0
    print("True coefficients:", true_coeffs[obj])
    print("Recovered coefficients:", rec_coeffs[obj])
    print("\nWavelength (nm)   True Spectrum   Recovered Spectrum")
    for x, t, r in zip(wl_array[::100], true_spectra[obj][::100], rec_spectra[obj][::100]):
        print(f"{x:8.1f}   {t:15.6f}   {r:15.6f}")