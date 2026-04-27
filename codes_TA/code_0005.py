import numpy as np
from scipy.integrate import simps

# ----------------------------------------------------------------------
# Wavelength grid
# ----------------------------------------------------------------------
def make_wavelength_grid(start=400.0, stop=700.0, n_points=1000):
    return np.linspace(start, stop, n_points)


# ----------------------------------------------------------------------
# Basis spectral model: a set of Gaussian line profiles
# ----------------------------------------------------------------------
def gaussian(wl, center, width, amplitude=1.0):
    return amplitude * np.exp(-0.5 * ((wl - center) / width) ** 2)


def build_basis_functions(wl, centers, widths, amplitudes=None):
    if amplitudes is None:
        amplitudes = np.ones_like(centers)
    return [
        gaussian(wl, c, w, a)
        for c, w, a in zip(centers, widths, amplitudes)
    ]


# ----------------------------------------------------------------------
# Generate a synthetic spectrum as a linear combination of basis functions
# ----------------------------------------------------------------------
def synthesize_spectrum(basis_funcs, coeffs):
    return np.dot(coeffs, basis_funcs)


# ----------------------------------------------------------------------
# Simple photometric filter responses (Gaussian bandpasses)
# ----------------------------------------------------------------------
def build_filter_response(wl, center, width):
    return gaussian(wl, center, width, amplitude=1.0)


def build_filters(wl, filter_specs):
    """
    filter_specs : list of tuples (name, center, width)
    """
    filters = {}
    for name, cen, wid in filter_specs:
        filters[name] = build_filter_response(wl, cen, wid)
    return filters


# ----------------------------------------------------------------------
# Compute synthetic photometry: integrated flux through each filter
# ----------------------------------------------------------------------
def compute_photometry(spectrum, filters, wl):
    phot = {}
    for name, resp in filters.items():
        # Simple trapezoidal integration; no zero-point conversion
        flux = simps(spectrum * resp, wl) / simps(resp, wl)
        phot[name] = flux
    return phot


# ----------------------------------------------------------------------
# Reconstruct spectrum from photometry via least‑squares on basis
# ----------------------------------------------------------------------
def reconstruct_from_photometry(phot, basis_funcs, filters, wl):
    filter_names = list(filters.keys())
    n_filters = len(filter_names)
    n_basis = len(basis_funcs)

    # Build design matrix A (filters x basis)
    A = np.empty((n_filters, n_basis))
    for i, fname in enumerate(filter_names):
        resp = filters[fname]
        for j, basis in enumerate(basis_funcs):
            A[i, j] = simps(basis * resp, wl) / simps(resp, wl)

    # Vector of observed fluxes
    y = np.array([phot[fname] for fname in filter_names])

    # Least-squares solution for coefficients
    coeffs_rec, *_ = np.linalg.lstsq(A, y, rcond=None)

    # Reconstructed spectrum
    spec_rec = synthesize_spectrum(basis_funcs, coeffs_rec)

    return coeffs_rec, spec_rec


# ----------------------------------------------------------------------
# Demo pipeline
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Create wavelength grid
    wl = make_wavelength_grid()

    # 2. Define basis functions (Gaussians)
    basis_centers = [450, 520, 590, 660, 730]
    basis_widths   = [10, 12, 15, 10, 8]
    basis_funcs = build_basis_functions(wl, basis_centers, basis_widths)

    # 3. Random coefficients for synthetic spectrum
    np.random.seed(42)
    true_coeffs = np.random.uniform(-1, 1, size=len(basis_funcs))

    # 4. Generate synthetic spectrum
    spectrum_true = synthesize_spectrum(basis_funcs, true_coeffs)

    # 5. Define simple photometric filters
    filter_specs = [
        ("U", 350.0, 30.0),
        ("B", 440.0, 20.0),
        ("V", 550.0, 25.0),
        ("R", 650.0, 30.0),
    ]
    filters = build_filters(wl, filter_specs)

    # 6. Compute synthetic photometry
    phot = compute_photometry(spectrum_true, filters, wl)

    # 7. Reconstruct spectrum from photometry
    rec_coeffs, spectrum_rec = reconstruct_from_photometry(phot, basis_funcs, filters, wl)

    # 8. Print comparison of true vs reconstructed coefficients
    print("True coefficients   :", true_coeffs)
    print("Recovered coeffs    :", rec_coeffs)

    # 9. Optional: compute residuals
    residual = spectrum_true - spectrum_rec
    rms_error = np.sqrt(np.mean(residual**2))
    print(f"Reconstruction RMS error: {rms_error:.5e}")