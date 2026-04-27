import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def wavelength_grid(start=400, stop=800, N=1000):
    """Create a wavelength array in nm."""
    return np.linspace(start, stop, N)

def gaussian_profile(x, amp, cen, sigma):
    """Gaussian spectral feature."""
    return amp * np.exp(-0.5 * ((x - cen) / sigma)**2)

def synthetic_spectrum(wl, features):
    """
    Sum multiple Gaussian features to produce a synthetic spectrum.
    `features` is a list of dicts with keys: amp, cen, sigma.
    """
    spec = np.zeros_like(wl)
    for f in features:
        spec += gaussian_profile(wl, f["amp"], f["cen"], f["sigma"])
    return spec

def filter_response(wl, center, width, shape='gaussian'):
    """
    Simple filter transmission curve.
    For now, Gaussian shape.
    """
    if shape == 'gaussian':
        # Convert width to sigma (FWHM ~ 2.355*sigma)
        sigma = width / 2.355
        return gaussian_profile(wl, 1.0, center, sigma)
    else:
        raise ValueError("Unsupported filter shape")

def integrate_flux(flux, wl):
    """Numerical integration over wavelength."""
    return np.trapz(flux, wl)

def photometry(spectrum, wl, filters):
    """
    Compute photometric fluxes through given filters.
    `filters` is a list of dicts: {center, width}.
    """
    phots = []
    for flt in filters:
        T = filter_response(wl, flt["center"], flt["width"])
        F = integrate_flux(spectrum * T, wl)
        norm = integrate_flux(T, wl)
        phots.append(F / norm)
    return np.array(phots)

def basis_matrix(filters, wl):
    """
    Build matrix A where each column is a basis function integrated over filters.
    Basis functions are the same as filter transmissions.
    """
    n_filters = len(filters)
    A = np.empty((n_filters, n_filters))
    for i, fi in enumerate(filters):
        Ti = filter_response(wl, fi["center"], fi["width"])
        for j, fj in enumerate(filters):
            Tj = filter_response(wl, fj["center"], fj["width"])
            A[i, j] = integrate_flux(Ti * Tj, wl)
    return A

def reconstruct_spectrum(basis_funcs, coeffs, wl):
    """Reconstruct spectrum as linear combination of basis functions."""
    recon = np.zeros_like(wl)
    for c, bf in zip(coeffs, basis_funcs):
        recon += c * bf
    return recon

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    # 1. Wavelength grid
    wl = wavelength_grid()

    # 2. Synthetic spectral model (true spectrum)
    true_features = [
        {"amp": 1.0, "cen": 450, "sigma": 10},
        {"amp": 0.6, "cen": 520, "sigma": 15},
        {"amp": 0.8, "cen": 600, "sigma": 12},
        {"amp": 0.4, "cen": 680, "sigma": 8},
    ]
    true_spec = synthetic_spectrum(wl, true_features)

    # 3. Define photometric filters
    filters = [
        {"center": 460, "width": 80},
        {"center": 550, "width": 80},
        {"center": 640, "width": 80},
    ]

    # 4. Generate photometric data
    photo = photometry(true_spec, wl, filters)

    # 5. Build basis functions (same as filters)
    basis_funcs = [filter_response(wl, f["center"], f["width"]) for f in filters]

    # 6. Construct matrix A
    A = basis_matrix(filters, wl)

    # 7. Solve for coefficients using ridge regression
    reg = Ridge(alpha=1e-3, fit_intercept=False, solver="auto")
    reg.fit(A, photo)
    coeffs = reg.coef_

    # 8. Reconstruct spectrum
    recon_spec = reconstruct_spectrum(basis_funcs, coeffs, wl)

    # 9. Output results
    print("Photometric measurements:")
    print(photo)
    print("\nReconstruction coefficients:")
    print(coeffs)
    print("\nFirst 10 values of true vs reconstructed spectrum:")
    print(np.column_stack([true_spec[:10], recon_spec[:10]]))

if __name__ == "__main__":
    main()