import numpy as np
from scipy.integrate import trapz

# ------------------------------------------------------------------
# 1. Define spectral model (basis functions)
# ------------------------------------------------------------------
def gaussian_basis(wl, centers, widths):
    """
    Build a set of Gaussian basis functions on wavelength grid.
    
    Parameters
    ----------
    wl : array_like
        Wavelength grid (nm).
    centers : array_like
        Centers of Gaussian functions (nm).
    widths : array_like
        Standard deviations of Gaussian functions (nm).
        
    Returns
    -------
    basis : ndarray shape (n_basis, len(wl))
        Normalized Gaussian basis functions.
    """
    basis = []
    for c, w in zip(centers, widths):
        gauss = np.exp(-0.5 * ((wl - c) / w)**2)
        gauss /= trapz(gauss, wl)  # normalise to unit area
        basis.append(gauss)
    return np.vstack(basis)

# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def generate_spectrum(basis, coeffs):
    """
    Produce a synthetic spectrum as a linear combination of basis functions.
    
    Parameters
    ----------
    basis : ndarray shape (n_basis, n_wl)
    coeffs : array_like shape (n_basis,)
    
    Returns
    -------
    flux : ndarray shape (n_wl,)
    """
    return coeffs @ basis

def generate_random_coeffs(n_basis, rng=None):
    """
    Draw random coefficients from a uniform distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(0.5, 1.5, size=n_basis)

# ------------------------------------------------------------------
# 3. Define photometric filters
# ------------------------------------------------------------------
def gaussian_filter(wl, center, width):
    """
    Gaussian filter transmission curve.
    """
    trans = np.exp(-0.5 * ((wl - center) / width)**2)
    return trans

def build_filters(wl):
    """
    Create a list of filter transmission curves.
    """
    centers = [450, 550, 650, 750]
    widths  = [30, 30, 30, 30]
    return [gaussian_filter(wl, c, w) for c, w in zip(centers, widths)]

# ------------------------------------------------------------------
# 4. Compute photometry from spectra
# ------------------------------------------------------------------
def compute_photometry(flux, wl, filters):
    """
    Integrate flux over each filter transmission curve.
    """
    phot = []
    for filt in filters:
        integ = trapz(flux * filt, wl)
        filt_norm = trapz(filt, wl)
        phot.append(integ / filt_norm)
    return np.array(phot)

# ------------------------------------------------------------------
# 5. Reconstruct spectrum from photometry
# ------------------------------------------------------------------
def reconstruct_coeffs(phot, wl, filters, basis):
    """
    Solve for basis coefficients that best reproduce photometric data
    using least‑squares.
    """
    n_basis = basis.shape[0]
    n_filt  = len(filters)
    
    # Build design matrix: each filter's flux contribution from one basis
    A = np.zeros((n_filt, n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            integ = trapz(basis[j] * filt, wl)
            filt_norm = trapz(filt, wl)
            A[i, j] = integ / filt_norm
    
    coeffs, *_ = np.linalg.lstsq(A, phot, rcond=None)
    return coeffs

# ------------------------------------------------------------------
# 6. Main routine – synthetic example
# ------------------------------------------------------------------
def main():
    # Wavelength grid
    wl = np.linspace(400, 800, 400)          # 400–800 nm
    
    # Basis functions
    centers = [450, 550, 650]
    widths  = [20, 20, 20]
    basis   = gaussian_basis(wl, centers, widths)   # shape (3, 400)
    
    # Filters
    filters = build_filters(wl)            # 4 filters
    
    # Generate synthetic spectra
    n_spectra = 5
    rng = np.random.default_rng(42)
    true_coeffs_list = []
    spectra = []
    for _ in range(n_spectra):
        coeffs = generate_random_coeffs(len(centers), rng)
        true_coeffs_list.append(coeffs)
        flux = generate_spectrum(basis, coeffs)
        spectra.append(flux)
    
    # Compute photometry
    photometry = np.array([compute_photometry(f, wl, filters) for f in spectra])
    
    # Reconstruct coefficients
    estimated_coeffs = []
    for phot in photometry:
        est = reconstruct_coeffs(phot, wl, filters, basis)
        estimated_coeffs.append(est)
    
    # Display results
    print("True vs Estimated Coefficients")
    for i, (true, est) in enumerate(zip(true_coeffs_list, estimated_coeffs)):
        print(f"Spectrum {i+1}:")
        print(f"  True : {true}")
        print(f"  Est  : {est}\n")

if __name__ == "__main__":
    main()