import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ---------- Parameters ----------
WAVELENGTHS = np.linspace(400, 800, 1000)   # nm
FILTER_WLCS = [450, 550, 650]               # nm
FILTER_WDTHS = [50, 60, 70]                 # nm
BASIS_CENTERS = np.linspace(420, 780, 20)   # nm
BASIS_WIDTH = 15                            # nm
np.random.seed(0)

# ---------- Helper functions ----------
def gaussian(wl, mu, sigma):
    """One-dimensional Gaussian."""
    return np.exp(-0.5 * ((wl - mu) / sigma)**2)

def basis_functions(wl, centers, width):
    """Generate a list of Gaussian basis functions."""
    return np.vstack([gaussian(wl, c, width) for c in centers]).T

def filter_response(wl, center, width):
    """Simple Gaussian filter transmission."""
    return gaussian(wl, center, width)

# ---------- Core stages ----------
def create_synthetic_spectrum(wl):
    """Generate a synthetic spectrum as a sum of random Gaussians."""
    num_peaks = 5
    amps = np.random.uniform(0.5, 1.5, size=num_peaks)
    mus = np.random.uniform(420, 780, size=num_peaks)
    sigmas = np.random.uniform(10, 30, size=num_peaks)
    spectrum = np.zeros_like(wl)
    for amp, mu, sigma in zip(amps, mus, sigmas):
        spectrum += amp * gaussian(wl, mu, sigma)
    return spectrum, dict(amps=amps, mus=mus, sigmas=sigmas)

def compute_photometry(spectrum, wl, filter_wlcs, filter_widths):
    """Compute photometric fluxes by integrating spectrum times filter."""
    fluxes = []
    for fc, fw in zip(filter_wlcs, filter_widths):
        filt = filter_response(wl, fc, fw)
        flux = trapz(spectrum * filt, wl)
        fluxes.append(flux)
    return np.array(fluxes)

def reconstruct_spectrum(fluxes, wl, filter_wlcs, filter_widths, centers, width):
    """Reconstruct spectrum via linear regression on Gaussian basis."""
    # Build design matrix A: A[i,j] = ∫ B_j(λ) * T_i(λ) dλ
    basis = basis_functions(wl, centers, width)           # shape (len(wl), n_basis)
    n_basis = basis.shape[1]
    A = np.zeros((len(filter_wlcs), n_basis))
    for i, (fc, fw) in enumerate(zip(filter_wlcs, filter_widths)):
        filt = filter_response(wl, fc, fw)
        A[i, :] = trapz(basis * filt[:, None], wl, axis=0)
    # Least-squares fit
    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, fluxes)
    coeffs = lr.coef_
    # Reconstruct spectrum
    spectrum_rec = basis @ coeffs
    return spectrum_rec, coeffs

# ---------- Example workflow ----------
if __name__ == "__main__":
    # 1. Create synthetic spectrum
    spec_true, true_params = create_synthetic_spectrum(WAVELENGTHS)

    # 2. Compute photometry
    flux_meas = compute_photometry(spec_true, WAVELENGTHS,
                                   FILTER_WLCS, FILTER_WDTHS)

    # 3. Reconstruct spectrum
    spec_rec, rec_coeffs = reconstruct_spectrum(
        flux_meas, WAVELENGTHS, FILTER_WLCS, FILTER_WDTHS,
        BASIS_CENTERS, BASIS_WIDTH)

    # 4. Simple error estimate
    error = np.linalg.norm(spec_true - spec_rec) / np.linalg.norm(spec_true)
    print(f"Relative L2 error between true and reconstructed spectrum: {error:.4f}")