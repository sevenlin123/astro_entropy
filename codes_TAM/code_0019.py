import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def gaussian(wl, amp, cen, sigma):
    """Simple Gaussian profile."""
    return amp * np.exp(-0.5 * ((wl - cen) / sigma) ** 2)

def spectral_model(wl, params):
    """Sum of Gaussians defined by params = [(amp, cen, sigma), ...]."""
    flux = np.zeros_like(wl)
    for amp, cen, sigma in params:
        flux += gaussian(wl, amp, cen, sigma)
    return flux

def generate_synthetic_spectrum(wl, n_lines=3, rng=None):
    """Generate a synthetic spectrum and the parameters used."""
    rng = rng or np.random.default_rng()
    params = []
    for _ in range(n_lines):
        amp = rng.uniform(0.5, 1.5)
        cen = rng.uniform(450, 650)
        sigma = rng.uniform(5, 15)
        params.append((amp, cen, sigma))
    flux = spectral_model(wl, params)
    return flux, params

def create_filters(wl, n_filters=4, rng=None):
    """Create random photometric filter transmission curves."""
    rng = rng or np.random.default_rng()
    filters = []
    for _ in range(n_filters):
        start = rng.uniform(400, 600)
        end = rng.uniform(start + 20, 700)
        mid = (start + end) / 2
        width = (end - start) / 2
        trans = np.exp(-0.5 * ((wl - mid) / width) ** 2)
        trans /= trans.max()
        filters.append(trans)
    return np.array(filters)

def compute_photometry(flux, wl, filters):
    """Integrate flux over each filter to obtain synthetic photometry."""
    phot = []
    for filt in filters:
        phot.append(simps(flux * filt, wl))
    return np.array(phot)

def reconstruct_spectrum(phot, wl, filters, n_basis=12, alpha=1.0):
    """
    Reconstruct a spectrum from photometric measurements.
    Uses a set of Gaussian basis functions and ridge regression.
    """
    # Build basis functions
    centers = np.linspace(400, 700, n_basis)
    basis = np.vstack([gaussian(wl, 1.0, c, 10.0) for c in centers])  # shape (n_basis, N_wl)

    # Design matrix: integral of basis * filter
    M = np.zeros((len(filters), n_basis))
    for k, filt in enumerate(filters):
        for j in range(n_basis):
            M[k, j] = simps(basis[j] * filt, wl)

    # Solve for coefficients
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(M, phot)
    coeff = reg.coef_

    # Reconstruct flux
    rec_flux = coeff @ basis
    return rec_flux

# ------------------------------------------------------------------
# Main example
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Wavelength grid
    wl = np.linspace(400, 700, 300)

    # Generate a synthetic spectrum
    true_flux, true_params = generate_synthetic_spectrum(wl, rng=rng)
    # Add small noise
    noisy_flux = true_flux + rng.normal(scale=0.05, size=true_flux.shape)

    # Create photometric filters
    filters = create_filters(wl, rng=rng)

    # Compute synthetic photometry
    phot = compute_photometry(noisy_flux, wl, filters)

    # Reconstruct spectrum from photometry
    rec_flux = reconstruct_spectrum(phot, wl, filters)

    # Print results
    print("True line parameters:")
    for i, (amp, cen, sigma) in enumerate(true_params):
        print(f"  Line {i+1}: amp={amp:.3f}, cen={cen:.1f} nm, sigma={sigma:.1f} nm")

    print("\nSynthetic photometry:")
    for i, val in enumerate(phot):
        print(f"  Filter {i+1}: {val:.3f}")

    print("\nFirst 10 values of reconstructed flux:")
    print(rec_flux[:10])