import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def create_wavelength_grid(start=4000, end=8000, num=1000):
    """Create a wavelength grid in Angstroms."""
    return np.linspace(start, end, num)

def gaussian_basis(wl, centers, widths):
    """Generate Gaussian basis functions."""
    bas = []
    for c, w in zip(centers, widths):
        bas.append(np.exp(-0.5 * ((wl - c) / w)**2))
    return np.array(bas)  # shape (n_basis, n_wave)

# ---------- Synthetic spectra ----------
def generate_synthetic_spectrum(basis, rng=None):
    """Generate a synthetic spectrum as a random linear combo of basis."""
    rng = rng or np.random.default_rng()
    coeffs = rng.uniform(-1, 1, size=basis.shape[0])
    return coeffs @ basis  # shape (n_wave,)

# ---------- Filters ----------
def random_top_hat_filters(n_filters, wl, rng=None):
    """Generate random top‑hat filter transmission curves."""
    rng = rng or np.random.default_rng()
    filt = np.zeros((n_filters, len(wl)))
    for i in range(n_filters):
        width = rng.integers(200, 600)
        center = rng.uniform(wl.min() + width/2, wl.max() - width/2)
        low = center - width/2
        high = center + width/2
        filt[i, (wl >= low) & (wl <= high)] = 1.0
    return filt  # shape (n_filters, n_wave)

# ---------- Photometry ----------
def photometry_from_spectrum(spectrum, filt, wl):
    """Integrate spectrum over each filter to get fluxes."""
    dwl = wl[1] - wl[0]
    return (filt * spectrum).sum(axis=1) * dwl  # shape (n_filters,)

# ---------- Reconstruction ----------
def reconstruct_weights(phot, filt, basis, wl):
    """Reconstruct basis coefficients from photometry."""
    dwl = wl[1] - wl[0]
    # Build design matrix: integral(filter * basis)
    A = (filt[:, :, None] * basis[None, :, :]).sum(axis=2) * dwl
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A.T, phot)
    return reg.coef_.T  # shape (n_basis,)

def reconstruct_spectrum(weights, basis):
    """Reconstruct full spectrum from basis coefficients."""
    return weights @ basis  # shape (n_wave,)

# ---------- Main demo ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid
    wl = create_wavelength_grid()

    # Basis functions
    centers = np.linspace(4100, 7900, 5)
    widths = np.full_like(centers, 300.0)
    basis = gaussian_basis(wl, centers, widths)

    # Generate synthetic spectrum
    true_spec = generate_synthetic_spectrum(basis, rng)

    # Filters
    filt = random_top_hat_filters(10, wl, rng)

    # Photometric data
    photo = photometry_from_spectrum(true_spec, filt, wl)

    # Reconstruct weights
    recon_weights = reconstruct_weights(photo, filt, basis, wl)

    # Reconstruct spectrum
    recon_spec = reconstruct_spectrum(recon_weights, basis)

    # Show results
    print("True coefficients:")
    print(rng.uniform(-1, 1, size=basis.shape[0]))  # not stored, just placeholder
    print("\nReconstructed coefficients:")
    print(recon_weights.flatten())

    # Compare spectra
    import matplotlib.pyplot as plt
    plt.plot(wl, true_spec, label="True spectrum")
    plt.plot(wl, recon_spec, "--", label="Reconstructed spectrum")
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.legend()
    plt.show()