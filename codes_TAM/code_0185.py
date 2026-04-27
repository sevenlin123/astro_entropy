import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def planck(wl, T):
    """Planck function in W sr^-1 m^-2 µm^-1."""
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    wl_m = wl * 1e-6
    return (2*h*c**2 / wl_m**5) / (np.exp(h*c/(wl_m*k*T)) - 1)

def get_basis_spectra(wl):
    """Return list of basis spectra."""
    temps = [4000, 5000, 6000]            # K
    return np.array([planck(wl, T) for T in temps])  # shape (3, len)

# ---------- Filters ----------
def gaussian_filter(wl, center, width):
    return np.exp(-0.5*((wl-center)/width)**2)

def get_filters():
    """Return dictionary of filter responses."""
    centers = {'U': 360, 'B': 440, 'V': 550}
    widths  = {'U': 20,  'B': 35,  'V': 40}
    return {name: gaussian_filter(np.linspace(300, 800, 1000), c, w)
            for name, c, w in zip(centers.keys(), centers.values(), widths.values())}

# ---------- Generate synthetic data ----------
def generate_synthetic_spectra(num, wl, basis, rng):
    """Generate spectra with random weights."""
    weights = rng.uniform(0.5, 1.5, size=(num, basis.shape[0]))
    spectra = weights @ basis          # shape (num, len)
    return spectra, weights

def compute_photometry(spectra, wl, filters):
    """Integrate spectra over filter responses."""
    phot = []
    for filt in filters.values():
        integ = np.trapz(spectra * filt, wl, axis=1)
        phot.append(integ)
    return np.column_stack(phot)       # shape (num, num_filters)

# ---------- Reconstruction ----------
def reconstruct_weights(phot, filters, basis, wl):
    """Reconstruct basis weights from photometry."""
    # Build design matrix M where M_{ij} = ∫ basis_j * filter_i dλ
    M = []
    for filt in filters.values():
        M.append(np.trapz(basis * filt[:,None], wl, axis=1))
    M = np.vstack(M).T                 # shape (len(basis), num_filters)
    # Solve least squares: phot ≈ M^T w
    model = LinearRegression(fit_intercept=False)
    model.fit(M.T, phot)
    return model.coef_.T                # shape (num, len(basis))

# ---------- Demo ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    wl = np.linspace(300, 800, 200)        # nm
    basis = get_basis_spectra(wl)          # (3, 200)
    filters = get_filters()                # dict of 3 filters

    # Generate training data
    num_samples = 50
    spectra, true_weights = generate_synthetic_spectra(num_samples, wl, basis, rng)
    phot = compute_photometry(spectra, wl, filters)

    # Reconstruct weights
    rec_weights = reconstruct_weights(phot, filters, basis, wl)

    # Compare
    print("True weights vs Reconstructed weights:")
    for i in range(num_samples):
        print(f"{i+1:02d}: true={true_weights[i]}, recon={rec_weights[i]}")

    # Reconstruct spectra
    rec_spectra = rec_weights @ basis      # (num, len)
    # Optionally, evaluate error
    err = np.mean((spectra - rec_spectra)**2)
    print(f"\nMean squared reconstruction error: {err:.4e}")