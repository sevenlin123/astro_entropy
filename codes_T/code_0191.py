import numpy as np

# Physical constants
h = 6.62607015e-34      # Planck constant (J s)
c = 2.99792458e8        # Speed of light (m/s)
k = 1.380649e-23        # Boltzmann constant (J/K)

def planck_lam(wl_m, T):
    """Planck black‑body spectrum (W sr⁻¹ m⁻³)."""
    return 2.0 * h * c**2 / wl_m**5 / (np.exp(h * c / (wl_m * k * T)) - 1.0)

def make_filters(wavelengths):
    """Create five Gaussian filters."""
    centers = np.array([360, 440, 550, 640, 780])  # nm
    sigma   = 30.0                                 # nm
    filters = []
    for cen in centers:
        filt = np.exp(-0.5 * ((wavelengths - cen) / sigma)**2)
        filters.append(filt)
    return np.array(filters)                       # shape (5, N)

def integrate_filter(spectrum, wavelengths, filt):
    """Integrate spectrum through a single filter."""
    # wavelength spacing in nm (constant)
    dl = 1.0
    num = np.sum(spectrum * filt * dl)
    den = np.sum(filt * dl)
    return num / den if den != 0 else 0.0

def compute_photometry(spectra, wavelengths, filters):
    """Compute photometric fluxes for all spectra."""
    N = spectra.shape[0]
    F = filters.shape[0]
    P = np.zeros((N, F))
    for i in range(N):
        for j in range(F):
            P[i, j] = integrate_filter(spectra[i], wavelengths, filters[j])
    return P

def generate_synthetic_spectra(n_samp, wavelengths):
    """Generate black‑body spectra at random temperatures."""
    temps = np.random.uniform(4000, 8000, size=n_samp)  # K
    wl_m = wavelengths * 1e-9                          # convert to meters
    spectra = np.array([planck_lam(wl_m, T) for T in temps])
    # Normalize each spectrum (optional)
    spectra /= np.max(spectra, axis=1, keepdims=True)
    return spectra, temps

def train_coefficients(P, S):
    """Fit linear map from photometry to spectrum."""
    # Solve P*C ≈ S for C
    return np.linalg.pinv(P) @ S      # shape (F, W)

def reconstruct_spectrum(p_new, C):
    """Reconstruct spectrum from a new photometric vector."""
    return p_new @ C                   # shape (W,)

def main():
    # Wavelength grid (nm)
    wavelengths = np.arange(300, 1001, 1)          # 300–1000 nm
    # Create filters
    filters = make_filters(wavelengths)            # (5, N)
    # Generate synthetic spectra
    n_samples = 200
    spectra, temps = generate_synthetic_spectra(n_samples, wavelengths)   # (N, W)
    # Compute photometric observations
    P = compute_photometry(spectra, wavelengths, filters)                 # (N, F)
    # Train reconstruction coefficients
    C = train_coefficients(P, spectra)                                   # (F, W)
    # Test on a new synthetic spectrum
    test_temp = 6200.0
    wl_m = wavelengths * 1e-9
    true_spec = planck_lam(wl_m, test_temp)
    true_spec /= np.max(true_spec)
    # Compute photometry for test spectrum
    p_test = np.array([integrate_filter(true_spec, wavelengths, f) for f in filters])
    # Reconstruct spectrum
    recon_spec = reconstruct_spectrum(p_test, C)
    # Compute reconstruction error
    err = np.linalg.norm(true_spec - recon_spec) / np.linalg.norm(true_spec)
    print(f"Reconstruction relative error: {err:.4f}")

if __name__ == "__main__":
    main()