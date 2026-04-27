import numpy as np
from scipy import integrate

# --------------------
# Spectral model
# --------------------
def generate_basis(n_wavelengths=200, n_bases=5, rng=None):
    """Generate a simple basis for spectra (e.g., sinusoidal components)."""
    rng = np.random.default_rng(rng)
    wavelengths = np.linspace(400, 800, n_wavelengths)  # nm
    B = np.zeros((n_wavelengths, n_bases))
    for i in range(n_bases):
        freq = rng.uniform(0.01, 0.05)  # rad/nm
        phase = rng.uniform(0, 2*np.pi)
        B[:, i] = np.sin(freq * wavelengths + phase)
    return wavelengths, B

# --------------------
# Synthetic spectra
# --------------------
def generate_synthetic_spectra(B, n_samples=10, rng=None):
    """Generate synthetic spectra as linear combinations of basis vectors."""
    rng = np.random.default_rng(rng)
    n_bases = B.shape[1]
    coeffs = rng.normal(size=(n_samples, n_bases))
    S = coeffs @ B.T          # shape (n_samples, n_wavelengths)
    return S, coeffs

# --------------------
# Bandpasses
# --------------------
def gaussian_bandpass(wavelengths, center, sigma):
    """Return a Gaussian bandpass response."""
    return np.exp(-0.5 * ((wavelengths - center)/sigma)**2)

def build_bandpasses(wavelengths, centers, sigmas):
    """Build a list of bandpass responses."""
    return [gaussian_bandpass(wavelengths, c, s) for c, s in zip(centers, sigmas)]

# --------------------
# Photometric data generation
# --------------------
def generate_photometric_data(S, bandpasses):
    """
    Compute synthetic photometric fluxes for each spectrum.
    Flux in a band is the integral of spectrum * response divided by integral of response.
    """
    n_samples, n_wave = S.shape
    n_bands = len(bandpasses)
    P = np.zeros((n_samples, n_bands))
    for i, bp in enumerate(bandpasses):
        norm = integrate.trapz(bp, axis=-1)
        P[:, i] = integrate.trapz(S * bp, axis=1) / norm
    return P

# --------------------
# Reconstruction
# --------------------
def reconstruct_spectrum(photon, B, bandpasses):
    """
    Reconstruct spectrum from photometric fluxes for a single object.
    Solve linear least squares: (M @ B) c ≈ photon,
    where M are bandpass responses integrated over wavelength.
    """
    # Build measurement matrix M
    n_bands = len(bandpasses)
    n_wave = B.shape[0]
    M = np.array([bandpasses[i] / integrate.trapz(bandpasses[i]) for i in range(n_bands)])
    # Convert to 2D: shape (n_bands, n_wave)
    M = np.vstack(M)
    # Solve for coefficients c
    A = M @ B          # shape (n_bands, n_bases)
    coeff, *_ = np.linalg.lstsq(A, photon, rcond=None)
    # Reconstruct spectrum
    spectrum = B @ coeff
    return spectrum, coeff

# --------------------
# Example usage
# --------------------
if __name__ == "__main__":
    rng = 42
    # 1. Define spectral model
    wavelengths, B = generate_basis(n_wavelengths=300, n_bases=6, rng=rng)

    # 2. Generate synthetic spectra
    S, true_coeffs = generate_synthetic_spectra(B, n_samples=5, rng=rng)

    # 3. Define bandpasses (3 bands)
    centers = [450, 550, 650]   # nm
    sigmas  = [20, 30, 25]      # nm
    bandpasses = build_bandpasses(wavelengths, centers, sigmas)

    # 4. Generate photometric data
    P = generate_photometric_data(S, bandpasses)

    # 5. Reconstruct spectra for each object
    recon_spectra = []
    recon_coeffs = []
    for i in range(S.shape[0]):
        spectrum, coeff = reconstruct_spectrum(P[i], B, bandpasses)
        recon_spectra.append(spectrum)
        recon_coeffs.append(coeff)

    recon_spectra = np.array(recon_spectra)
    recon_coeffs = np.array(recon_coeffs)

    # Print comparison for first sample
    idx = 0
    print("True coefficients:", true_coeffs[idx])
    print("Reconstructed coefficients:", recon_coeffs[idx])
    print("Spectra error (L2 norm):", np.linalg.norm(S[idx] - recon_spectra[idx]))