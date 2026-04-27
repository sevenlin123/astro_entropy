import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def gaussian(wave, amp, cen, width):
    """Simple Gaussian line."""
    return amp * np.exp(-0.5 * ((wave - cen) / width) ** 2)

def spectral_model(wave, amps, cents, widths):
    """Sum of Gaussians."""
    spec = np.zeros_like(wave)
    for amp, cen, wid in zip(amps, cents, widths):
        spec += gaussian(wave, amp, cen, wid)
    return spec

# ---------- Basis functions ----------
def generate_basis(wave, num=5):
    """Create a set of Gaussian basis functions."""
    cents = np.linspace(wave.min(), wave.max(), num)
    widths = np.full(num, (wave.max() - wave.min()) / (num * 4))
    basis = [gaussian(wave, 1.0, cen, wid) for cen, wid in zip(cents, widths)]
    return np.array(basis), cents, widths

# ---------- Synthetic spectra ----------
def generate_synthetic_spectra(n, wave, basis, noise_std=0.05):
    """Generate n synthetic spectra with random amplitudes."""
    num_basis = basis.shape[0]
    amps = np.random.uniform(0.5, 1.5, size=(n, num_basis))
    spectra = []
    for i in range(n):
        spec = np.dot(amps[i], basis)
        spec += np.random.normal(scale=noise_std, size=wave.shape)
        spectra.append(spec)
    return np.array(spectra), amps

# ---------- Filters ----------
def top_hat_filter(wave, center, width):
    """Top‑hat filter centered at 'center' with half‑width 'width'."""
    return ((np.abs(wave - center) <= width / 2)).astype(float)

def generate_filters(wave, centers, widths):
    """Generate filter transmission curves."""
    return np.array([top_hat_filter(wave, c, w) for c, w in zip(centers, widths)])

# ---------- Photometry ----------
def compute_photometry(spectra, wave, filters):
    """Integrate spectra through filters."""
    n_spec = spectra.shape[0]
    n_filt = filters.shape[0]
    phot = np.zeros((n_spec, n_filt))
    for i in range(n_filt):
        filt = filters[i]
        norm = simps(filt, wave)
        for j in range(n_spec):
            phot[j, i] = simps(spectra[j] * filt, wave) / norm
    return phot

# ---------- Reconstruction ----------
def reconstruct_spectrum(photometry, filters, basis, wave):
    """
    Reconstruct spectral coefficients via linear least squares.
    Returns the coefficient vector and the reconstructed spectra.
    """
    n_filt = filters.shape[0]
    # Build matrix A where A[i,j] = <basis[j], filt[i]> / <filt[i]>
    A = np.zeros((n_filt, basis.shape[0]))
    for i in range(n_filt):
        filt = filters[i]
        denom = simps(filt, wave)
        for j in range(basis.shape[0]):
            A[i, j] = simps(basis[j] * filt, wave) / denom

    # Fit each spectrum's photometry
    coeffs = np.linalg.lstsq(A, photometry.T, rcond=None)[0].T  # shape (n_spec, n_basis)
    # Reconstruct spectra
    recon = np.dot(coeffs, basis)
    return coeffs, recon

# ---------- Main ----------
def main():
    # Wavelength grid
    wave = np.linspace(400, 800, 1000)  # nm

    # Basis functions
    basis, cent_basis, width_basis = generate_basis(wave, num=5)

    # Generate synthetic spectra
    n_spectra = 10
    spectra, true_amps = generate_synthetic_spectra(n_spectra, wave, basis, noise_std=0.02)

    # Filter definitions
    filt_centers = [450, 550, 650]  # nm
    filt_widths   = [50, 50, 50]    # nm
    filters = generate_filters(wave, filt_centers, filt_widths)

    # Photometric observations
    phot = compute_photometry(spectra, wave, filters)

    # Reconstruct spectra
    coeffs, recon_spectra = reconstruct_spectrum(phot, filters, basis, wave)

    # Example output
    print("True coefficients (first spectrum):", true_amps[0])
    print("Recovered coefficients (first spectrum):", coeffs[0])
    print("Reconstruction error (RMS, first spectrum):",
          np.sqrt(np.mean((spectra[0] - recon_spectra[0]) ** 2)))

if __name__ == "__main__":
    main()