import numpy as np
from scipy import integrate

# ---------- Spectral model ----------
def spectral_basis(n_basis, wavelengths):
    """Generate orthogonal polynomial basis functions (Chebyshev)."""
    # Normalize wavelengths to [-1, 1]
    x = 2 * (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min()) - 1
    basis = np.polynomial.chebyshev.chebvander(x, n_basis - 1)
    return basis  # shape (N_wave, n_basis)

def generate_synthetic_spectrum(coeffs, basis):
    """Compute synthetic spectrum as linear combination of basis."""
    return basis @ coeffs  # shape (N_wave,)

# ---------- Photometry ----------
def bandpass_response(wavelengths, center, width):
    """Gaussian bandpass response."""
    resp = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    return resp / resp.sum()  # normalise

def compute_photometry(spectrum, wavelengths, band_centers, band_widths):
    """Integrate spectrum over each bandpass to get photometric fluxes."""
    phot = []
    for c, w in zip(band_centers, band_widths):
        resp = bandpass_response(wavelengths, c, w)
        phot.append(integrate.trapz(spectrum * resp, wavelengths))
    return np.array(phot)  # shape (n_bands,)

# ---------- Forward model ----------
def forward_matrix(basis, wavelengths, band_centers, band_widths):
    """Matrix that maps basis coefficients to photometric measurements."""
    n_bands, n_basis = len(band_centers), basis.shape[1]
    A = np.zeros((n_bands, n_basis))
    for i, (c, w) in enumerate(zip(band_centers, band_widths)):
        resp = bandpass_response(wavelengths, c, w)
        A[i, :] = integrate.trapz(basis * resp[:, None], wavelengths, axis=0)
    return A  # shape (n_bands, n_basis)

# ---------- Reconstruction ----------
def reconstruct_coeffs(photometry, forward_mat):
    """Least‑squares fit to recover basis coefficients."""
    coeffs, *_ = np.linalg.lstsq(forward_mat, photometry, rcond=None)
    return coeffs

def reconstruct_spectrum(coeffs, basis):
    """Reconstruct spectrum from estimated coefficients."""
    return basis @ coeffs

# ---------- Main script ----------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(400, 1000, 500)  # nm

    # Basis and synthetic spectrum
    n_basis = 5
    basis = spectral_basis(n_basis, wav)
    true_coeffs = np.random.randn(n_basis)
    true_spectrum = generate_synthetic_spectrum(true_coeffs, basis)

    # Photometric bands
    band_centers = np.array([450, 550, 650])  # nm
    band_widths  = np.array([20, 20, 20])     # nm

    # Simulate photometry
    phot_obs = compute_photometry(true_spectrum, wav, band_centers, band_widths)

    # Forward matrix and reconstruction
    A = forward_matrix(basis, wav, band_centers, band_widths)
    est_coeffs = reconstruct_coeffs(phot_obs, A)
    recon_spectrum = reconstruct_spectrum(est_coeffs, basis)

    # Evaluate
    mse = np.mean((true_spectrum - recon_spectrum) ** 2)
    print(f"True coeffs:   {true_coeffs}")
    print(f"Estimated coeffs: {est_coeffs}")
    print(f"Mean squared error in spectrum: {mse:.6f}")