import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Spectral model definition
# -----------------------------
def generate_basis_spectra(n_bases, wavelengths):
    """Generate n_bases basis spectra as Gaussian bumps."""
    np.random.seed(0)
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_bases)
    widths = (wavelengths.max() - wavelengths.min()) / (4 * n_bases)
    bases = []
    for c in centers:
        spectrum = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        bases.append(spectrum)
    return np.array(bases)  # shape (n_bases, n_wavelengths)

# -----------------------------
# 2. Synthetic spectra generation
# -----------------------------
def synthesize_spectrum(coeffs, bases):
    """Linear combination of basis spectra."""
    return coeffs @ bases  # shape (n_wavelengths,)

def add_noise(spectrum, snr=20):
    """Add Gaussian noise based on specified signal-to-noise ratio."""
    signal_power = np.mean(spectrum**2)
    noise_std = np.sqrt(signal_power) / snr
    noise = np.random.normal(0, noise_std, size=spectrum.shape)
    return spectrum + noise

# -----------------------------
# 3. Photometric data generation
# -----------------------------
def generate_bandpasses(n_bands, wavelengths):
    """Create Gaussian bandpasses with random centers and fixed width."""
    np.random.seed(1)
    centers = np.linspace(wavelengths.min()+20, wavelengths.max()-20, n_bands)
    width = 30.0  # nm
    bandpasses = []
    for c in centers:
        trans = np.exp(-0.5 * ((wavelengths - c) / width)**2)
        bandpasses.append(trans)
    return np.array(bandpasses)  # shape (n_bands, n_wavelengths)

def compute_photometry(spectrum, bandpasses, wavelengths):
    """Integrate spectrum times bandpass over wavelengths."""
    phot = []
    for trans in bandpasses:
        flux = simps(spectrum * trans, wavelengths)
        phot.append(flux)
    return np.array(phot)  # shape (n_bands,)

# -----------------------------
# 4. Spectrum reconstruction
# -----------------------------
def construct_forward_matrix(bases, bandpasses, wavelengths):
    """Forward matrix mapping coefficients to photometric fluxes."""
    A = []
    for trans in bandpasses:
        row = [simps(base * trans, wavelengths) for base in bases]
        A.append(row)
    return np.array(A)  # shape (n_bands, n_bases)

def reconstruct_coeffs(photometry, A):
    """Solve least-squares problem to recover coefficients."""
    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, photometry)
    return lr.coef_  # shape (n_bases,)

def reconstruct_spectrum(coeffs, bases):
    """Reconstruct spectrum from coefficients and basis set."""
    return coeffs @ bases  # shape (n_wavelengths,)

# -----------------------------
# 5. Demo: end-to-end pipeline
# -----------------------------
if __name__ == "__main__":
    # Setup
    n_wavelengths = 100
    wavelengths = np.linspace(400.0, 800.0, n_wavelengths)  # nm
    n_bases = 5
    n_bands = 7
    n_samples = 10

    # Generate basis spectra
    bases = generate_basis_spectra(n_bases, wavelengths)

    # Generate bandpasses
    bandpasses = generate_bandpasses(n_bands, wavelengths)

    # Forward matrix for reconstruction
    A = construct_forward_matrix(bases, bandpasses, wavelengths)

    # Storage for results
    recon_errors = []

    for i in range(n_samples):
        # Random coefficients
        true_coeffs = np.random.uniform(-1, 1, size=n_bases)

        # Synthetic spectrum
        spectrum = synthesize_spectrum(true_coeffs, bases)
        noisy_spectrum = add_noise(spectrum, snr=30)

        # Photometry from noisy spectrum
        photometry = compute_photometry(noisy_spectrum, bandpasses, wavelengths)

        # Reconstruct coefficients from photometry
        rec_coeffs = reconstruct_coeffs(photometry, A)

        # Reconstruct spectrum
        rec_spectrum = reconstruct_spectrum(rec_coeffs, bases)

        # Compute reconstruction error (RMSE)
        rmse = np.sqrt(np.mean((spectrum - rec_spectrum)**2))
        recon_errors.append(rmse)

        print(f"Sample {i+1}: RMSE={rmse:.4f}")

    mean_rmse = np.mean(recon_errors)
    print(f"\nAverage RMSE over {n_samples} samples: {mean_rmse:.4f}")