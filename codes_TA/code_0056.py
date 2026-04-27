import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: linear combination of Gaussian basis functions
def gaussian_basis(wavelengths, centers, widths):
    """Generate Gaussian basis functions."""
    return np.exp(-0.5 * ((wavelengths[:, None] - centers) / widths)**2)

def build_spectrum(basis, coeffs):
    """Construct a spectrum from basis coefficients."""
    return basis @ coeffs

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples, wavelengths):
    np.random.seed(42)
    # Random centers and widths for basis
    n_basis = 5
    centers = np.linspace(4000, 8000, n_basis)
    widths = np.full(n_basis, 200)
    basis = gaussian_basis(wavelengths, centers, widths)

    # Random coefficient matrix (n_samples x n_basis)
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = basis @ coeffs.T  # shape (n_wavelengths, n_samples)
    return spectra, coeffs, basis

# 3. Generate photometric data from synthetic spectra
def compute_photometry(spectra, wavelengths, bandpass_centers, bandwidth=500):
    """Simulate photometry by integrating spectrum over bandpasses."""
    n_bands = len(bandpass_centers)
    n_samp = spectra.shape[1]
    photometry = np.zeros((n_bands, n_samp))

    for i, center in enumerate(bandpass_centers):
        mask = (wavelengths >= center - bandwidth/2) & (wavelengths <= center + bandwidth/2)
        # Simple box filter
        integrand = spectra[mask, :]
        phot = simps(integrand, wavelengths[mask], axis=0)
        photometry[i, :] = phot
    return photometry

# 4. reconstruct a synthetic spectrum from photometric data
def reconstruct_spectrum(photometry, bandpass_centers, wavelengths, n_basis=10):
    """
    Recover spectral coefficients using Ridge regression.
    Each basis function is integrated over bandpasses to form design matrix.
    """
    # Build design matrix for photometry
    n_bands = len(bsi = bandpass_centers)
    n_wav = wavelengths.size

    # Create Gaussian basis functions (we're using n_basis)
    w = np.arange(0, maybe? (???) )
We need to produce runnable python code with clear functions: define spectral model, generate synthetic spectra, generate photometric data, reconstruct a synthetic spectrum from photometry. Use numpy, scipy, sklearn. Provide a script. The earlier attempt started but got messed up. Let's craft clean code.

We need:

- define a spectral model: maybe use Gaussian basis functions.

Functions:

- gaussian_basis(wavelengths, centers, widths): returns basis matrix.

- build_spectrum(basis, coeffs): returns spectrum.

- generate_synthetic_spectra(n_samples, wavelengths): generate random coefficients, basis from some predefined centers and widths.

- compute_photometry(spectra, wavelengths, bandpass_centers, bandwidth): integrate.

- reconstruct_spectrum(photometry, bandpass_centers, wavelengths, n_basis): reconstruct via linear regression (Ridge). We'll need to compute design matrix for photometry given basis. For each band, integrate each basis over that band. So we create basis_for_band: integrate basis over band. Then we have matrix M of shape (n_bands, n_basis). Photometry = M @ coeffs. We can solve via ridge: coeffs_est = ridge.fit(M, photometry).predict? Actually we can compute coefficients using ridge regression on per sample? Wait: we want to reconstruct each spectrum from photometry. Suppose we have photometry vector p for one sample. We can solve linear system p = M @ coeffs. We can invert with regularization: use Ridge. For each sample we can solve: coeffs_est = ridge.fit(M, p). But Ridge expects multiple samples. Instead we can use linear algebra: coeffs_est = np.linalg.lstsq(M.T @ M + alpha*I, M.T @ p)[0]. Or we can use Ridge with X=M, y=p across many samples. But if we have many spectra, we can fit ridge to predict photometry from coefficients? Actually we have known basis and photometry; we want to recover coefficients. This is typical inverse problem. So we can solve for each sample individually using ridge.

Simpler: Use numpy's lstsq with regularization: compute pseudo-inverse with lambda. Let's implement reconstruction function that for each photometry sample solves: coeffs_est = np.linalg.pinv(M.T @ M + reg*np.eye(n_basis)) @ M.T @ p. That yields coefficients. Then we can rebuild spectrum using basis. We'll produce reconstructed_spectra.

Ok.

Let's produce full script: generate wavelengths array e.g., 4000-8000 Angstroms, 1000 points. Generate synthetic spectra 20 samples. Bandpass centers maybe 5 photometric bands. Reconstruct spectra for first sample.

Let's code.

Also we may want to compare original vs reconstructed.

Return only code. Ensure no syntax errors. Use proper imports.

Let's craft code now.assistantfinal code