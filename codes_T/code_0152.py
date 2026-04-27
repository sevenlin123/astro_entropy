import numpy as np
from sklearn.linear_model import Ridge

# ----- Spectral model -----
def make_wavelength_grid(Nlambda=500):
    """Create a logarithmically spaced wavelength grid from 400 to 2500 nm."""
    return np.logspace(np.log10(400), np.log10(2500), Nlambda)

# ----- Synthetic spectra generation -----
def generate_basis_spectra(Nbasis, Nlambda):
    """Generate a set of basis spectra (Gaussian peaks at random centers)."""
    wav = make_wavelength_grid(Nlambda)
    basis = []
    rng = np.random.default_rng(seed=42)
    for _ in range(Nbasis):
        center = rng.uniform(wav[0], wav[-1])
        width = rng.uniform(50, 200)
        amp   = rng.uniform(0.5, 1.5)
        spec = amp * np.exp(-0.5 * ((wav - center) / width)**2)
        basis.append(spec)
    return np.array(basis)  # shape (Nbasis, Nlambda)

def generate_synthetic_spectra(n_samples, basis_spectra, noise_level=0.02):
    """Produce spectra as random linear combos of basis spectra plus noise."""
    rng = np.random.default_rng()
    coeffs = rng.uniform(0.0, 1.0, size=(n_samples, basis_spectra.shape[0]))
    spectra = coeffs @ basis_spectra
    noise = rng.normal(scale=noise_level, size=spectra.shape)
    return spectra + noise

# ----- Filter curve generation -----
def generate_filter_curves(Nfilters, Nlambda):
    """Generate simple Gaussian filter curves at evenly spaced wavelengths."""
    wav = make_wavelength_grid(Nlambda)
    filters = []
    rng = np.random.default_rng(seed=7)
    for i in range(Nfilters):
        center = np.interp(i, [0, Nfilters-1], [wav[0], wav[-1]])
        width  = rng.uniform(20, 80)
        filt   = np.exp(-0.5 * ((wav - center) / width)**2)
        filt   /= filt.sum()  # normalize
        filters.append(filt)
    return np.array(filters)  # shape (Nfilters, Nlambda)

# ----- Photometry computation -----
def compute_photometry(spectra, filters):
    """Simulate photometric measurements as inner products of spectra with filters."""
    return spectra @ filters.T  # (nsamples, Nfilters)

# ----- Spectrum reconstruction -----
def reconstruct_spectrum(photometric, filters, alpha=1e-3):
    """
    Reconstruct a spectrum from its photometric measurements.
    
    Parameters
    ----------
    photometric : array, shape (Nfilters,)
        Measured photometric fluxes.
    filters : array, shape (Nfilters, Nlambda)
        Filter response functions.
    alpha : float
        Regularization strength for Ridge regression.
        
    Returns
    -------
    spectrum : array, shape (Nlambda,)
        Reconstructed spectrum.
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False, positive=True, solver='auto')
    ridge.fit(filters, np.eye(filters.shape[0]))  # fit filters as basis
    spectrum = ridge.predict(photometric.reshape(1, -1)).ravel()
    return spectrum

# ----- Main demonstration -----
if __name__ == "__main__":
    Nlambda   = 500
    Nbasis    = 5
    nsamples  = 100
    Nfilters  = 7
    
    # Build model components
    basis_spectra = generate_basis_spectra(Nbasis, Nlambda)
    spectra       = generate_synthetic_spectra(nsamples, basis_spectra)
    filters       = generate_filter_curves(Nfilters, Nlambda)
    
    # Simulate photometry
    photometry = compute_photometry(spectra, filters)
    
    # Reconstruct a single spectrum (here, the first one)
    target_phot = photometry[0]
    reconstructed = reconstruct_spectrum(target_phot, filters)
    
    # Simple sanity check
    original = spectra[0]
    err = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
    print(f"Relative reconstruction error: {err:.4f}")