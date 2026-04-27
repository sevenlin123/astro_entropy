import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------------------------------------------
# Spectral model – simple sum of Gaussian lines
# -------------------------------------------------------------------
def generate_spectrum(wavelength, params):
    """
    Generate a synthetic spectrum as a sum of Gaussian components.
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelength grid (Angstrom).
    params : list of tuples
        Each tuple contains (amplitude, center, width) for a Gaussian.
        
    Returns
    -------
    flux : ndarray
        Flux values on the wavelength grid.
    """
    flux = np.zeros_like(wavelength)
    for amp, cen, wid in params:
        flux += amp * np.exp(-(wavelength - cen)**2 / (2 * wid**2))
    return flux

def sample_random_spectrum(wavelength, n_components=5):
    """
    Sample random parameters for the spectrum.
    """
    params = []
    for _ in range(n_components):
        amp  = np.random.uniform(0.5, 3.0)
        cen  = np.random.uniform(wavelength.min() + 50, wavelength.max() - 50)
        wid  = np.random.uniform(10, 30)
        params.append((amp, cen, wid))
    return generate_spectrum(wavelength, params)

# -------------------------------------------------------------------
# Photometric filter definitions
# -------------------------------------------------------------------
def make_tophat_filter(wavelength, center, width):
    """Simple top–hat filter transmission."""
    return np.where(np.abs(wavelength - center) <= width/2.0, 1.0, 0.0)

def make_gaussian_filter(wavelength, center, fwhm):
    """Gaussian filter transmission."""
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    return np.exp(-(wavelength - center)**2 / (2*sigma**2))

# -------------------------------------------------------------------
# Photometry calculation
# -------------------------------------------------------------------
def compute_photometry(spectrum, wavelength, filters):
    """
    Integrate a spectrum through each filter to obtain photometric
    fluxes (equivalent magnitudes).
    
    Parameters
    ----------
    spectrum : ndarray
        Flux array.
    wavelength : ndarray
        Wavelength grid.
    filters : list of ndarray
        Transmission curves sampled on the same wavelength grid.
    
    Returns
    -------
    phot : ndarray
        Photometric fluxes for each filter.
    """
    phot = np.array([simps(spectrum * filt, wavelength) for filt in filters])
    return phot

# -------------------------------------------------------------------
# Reconstruction framework
# -------------------------------------------------------------------
class SpectrumReconstructor:
    """
    Reconstructs spectra from photometry using linear regression.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.regressors = None
        
    def fit(self, photometry, spectra):
        """
        Fit a separate linear regression for each wavelength point.
        """
        # Transpose spectra to shape (n_samples, n_wavelength)
        X = photometry
        Y = spectra.T   # (n_wavelength, n_samples).T -> (n_samples, n_wavelength)
        self.regressors = [LinearRegression().fit(X, Y[:, i]) for i in range(Y.shape[1])]
        
    def predict(self, photometry):
        """
        Predict spectra from photometry.
        """
        preds = np.column_stack([regressor.predict(photometry) for regressor in self.regressors])
        return preds

# -------------------------------------------------------------------
# Synthetic dataset generation
# -------------------------------------------------------------------
np.random.seed(42)

# Wavelength grid
lam = np.linspace(4000, 8000, 2000)  # Angstrom

# Filters (e.g., u,g,r,i,z bands)
filters = [
    make_tophat_filter(lam, 3500, 600),   # u band
    make_tophat_filter(lam, 4750, 500),   # g band
    make_tophat_filter(lam, 6200, 500),   # r band
    make_tophat_filter(lam, 7600, 500),   # i band
    make_tophat_filter(lam, 9100, 500)    # z band
]

# Generate training spectra
n_train = 200
train_spectra = np.array([sample_random_spectrum(lam) for _ in range(n_train)])
train_phot = np.array([compute_photometry(flux, lam, filters) for flux in train_spectra])

# Generate test spectra
n_test = 20
test_spectra = np.array([sample_random_spectrum(lam) for _ in range(n_test)])
test_phot = np.array([compute_photometry(flux, lam, filters) for flux in test_spectra])

# -------------------------------------------------------------------
# Training and evaluation
# -------------------------------------------------------------------
recon = SpectrumReconstructor()
recon.fit(train_phot, train_spectra)

predicted = recon.predict(test_phot)

mse = mean_squared_error(test_spectra, predicted)
print(f"Mean squared reconstruction error on test set: {mse:.4f}")

# -------------------------------------------------------------------
# Example reconstruction plot (optional)
# -------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    idx = np.random.randint(n_test)
    plt.figure(figsize=(8,4))
    plt.plot(lam, test_spectra[idx], label='True Spectrum')
    plt.plot(lam, predicted[idx], label='Reconstructed Spectrum', linestyle='--')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Spectrum Reconstruction Example')
    plt.legend()
    plt.tight_layout()
    plt.show()
except ImportError:
    pass  # Matplotlib not available; skip plotting