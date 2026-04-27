import numpy as np
from scipy.signal import gaussian
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------
# Spectral model
# ----------------------------------------------------------------------
def generate_wavelength_grid(start=300, end=800, n_points=500):
    """Generate wavelength grid in nm."""
    return np.linspace(start, end, n_points)

def synthetic_spectrum(wavelengths, n_features=3, random_state=None):
    """
    Create a synthetic spectrum as sum of Gaussian features.
    Returns an array of flux values.
    """
    rng = np.random.default_rng(random_state)
    spectrum = np.zeros_like(wavelengths)
    for _ in range(n_features):
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform(5, 30)      # nm
        amplitude = rng.uniform(0.5, 1.5)
        gauss = amplitude * np.exp(-0.5 * ((wavelengths - center)/width)**2)
        spectrum += gauss
    # Add continuum
    continuum = 0.1 + 0.05 * rng.standard_normal()
    spectrum += continuum
    return spectrum

def generate_synthetic_spectra(n_samples, wavelengths, n_features=3, random_state=None):
    """Generate a dataset of synthetic spectra."""
    spectra = np.array([synthetic_spectrum(wavelengths, n_features=n_features,
                                          random_state=random_state + i)
                        for i in range(n_samples)])
    return spectra

# ----------------------------------------------------------------------
# Photometry
# ----------------------------------------------------------------------
def gaussian_bandpass(center, width, wavelengths):
    """Gaussian bandpass transmission curve."""
    trans = np.exp(-0.5 * ((wavelengths - center)/width)**2)
    return trans / trans.sum()  # normalize

def generate_bandpasses(n_bands, wavelengths, random_state=None):
    """Generate random bandpass transmission curves."""
    rng = np.random.default_rng(random_state)
    bandpasses = []
    for _ in range(n_bands):
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform(10, 50)
        trans = gaussian_bandpass(center, width, wavelengths)
        bandpasses.append(trans)
    return np.array(bandpasses)  # shape (n_bands, n_wavelengths)

def compute_photometry(spectra, bandpasses):
    """
    Compute photometric fluxes by integrating spectra over each bandpass.
    spectra: (n_samples, n_wavelengths)
    bandpasses: (n_bands, n_wavelengths)
    Returns: (n_samples, n_bands)
    """
    return spectra @ bandpasses.T

# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def train_pca_and_regression(X_phot, Y_spec, n_components=20):
    """
    Fit PCA on spectra and linear regression from photometry to PCA coeffs.
    Returns PCA object and regression model.
    """
    pca = PCA(n_components=n_components, svd_solver='randomized',
              random_state=42)
    Y_pca = pca.fit_transform(Y_spec)
    reg = LinearRegression()
    reg.fit(X_phot, Y_pca)
    return pca, reg

def reconstruct_spectrum(photometry, pca, reg):
    """
    Predict spectrum from photometry.
    """
    pca_coeffs = reg.predict(photometry.reshape(1, -1))
    spectrum = pca.inverse_transform(pca_coeffs)
    return spectrum.ravel()

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    wavelengths = generate_wavelength_grid()
    n_samples = 200
    spectra = generate_synthetic_spectra(n_samples, wavelengths,
                                         n_features=4,
                                         random_state=0)
    n_bands = 7
    bandpasses = generate_bandpasses(n_bands, wavelengths,
                                     random_state=1)
    photometry = compute_photometry(spectra, bandpasses)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        photometry, spectra, test_size=0.2, random_state=42)

    # Train models
    pca, reg = train_pca_and_regression(X_train, y_train,
                                        n_components=15)

    # Reconstruct test spectra
    reconstructions = np.array([reconstruct_spectrum(x, pca, reg)
                                for x in X_test])

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, reconstructions, multioutput='raw_values'))
    print(f"RMSE per wavelength: {rmse.mean():.4f}")