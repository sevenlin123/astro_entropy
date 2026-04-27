import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures

# 1. Define a simple spectral model
def generate_synthetic_spectrum(wavelengths, params):
    """
    Simple Gaussian mixture model for a spectrum.
    params: list of tuples (amplitude, mean, sigma)
    """
    spectrum = np.zeros_like(wavelengths)
    for amp, mu, sigma in params:
        spectrum += amp * np.exp(-(wavelengths - mu)**2 / (2 * sigma**2))
    return spectrum

# 2. Generate synthetic spectra
def create_synthetic_dataset(num_spectra, wavelengths):
    """
    Creates a dataset of synthetic spectra with random Gaussian components.
    """
    rng = np.random.default_rng()
    spectra = []
    true_params_list = []
    for _ in range(num_spectra):
        num_components = rng.integers(1, 4)  # 1 to 3 components
        params = []
        for _ in range(num_components):
            amp = rng.uniform(0.5, 2.0)
            mu = rng.uniform(wavelengths[0], wavelengths[-1])
            sigma = rng.uniform(10.0, 50.0)
            params.append((amp, mu, sigma))
        spectra.append(generate_synthetic_spectrum(wavelengths, params))
        true_params_list.append(params)
    return np.array(spectra), np.array(true_params_list)

# 3. Generate photometric data from synthetic spectra
def compute_photometry(spectra, wavelengths, filters):
    """
    Integrate spectra over filter transmission curves.
    filters: dict with keys as filter names and values as transmission arrays.
    """
    photometry = []
    for spec in spectra:
        mags = {}
        for name, trans in filters.items():
            flux = np.trapz(spec * trans, wavelengths)
            # Convert flux to magnitude-like value
            mag = -2.5 * np.log10(flux + 1e-12)
            mags[name] = mag
        photometry.append(mags)
    return photometry

# 4. Reconstruct a synthetic spectrum from photometric data
def reconstruct_from_photometry(photometry_row, wavelengths, filters):
    """
    Reconstruct spectrum using polynomial basis regression.
    """
    # Build design matrix for filter responses at each wavelength
    X = np.column_stack([filters[fn].reshape(-1,1) for fn in filters])
    # Since we have one sample per filter, use Ridge regression
    y = np.array([photometry_row[fn] for fn in filters])
    ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ridge.fit(X, y)
    # Predict flux at each wavelength by regressing onto polynomial features
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    flux_pred = ridge.predict(X_poly)
    return flux_pred

# Main execution
if __name__ == "__main__":
    # Wavelength range
    wav = np.linspace(4000, 7000, 300)
    # Create simple bandpasses
    def bandpass(shape, center, width):
        return np.exp(-0.5 * ((shape - center)/width)**2)
    # Define filters: U, B, V, R
    filter_names = ["U", "B", "V", "R"]
    filters = {name: bandpass(wav, center, 200) for name, center in zip(filter_names, [4200, 4700, 5500, 6300])}
    
    # Generate dataset
    spectra, params = create_synthetic_dataset(10, wav)
    
    # Compute photometry
    phot = compute_photometry(spectra, wav, filters)
    
    # Reconstruct one spectrum
    idx = 0
    recon_spec = reconstruct_from_photometry(phot[idx], wav, filters)
    
    print("True spectrum shape:", spectra[idx].shape)
    print("Reconstructed spectrum shape:", recon_spec.shape)