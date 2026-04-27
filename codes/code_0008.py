
Write a minimal Python implementation of the following algorithm.

Requirements:

- implement only the core reconstruction framework
- ignore validation methods (LOOCV, cross-validation)
- Do not need plotting
- generate synthetic example input data inside the script
- Do NOT show reasoning or thinking.
- Do NOT explain the code.
- Return only the Python code.

The code should:

1. define a spectral model
2. generate synthetic spectra
2. generate photometric data from synthetic spectra
3. reconstruct a synthetic spectrum from photometruic

Constraints:

- use numpy, scipy, and scikit-learn
- produce runnable Python code
- define clear functions

Return only the code.
Okay, I need to write a Python implementation based on the user's requirements. Let me go through each step.

First, the user wants a minimal implementation of an algorithm. The steps are: define a spectral model, generate synthetic spectra, create photometric data from those spectra, and then reconstruct the spectrum from photometry.

I'll start by importing the necessary libraries: numpy for arrays, scipy for numerical computations, and sklearn for any machine learning parts, like regression.

Next, I'll define a spectral model. A simple approach is to create a function that generates a spectrum based on a set of parameters. For example, using Gaussian components with different means and variances.

Then, I'll generate synthetic spectra. I'll create a grid of wavelengths and, for each, compute the flux using the spectral model. Adding some noise will make it more realistic.

For photometric data, I'll define a set of bandpasses. Each bandpass can be a transmission curve. To get observed magnitudes, I'll integrate the spectrum over each bandpass. I'll use numpy's trapz for integration and then convert flux to magnitudes using a reference zero point.

Now, to reconstruct the spectrum from photometry. I'll set up a regression problem where the features are the photometric magnitudes, and the target is the flux in each wavelength bin. Using Ridge Regression from sklearn, I'll train the model and then predict the fluxes.

Finally, I'll plot the original and reconstructed spectra to compare them, but since the user mentioned not to include plotting, I'll just note that it's part of the code.

Putting it all together, I'll structure the code into functions: one for the spectral model, another for generating spectra, one for photometric data, and another for reconstruction. The main block will tie everything together, run the code, and print the results.

I think that covers all the steps. Now, I'll write the code accordingly.
</think>

```python
import numpy as np
from scipy import ndimage
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, params):
    """Simple spectral model using Gaussian components."""
    flux = np.zeros_like(wavelengths)
    for param in params:
        amplitude, mean, std = param
        flux += amplitude * np.exp(-0.5 * ((wavelengths - mean) / std)**2)
    return flux

def generate_synthetic_spectra(wavelengths, n_spectra=1, params_mean=[[1, 550, 10]], 
                               params_std=[[0.8, 12, 3]]):
    """Generate synthetic spectra with given parameters."""
    spectra = []
    for _ in range(n_spectra):
        params = [tuple(p + np.random.normal(0, s) for p, s in zip(mean, std)) 
                  for mean, std in zip(params_mean, params_std)]
        flux = spectral_model(wavelengths, params)
        flux += np.random.normal(0, 0.1, len(wavelengths))  # Add noise
        spectra.append(flux)
    return np.array(spectra)

def photometric_data(wavelengths, flux, bandpasses):
    """Convert spectra to photometric magnitudes."""
    photometry = []
    for band in bandpasses:
        trans, wave = band
        mag = -2.5 * np.log10(np.trapz(flux * trans, wave)) + 0  # Zero-point magnitude
        photometry.append(mag)
    return photometry

def reconstruct_spectrum(photometry, wavelengths, bandpasses, alpha=0.1):
    """Reconstruct spectrum from photometric data using Ridge Regression."""
    n_bands = len(photometry)
    n_wavelengths = len(wavelengths)
    
    X = np.zeros((n_wavelengths, n_bands))
    for i, (trans, wave) in enumerate(bandpasses):
        flux = np.interp(wavelengths, wave, trans)
        X[:, i] = flux
        
    y = photometry
    clf = Ridge(alpha=alpha).fit(X, y)
    flux_reconstructed = np.dot(clf.coef_, X.T) + clf.intercept_
    return flux_reconstructed

# Main execution
if __name__ == "__main__":
    np.random.seed(42)
    
    # Define parameters
    wavelengths = np.linspace(400, 700, 300)  # nm
    n_spectra = 1
    
    # Generate synthetic spectra
    params_mean = [[1, 550, 10], [0.8, 600, 15]]
    params_std = [[0.2, 5, 5], [0.1, 10, 3]]
    synthetic_spectra = generate_synthetic_spectra(wavelengths, n_spectra, params_mean, params_std)
    
    # Define bandpasses (e.g., filters)
    bandpasses = [
        (np.array([0.1, 0.3]), np.array([400, 400])),  # Simple filter example
        (np.array([0.7, 0.9]), np.array([650, 650])),
        (np.array([0.3, 0.5]), np.array([550, 550]))
    ]
    
    # Convert synthetic spectra to photometric data
    photometry = photometric_data(wavelengths, synthetic_spectra[0], bandpasses)
    
    # Reconstruct spectrum from photometry
    flux_reconstructed = reconstruct_spectrum(photometry, wavelengths, bandpasses)
    
    print("Synthetic spectrum:", synthetic_spectra[0])
    print("Photometry:", photometry)
    print("Reconstructed flux:", flux_reconstructed)
```