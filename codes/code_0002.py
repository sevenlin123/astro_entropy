
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
Okay, I need to write a Python implementation based on the user's requirements. Let's break down what they asked for.

First, the algorithm needs to define a spectral model. I'll choose a polynomial model because it's straightforward and fits well for spectral analysis. Using something like a 5th-degree polynomial makes sense.

Next, I have to generate synthetic spectra. I'll create a wavelength array and compute the flux using the polynomial. Adding some noise will make it more realistic.

Then, I need to generate photometric data from these spectra. This involves integrating the flux over specific bands. I'll define a set of photometric bands as intervals and calculate the flux in each by integrating the synthetic spectrum within those intervals.

Now, for the reconstruction part. I'll use photometric data to fit a polynomial model. Here, I'll apply regularization with Lasso to handle potential overfitting. Training and testing sets will be used to evaluate the model's performance.

Putting it all together, I'll structure the code into functions: one to generate spectra, another to create photometric data, and a third for reconstruction. Using numpy for arrays, scipy for integration and regularization, and scikit-learn for the Lasso model ensures all constraints are met.

I'll make sure the code is clean, with clear function definitions, and that it runs without additional inputs. The synthetic example is self-contained, so no external data is needed. Finally, I'll test the code mentally to ensure each step works as intended.
</think>

```python
import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import Lasso

def generate_synthetic_spectra(wavelength_range, flux_degree, noise_level):
    """
    Generates synthetic spectral data with a polynomial model and added noise.
    """
    wavelength = np.linspace(*wavelength_range, 100)
    coefficients = np.random.rand(flux_degree + 1)
    flux = np.polyval(coefficients, wavelength)
    flux += noise_level * np.random.normal(size=len(wavelength))
    return wavelength, flux

def create_photometric_bands(wavelength, flux, bands):
    """
    Creates photometric measurements by integrating flux over specified bands.
    """
    photometry = []
    for band in bands:
        mask = (wavelength >= band[0]) & (wavelength <= band[1])
        integrated_flux = trapz(flux[mask], wavelength[mask])
        photometry.append(integrated_flux)
    return photometry

def reconstruct_spectral_model(photometry, bands, wavelength, alpha=0.1):
    """
    Reconstructs a spectral model using photometric data and Lasso regression.
    """
    X = np.array([np.polyval(np.vander(band, len(bands)), i) 
                  for band in bands for i in range(len(wavelength))])
    y = np.array(photometry)
    
    model = Lasso(alpha=alpha).fit(X, y)
    reconstructed_flux = model.predict(X)
    return wavelength, reconstructed_flux

if __name__ == "__main__":
    # Define parameters
    wavelength_range = (300, 700)
    flux_degree = 3
    noise_level = 0.1
    photometric_bands = [
        (350, 450),
        (450, 550),
        (550, 650),
        (650, 700)
    ]
    
    # Generate synthetic spectra
    wavelength, flux = generate_synthetic_spectra(wavelength_range, flux_degree, noise_level)
    
    # Create photometric measurements
    photometry = create_photometric_bands(wavelength, flux, photometric_bands)
    
    # Reconstruct spectral model from photometry
    reconstructed_wavelength, reconstructed_flux = reconstruct_spectral_model(
        photometry, photometric_bands, wavelength
    )
    
    # (Optional) Compare original and reconstructed spectra
```