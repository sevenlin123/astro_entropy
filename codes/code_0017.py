
Write a minimal Python implementation of the following algorithm.

Requirements:

- implement only the core reconstruction framework
- ignore validation methods (LOOCV, cross-validation)
- need plotting
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
Okay, so I need to write a minimal Python implementation of an algorithm that does spectral model reconstruction. The user has given specific requirements, so I'll go through each step carefully.

First, the code needs to define a spectral model. I'll probably create a function that generates synthetic spectral data. Maybe using a Gaussian or some polynomial to represent the spectrum.

Next, I need to generate synthetic spectra. That means applying some noise and perhaps a baseline shift to make it realistic. Using numpy for this part makes sense.

Then, I have to generate photometric data from these synthetic spectra. Since it's synthetic, I can create bandpasses and integrate the spectra over these bands to get magnitudes. I'll need to define a few bandpasses, like maybe u, g, r bands, and then compute the flux in each.

After that, the main task is to reconstruct a synthetic spectrum from photometric data. This involves inverting the problem: given the magnitudes, recover the underlying spectrum. I'll set up an optimization problem where I minimize the difference between observed photometry and the model's prediction.

I'll use scipy's least squares solver for this. The model will be a polynomial fit to the spectrum, and I'll use the photometric data as constraints. The regularization term will help stabilize the solution, preventing overfitting.

Plotting is required, so I'll use matplotlib to visualize both the original and reconstructed spectra, along with the error bars for photometric data.

I need to structure the code into clear functions: one for generating spectra, another for computing photometry, and another for the reconstruction. Each function should do one thing and be well-defined.

I should also ensure that the code is runnable without any additional inputs. All data will be generated within the script.

Finally, I'll test the code mentally: create a synthetic spectrum, add noise and baseline, compute photometric data, then use optimization to reconstruct the spectrum. The plots should show the original and reconstructed spectra, along with the photometric points.

I think that covers all the steps. Now, putting it all together in code.
</think>

```python
import numpy as np
from scipy.optimize import leastsq
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def spectral_model(x, params):
    return np.exp(-x**2 / (2 * params[0]**2)) + params[1]

def generate_synthetic_spectra(n_spectra=1000, n_wavelengths=500):
    wavelengths = np.linspace(300, 700, n_wavelengths)
    true_params = np.random.normal([10, 0.5], [2, 0.1], size=(n_spectra, 2))
    spectra = np.zeros((n_spectra, n_wavelengths))
    for i in range(n_spectra):
        spectra[i] = spectral_model(wavelengths, true_params[i]) + np.random.normal(0, 0.1, n_wavelengths)
    return spectra, true_params

def compute_photometry(spectra, bandpasses):
    photometry = []
    for band in bandpasses:
        flux = np.trapz(spectra * band['response'], band['wavelength'])
        photometry.append(flux)
    photometry = np.array(photometry)
    return photometry

def spectrum_reconstructor(wavelengths, bandpasses):
    poly_features = PolynomialFeatures(degree=5)
    X = poly_features.fit_transform(wavelengths[:, None])
    
    def residuals(params, y_true, X):
        y_pred = np.dot(X, params)
        return y_true - y_pred
    
    def objective(params, y_true, X):
        return np.sum((y_true - np.dot(X, params))**2) + 0.1*np.sum(params**2)
    
    phot_data = compute_photometry(spectra[i:i+1], bandpasses)
    initial_params = np.zeros(len(params))
    result = leastsq(objective, initial_params, args=(phot_data.flatten(), X))
    reconstructed = np.dot(X, result.x)
    return reconstructed

def main():
    bandpasses = [
        {'wavelength': np.linspace(350, 450, 100), 'response': np.ones(100)},
        {'wavelength': np.linspace(450, 550, 100), 'response': np.ones(100)},
        {'wavelength': np.linspace(550, 650, 100), 'response': np.ones(100)}
    ]
    
    spectra, true_params = generate_synthetic_spectra()
    
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, spectra[0], label='True Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('Synthetic Spectrum')
    plt.legend()
    plt.show()
    
    photometric_data = compute_photometry(spectra, bandpasses)
    
    plt.figure(figsize=(8, 6))
    for i, band in enumerate(bandpasses):
        plt.errorbar(band['wavelength'][::10], photometric_data[:, i][::10], 
                    yerr=np.zeros_like(photometric_data[:, i][::10]), 
                    fmt='o', label=f'Band {i+1}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Photometric Flux')
    plt.title('Generated Photometric Data')
    plt.legend()
    plt.show()
    
    reconstructed_spectrum = spectrum_reconstructor(wavelengths, bandpasses)
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, spectra[0], label='True Spectrum')
    plt.plot(wavelengths, reconstructed_spectrum, label='Reconstructed Spectrum', linestyle='--')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('Spectrum Reconstruction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```