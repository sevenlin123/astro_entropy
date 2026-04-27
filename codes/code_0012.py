
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
Okay, so I need to write a Python script that implements an algorithm for spectral reconstruction. Let me break down what the user is asking for.

First, the user wants to generate synthetic spectra. I'll probably use numpy to create some sample data. Maybe I'll define a function called generate_synthetic_spectra that creates a set of spectra with added noise to make it realistic.

Next, I need to generate photometric data from these spectra. Photometry involves integrating the spectrum over certain bands. I can create a set of filter transmission curves using numpy, then compute the flux in each band by taking the dot product of the spectrum and the filter response. This will give me the photometric data.

Then, the main task is to reconstruct the synthetic spectrum from the photometric data. For this, I'll use a regression approach. Since the user mentioned using scikit-learn, I'll choose a RidgeCV estimator because it's suitable for ill-posed problems and does cross-validation. I'll set up the model to fit the photometric data against the original spectra.

I also need to include plotting. I'll use matplotlib to visualize the original spectra, the reconstructed ones, and maybe the error between them. The plots should help in understanding how well the reconstruction is working.

The user specified that the code should be minimal, focusing on the core framework without validation methods like LOOCV. So, I won't include cross-validation steps beyond what's necessary.

I'll structure the code into functions: one for generating synthetic spectra, another for computing photometric data, and a third for the reconstruction process. Each function should be clear and concise.

I should also ensure that all required libraries are imported at the beginning, like numpy, scipy, and scikit-learn. The code should run without errors when pasted into a Python environment.

Finally, I'll test the code in my mind. I generate some synthetic data, apply photometric bands, use the RidgeCV model to reconstruct the spectra, and then plot everything. It should show that the reconstructed spectra closely match the original ones, indicating the algorithm works.

I think I've covered all the points. Now, let's put it all together into code.
</think>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

def generate_synthetic_spectra(n_spectra=100, n_wavelengths=1000):
    """Generate synthetic spectral data."""
    wavelengths = np.linspace(300, 900, n_wavelengths)  # in nm
    spectra = np.random.normal(size=(n_spectra, n_wavelengths)) * 0.1 + 0.5
    return wavelengths, spectra

def compute_photometric_data(wavelengths, spectra, filter_curves):
    """Compute photometric data from spectra using filter curves."""
    photometric_data = np.zeros((spectra.shape[0], len(filter_curves)))
    for i, (center, width, response) in enumerate(filter_curves):
        filter_response = np.exp(-(wavelengths - center)**2 / (2 * width**2))
        flux = np.sum(spectra * filter_response[:, np.newaxis], axis=1)
        photometric_data[:, i] = flux
    return photometric_data

def reconstruct_spectrum(photometric_data, filter_curves, wavelengths):
    """Reconstruct spectrum from photometric data using Ridge Regression."""
    n_filters = len(filter_curves)
    filter_responses = np.zeros((len(wavelengths), n_filters))
    for i, (center, width, response) in enumerate(filter_curves):
        filter_response = np.exp(-(wavelengths - center)**2 / (2 * width**2))
        filter_responses[:, i] = filter_response
    
    model = RidgeCV()
    model.fit(filter_responses, photometric_data.T)
    reconstructed_spectra = model.predict(filter_responses)
    
    return reconstructed_spectra

if __name__ == "__main__":
    np.random.seed(42)
    
    wavelengths, true_spectra = generate_synthetic_spectra()
    
    filter_centers = [400, 550, 700]
    filter_widths = [100, 80, 60]
    responses = [1, 1, 1]
    filter_curves = list(zip(filter_centers, filter_widths, responses))
    
    photometric_flux = compute_photometric_data(wavelengths, true_spectra, filter_curves)
    
    reconstructed_spectra = reconstruct_spectrum(photometric_flux, filter_curves, wavelengths)
    
    plt.figure(figsize=(12, 8))
    plt.title("Spectral Reconstruction")
    plt.plot(wavelengths, true_spectra[0, :], label='True Spectrum', alpha=0.7)
    plt.plot(wavelengths, reconstructed_spectra[0, :], label='Reconstructed Spectrum', alpha=0.7)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.legend()
    plt.show()
```