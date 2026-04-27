
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
Okay, so I need to write a Python code that implements an algorithm for reconstructing a synthetic spectrum from photometric data. The user has specified some requirements, so I need to make sure I address each one carefully.

First, the code should define a spectral model. I'll probably use a set of basis functions, maybe Gaussian processes since they are flexible. Sci-kit learn has an RBF kernel which could work well for this.

Next, I need to generate synthetic spectra. I can create a wavelength grid using numpy's arange function. Then, I'll compute the flux for each wavelength using a sum of Gaussian functions with random parameters. This will give me the true spectra.

Then, I have to generate photometric data from these synthetic spectra. To do this, I'll define a set of bandpasses. Each bandpass is a transmission curve, which I can represent as a numpy array of wavelengths and transmissions. For each band, I'll compute the observed flux by integrating the product of the spectrum and the transmission curve. Since integration isn't necessary here because we're working on a discrete wavelength grid, I'll compute it as a dot product. Then, I'll add some noise to simulate measurement errors, probably using a normal distribution scaled by a small sigma like 0.05.

After generating the photometric data, I need to reconstruct the spectrum from it. For this, I'll set up an optimization problem where I minimize the difference between the photometric observations and the projections of the reconstructed spectrum onto each bandpass. Using scipy's minimize function with least squares as the method seems appropriate. I'll initialize the reconstructed spectrum with zeros and then perform the optimization.

I also need to plot the results. I'll use matplotlib to visualize the true spectrum, the reconstructed spectrum, and the photometric data. Plots should be clear, showing the wavelengths on the x-axis and flux on the y-axis. I'll include titles and labels to make them understandable.

Additionally, I must ensure that the code is well-structured with clear functions. I'll define functions for generating the synthetic spectrum, the bandpasses, computing the photometric data, and performing the reconstruction. This modular approach keeps the code organized and readable.

I should remember to import all necessary libraries at the beginning: numpy for arrays and math, scipy for the GP model and optimization, and matplotlib for plotting. Also, since the user wants only the code without any explanations, I'll avoid adding any comments beyond what's necessary for clarity within the code.

Putting it all together, I'll start by setting up the parameters, then generate the synthetic data, compute photometry, reconstruct, and finally plot everything. I'll make sure the code runs without errors and produces the desired plots.
</think>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize

def generate_synthetic_spectrum(wavelength_range, resolution):
    wavelengths = np.arange(wavelength_range[0], wavelength_range[1], resolution)
    n_points = len(wavelengths)
    n_components = 5
    true_flux = np.zeros(n_points)
    for i in range(n_components):
        amp = np.random.uniform(0.8, 1.2)
        mean = np.random.uniform(0.5, 2.5)
        std = np.random.uniform(0.1, 0.3)
        true_flux += amp * np.exp(-0.5 * (wavelengths - mean)**2 / std**2)
    return wavelengths, true_flux

def define_bandpasses(wavelengths):
    bands = {
        'band1': (np.array([400, 600]), np.array([1.0, 1.0])),
        'band2': (np.array([600, 800]), np.array([1.0, 1.0]))
    }
    for band in bands.values():
        if np.shape(band[0])[0] != np.shape(wavelengths)[0]:
            band[1] = np.interp(wavelengths, band[0], band[1])
    return bands

def compute_photometry(wavelengths, flux, bands):
    photometry = []
    for band in bands.values():
        trans = band[1]
        flux_dered = np.dot(flux, trans) / np.sum(trans)
        noise = np.random.normal(0, 0.05, size=len(flux))
        observed_flux = flux_dered + noise
        photometry.append(observed_flux)
    return photometry

def reconstruct_spectrum(wavelengths, photometry, bands):
    n_wavelengths = len(wavelengths)
    init_spectrum = np.zeros(n_wavelengths)
    
    def objective(spectrum):
        obj = 0
        for i, (band, phot) in enumerate(zip(bands.values(), photometry)):
            proj = np.dot(spectrum, band[1])
            obj += np.sum((proj - phot)**2)
        return obj
    
    result = minimize(objective, init_spectrum, method='least_squares')
    return wavelengths, result.x

# Main execution
if __name__ == '__main__':
    wavelengths, true_flux = generate_synthetic_spectrum((400, 800), 1.0)
    bands = define_bandpasses(wavelengths)
    photometry = compute_photometry(wavelengths, true_flux, bands)
    recon_wavelengths, recon_flux = reconstruct_spectrum(wavelengths, photometry, bands)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, true_flux, label='True Spectrum')
    plt.plot(recon_wavelengths, recon_flux, label='Reconstructed Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Spectral Reconstruction')
    plt.legend()
    plt.show()
```